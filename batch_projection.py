import sys
## add the path to the sys path
sys.path.append('../stylegan2-ada-pytorch')

import copy
import os
from time import perf_counter

import click
import imageio
import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F

import dnnlib
import legacy

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import torchvision


def project_batch(
    G,
    targets: torch.Tensor,  # Shape [batch_size, C, H, W] and dynamic range [0,255]
    *,
    num_steps=1000,
    w_avg_samples=10000,
    initial_learning_rate=0.1,
    initial_noise_factor=0.05,
    lr_rampdown_length=0.25,
    lr_rampup_length=0.05,
    noise_ramp_length=0.75,
    regularize_noise_weight=1e5,
    verbose=False,
    device: torch.device,
    return_list=True
):
    batch_size = targets.shape[0]
    assert targets.shape[1:] == (G.img_channels, G.img_resolution, G.img_resolution), "Target images must match G output resolution."

    def logprint(*args):
        if verbose:
            print(*args)

    G = copy.deepcopy(G).eval().requires_grad_(False).to(device)

    # Compute w stats.
    logprint(f'Computing W midpoint and stddev using {w_avg_samples} samples...')
    z_samples = np.random.RandomState(123).randn(w_avg_samples, G.z_dim)
    w_samples = G.mapping(torch.from_numpy(z_samples).to(device), None)
    w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32)
    w_avg = np.mean(w_samples, axis=0, keepdims=True)
    w_std = (np.sum((w_samples - w_avg) ** 2) / w_avg_samples) ** 0.5

    # Setup noise inputs.
    noise_bufs = {name: buf for (name, buf) in G.synthesis.named_buffers() if 'noise_const' in name}

    # Load VGG16 feature detector.
    url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
    with dnnlib.util.open_url(url) as f:
        vgg16 = torch.jit.load(f).eval().to(device)

    # Features for target images.
    target_images = targets.to(device).to(torch.float32)
    if target_images.shape[2] > 256:
        target_images = F.interpolate(target_images, size=(256, 256), mode='area')
    target_features = vgg16(target_images, resize_images=False, return_lpips=True)

    # Initialize w_opt for batch.
    # w_opt = torch.tensor(w_avg, dtype=torch.float32, device=device, requires_grad=True).repeat(batch_size, 1, 1)
    # optimizer = torch.optim.Adam([w_opt] + list(noise_bufs.values()), betas=(0.9, 0.999), lr=initial_learning_rate)

    # Initialize w_opt for batch as a leaf tensor
    # This is a key modification to address the "non-leaf Tensor" error
    w_opt = torch.tensor(w_avg, dtype=torch.float32, device=device, requires_grad=True).repeat(batch_size, 1, 1)
    # To make it a leaf tensor, wrap it in nn.Parameter. However, since the direct repeat operation makes it non-leaf,
    # you should directly create it in the required shape
    w_opt = torch.nn.Parameter(w_opt)

    # Now w_opt is properly set up as a leaf tensor, and you can proceed to use it with your optimizer
    optimizer = torch.optim.Adam([w_opt] + list(noise_bufs.values()), betas=(0.9, 0.999), lr=initial_learning_rate)


    # Initialize w_out to track optimization paths for each image.
    w_out = torch.zeros([batch_size, num_steps] + list(w_opt.shape[1:]), dtype=torch.float32, device=device)

    for step in range(num_steps):
        # Adjust learning rate and noise factor according to the schedule.
        t = step / num_steps
        w_noise_scale = w_std * initial_noise_factor * max(0.0, 1.0 - t / noise_ramp_length) ** 2
        lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
        lr = initial_learning_rate * lr_ramp
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Add noise to w_opt and generate synthetic images.
        w_noise = torch.randn_like(w_opt) * w_noise_scale
        ws = (w_opt + w_noise).repeat([1, G.mapping.num_ws, 1])
        synth_images = G.synthesis(ws, noise_mode='const')

        # Downsample synth images if necessary.
        synth_images = (synth_images + 1) * (255 / 2)
        if synth_images.shape[2] > 256:
            synth_images = F.interpolate(synth_images, size=(256, 256), mode='area')

        # Compute features for synth images and calculate loss.
        synth_features = vgg16(synth_images, resize_images=False, return_lpips=True)
        # print(target_features.shape)
        # print(synth_features.shape)

        dist = (target_features - synth_features).square().sum(dim=1)  # Calculate dist per image in batch.
        # print(dist.shape)

        # Regularize noise.
        reg_loss = torch.tensor(0.0, device=device)
        for name, buf in noise_bufs.items():
            noise = buf[None, None, :, :]  # Expand dims to [1,1,H,W].
            while True:
                reg_loss += (noise * torch.roll(noise, shifts=1, dims=3)).mean() ** 2
                reg_loss += (noise * torch.roll(noise, shifts=1, dims=2)).mean() ** 2
                if noise.shape[2] <= 8:
                    break
                noise = F.avg_pool2d(noise, kernel_size=2)
        
        loss = dist.mean() + reg_loss * regularize_noise_weight  # Average dist loss over batch and add regularized noise.

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        logprint(f'step {step+1:>4d}/{num_steps}: dist {dist.mean().item():<4.2f} loss {loss.item():<5.2f}')

        # Save projected W for each optimization step.
        w_out[:, step] = w_opt.detach().clone()

        # Normalize noise.
        with torch.no_grad():
            for buf in noise_bufs.values():
                buf -= buf.mean()
                buf *= buf.square().mean().rsqrt()

    if not return_list:
        return w_opt.detach().clone()
    else:
        repeated_w_out = w_out.repeat([1, 1, G.mapping.num_ws, 1])
        return repeated_w_out  # Now w_out contains the optimized w paths for each image across the steps.




def run_projection_batch(
    network_pkl: str,
    target_folder: str,
    outdir: str,
    # save_video: bool,
    seed: int,
    num_steps: int,
    batch_size: int
):
    """Project given image to the latent space of pretrained network pickle.

    Examples:

    \b
    python projector.py --outdir=out --target=~/mytargetimg.png \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load networks.
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as fp:
        G = legacy.load_network_pkl(fp)['G_ema'].requires_grad_(False).to(device) # type: ignore

    # Load target images from directory.
    transform = transforms.Compose([
        # transforms.Resize((G.img_resolution, G.img_resolution)),  ## no need to resize
        transforms.ToTensor(),
    ])

    dataset = torchvision.datasets.ImageFolder(target_folder, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    os.makedirs(outdir, exist_ok=True)
    img_dir = os.path.join(outdir, 'images')
    os.makedirs(img_dir, exist_ok=True)\
    
    w_all = []

    for i, (images, _) in enumerate(dataloader):
    
        print (f'Processing batch {i+1}...')
        start_time = perf_counter()
        projected_w_steps = project_batch(
            G,
            targets=images*255,  ## rescale to [0, 255]
            num_steps=num_steps,
            device=device,
            verbose=True
        )
        print (f'Elapsed: {(perf_counter()-start_time):.1f} s')
        

        for j in range(batch_size):
            target_pil = Image.fromarray((images[j].permute(1, 2, 0).numpy() + 1) * 127.5, 'RGB')
            target_pil.save(f'{outdir}/target_{i*batch_size+j}.png')
            projected_w = projected_w_steps[j][-1]
            synth_image = G.synthesis(projected_w.unsqueeze(0), noise_mode='const')
            synth_image = (synth_image + 1) * (255/2)
            synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
            PIL.Image.fromarray(synth_image, 'RGB').save(f'{img_dir}/proj_{i*batch_size+j}.png')
            np.savez(f'{outdir}/projected_w_{i*batch_size+j}.npz', w=projected_w.unsqueeze(0).cpu().numpy())



def run_projection_batch_w_only(
    network_pkl: str,
    target_folder: str,
    outdir: str,
    # save_video: bool,
    seed: int,
    num_steps: int,
    batch_size: int
):
    """Project given image to the latent space of pretrained network pickle.

    Examples:

    \b
    python projector.py --outdir=out --target=~/mytargetimg.png \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load networks.
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as fp:
        G = legacy.load_network_pkl(fp)['G_ema'].requires_grad_(False).to(device) # type: ignore

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    dataset = torchvision.datasets.ImageFolder(target_folder, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    os.makedirs(outdir, exist_ok=True)
    img_dir = os.path.join(outdir, 'images')
    os.makedirs(img_dir, exist_ok=True)\
    
    w_all = []

    for i, (images, _) in enumerate(dataloader):
    
        print (f'Processing batch {i+1}...')
        start_time = perf_counter()
        projected_w_steps = project_batch(
            G,
            targets=images*255,
            num_steps=num_steps,
            device=device,
            verbose=True,
            return_list=False
        )
        print (f'Elapsed: {(perf_counter()-start_time):.1f} s')
        w_all.append(projected_w_steps.cpu().numpy())

	# check ponit
        if i % 10 == 0:
             w_checkpoint = np.concatenate(w_all, axis=0)
             print(f'save at batch {i}')
             np.savez(f'{outdir}/projected_w_checkponit.npz', w=w_checkpoint)

    w_all = np.concatenate(w_all, axis=0)
    print (w_all.shape)
    np.savez(f'{outdir}/projected_w_all.npz', w=w_all)


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    # run_projection_batch(network_pkl='../stylegan2-ada-pytorch/stylegan2-afhqv2-512x512.pkl',
    #                     #  target_folder='../datasets/afhqv2/train',
    #                     target_folder='../datasets/afhqv2/samll_batch_test',
    #                         outdir='out_batch',
    #                         num_steps = 300,
    #                         batch_size=3,
    #                         seed=303
    #                      ) # pylint: disable=no-value-for-parameter


    run_projection_batch_w_only(network_pkl='../projectoer_styleGAN2/stylegan2-afhqv2-512x512.pkl',
                        #  target_folder='../datasets/afhqv2/train',
                            target_folder='../datasets/afhqv2/train',
                            outdir='out_batch',
                            num_steps = 300,
                            batch_size=12,
                            seed=303
                         ) # pylint: disable=no-value-for-parameter
