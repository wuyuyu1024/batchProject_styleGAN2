#!/bin/bash

#SBATCH --job-name=batch_projection             # Job name
#SBATCH --time=40:00:00                # Time limit: 30 hours

python batch_projection.py

