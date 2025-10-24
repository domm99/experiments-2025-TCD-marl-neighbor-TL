#!/bin/bash
#SBATCH --job-name=experiment-tl-tcd
#SBATCH --mail-type=ALL
#SBATCH --mail-user=davide.domini@unibo.it
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=31G
#SBATCH --output=output_exp1.txt
#SBATCH --chdir=/scratch.hpc/davide.domini/experiments-2025-TCD-marl-neighbor-TL
#SBATCH --gres=gpu:1

../venv/bin/python3 src/main.py
