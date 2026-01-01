#!/bin/bash
#SBATCH --job-name=preprocess_patch_redirects_part1
#SBATCH --output=logs/preprocess_patch_redirects_part1_%j.out
#SBATCH --error=logs/preprocess_patch_redirects_part1_%j.err
#SBATCH --account=gpu-research
#SBATCH --partition=killable
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=0
#SBATCH --ntasks=1

export PYTHONPATH=/home/morg/students/gottesman3/knowledge-analysis-suite/ReFinED/src:$PYTHONPATH
export HOME=/home/morg/students/gottesman3

echo "Preprocessing data..."

python src/refined/offline_data_generation/preprocess_all.py