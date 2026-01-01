#!/bin/bash
#SBATCH --job-name=run_refined_7
#SBATCH --output=run_refined/train_refined_%j.out
#SBATCH --error=run_refined/train_refined_%j.err
#SBATCH --account=gpu-research
#SBATCH --partition=gpu-morgeva
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --ntasks=1

# Set environment variables
export PYTHONPATH=/home/morg/students/gottesman3/knowledge-analysis-suite/ReFinED/src
export HOME=/home/morg/students/gottesman3

echo "GPU 7"

# Run the Python script with the specified arguments
python /home/morg/students/gottesman3/knowledge-analysis-suite/ReFinED/run_refined.py \
  --gpu_id 7 \
  --n_gpus 8 \
  --max_workers 8 \
  --batch_size 8