#!/bin/bash
#SBATCH --job-name="e2e baseline"
#SBATCH --time=8-00:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-40g:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=40000

nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"

python -c 'print ("Xavier weights initialization")'
 
python -c 'print ("Conda activated, starting copying data...")'

rsync -a /gpfs/space/projects/rally2023/rally-estonia-cropped-antialias /tmp

python -c 'print ("Data is copied. Running training...")'

python train.py --input-modality nvidia-camera --output-modality steering_angle --patience 10 --max-epochs 20 --model-name steering-angle --model-type pilotnet-conditional --dataset-folder /tmp/rally-estonia-cropped-antialias --wandb-project eestirally
