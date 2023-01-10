#!/bin/bash
 
# --------------------------------------------------------------
### PART 1: Requests resources to run your job.
# --------------------------------------------------------------

### Optional. Set the job name
#SBATCH --job-name=iefc
### Optional. Set the output filename. SLURM reads %x as the job name and %j as the job ID
#SBATCH --output=%x-%j.out
### REQUIRED. Specify the PI group for this job
#SBATCH --account=douglase
### Optional. Request email when job begins and ends
#SBATCH --mail-type=ALL
### Optional. Specify email address to use for notification
#SBATCH --mail-user=kianmilani@arizona.edu
### REQUIRED. Set the partition for your job.
#SBATCH --partition=high_priority
### REQUIRED. Set the number of cores that will be used for this job.
#SBATCH --ntasks=32
### REQUIRED. Set the number of nodes
#SBATCH --nodes=1
### REQUIRED. Set the memory required for this job.
#SBATCH --mem=64gb
### REQUIRED. Set the number of GPUs to use
#SBATCH --gres=gpu:1

### REQUIRED. Specify the time required for this job, hhh:mm:ss
#SBATCH --time=48:00:00

# --------------------------------------------------------------
### PART 2: Executes bash commands to run your job
# --------------------------------------------------------------

source ~/.bashrc
conda activate poppy39-exp-env
cd ~/Documents/roman-cgi-iefc
python run_spcw_iefc_annular.py

