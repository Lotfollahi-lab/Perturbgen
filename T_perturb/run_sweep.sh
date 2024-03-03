#date
date=$(date "+%Y-%m-%d")
#!/bin/bash
#BSUB -q gpu-huge # name of the partition to run job on (options: gpu-normal, gpu-huge, gpu-cellgeni-a100)
#BSUB -gpu 'mode=exclusive_process' # request for exclusive access to gpu
#BSUB -G teamtrynka # groupname for billing
#BSUB -cwd /lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/T_perturb/T_perturb # working directory
#BSUB -o logs/sweep/%J_$date_count.out # output file
#BSUB -e logs/sweep/%J_$date_count.err # error file
#BSUB -M 50000  # RAM memory part 2. Default: 100MB
#BSUB -R 'select[ngpus>0 && mem>50000] rusage[ngpus_physical=2.00,mem=50000] span[ptile=1]' # RAM memory part 1. Default: 100MB
#BSUB -W 12:00 # time for the job HH:MM:SS. Default: 1 min


# activate environment
source /lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/.tperturb/bin/activate
source ~/.bashrc
cwd=$(pwd)

export WANDB_DIR=$cwd/wandb
# run sweep
echo "--- Start sweep"
#paste wandb with sweep id
wandb agent k-ly/ttransformer_sweep/av9yl6i3
echo "--- Finished sweep"
