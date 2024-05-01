
#make a date directory if it does not exist
#!/bin/bash
#BSUB -q gpu-lotfollahi # name of the partition to run job on (options: gpu-normal, gpu-huge, gpu-cellgeni-a100)
#BSUB -gpu 'mode=exclusive_process:num=4' # request for exclusive access to gpu
#BSUB -n 32 # number of cores
#BSUB -G teamtrynka # groupname for billing
#BSUB -cwd /lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/T_perturb/T_perturb # working directory
#BSUB -o logs/masking_%J.out # output file
#BSUB -e logs/masking_%J.err # error file
#BSUB -M 50000  # RAM memory part 2. Default: 100MB
#BSUB -R 'select[mem>50000] rusage[mem=50000]' # RAM memory part 1. Default: 100MB
#BSUB -J eb_masking # job name
#BSUB -W 12:00 # time for the job HH:MM:SS. Default: 1 min

# load cuda
module load cuda-12.1.1

# activate conda environment
source /lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/.petra_cuda12/bin/activate
cwd=$(pwd)

export WANDB_DIR=$cwd/wandb
# Run python script to PETRA
echo "--- Start computing model"

python3 $cwd/train.py \
--train_mode masking \
--split True \
--splitting_mode random \
--generate False \
--src_dataset "./T_perturb/T_perturb/pp/res/eb/dataset_all_src/eb_all_Day 00-03.dataset" \
--tgt_dataset_folder "./T_perturb/T_perturb/pp/res/eb/dataset_all_tgt" \
--src_adata "./T_perturb/T_perturb/pp/res/eb/h5ad_pairing_all_src/eb_all_Day 00-03.h5ad" \
--tgt_adata_folder "./T_perturb/T_perturb/pp/res/eb/h5ad_pairing_all_tgt" \
--mapping_dict_path  "./T_perturb/T_perturb/pp/res/eb/token_id_to_genename_all.pkl" \
--batch_size 64 \
--max_len 2048 \
--epochs 100 \
--tgt_vocab_size 15280 \
--petra_lr 0.001 \
--count_lr 0.0005 \
--petra_wd 0.001 \
--count_wd 0.001 \
--mlm_prob 0.15 \
--n_workers 64 \
--loss_mode nb \
--time_steps 1 2 3 \
--var_list Time_point \

echo "--- Finished computing model"
