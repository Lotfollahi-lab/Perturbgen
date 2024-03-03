#date
date=$(date "+%Y-%m-%d")
#!/bin/bash
#BSUB -q gpu-cellgeni-a100 # name of the partition to run job on (options: gpu-normal, gpu-huge, gpu-cellgeni-a100)
#BSUB -gpu 'mode=exclusive_process' # request for exclusive access to gpu
#BSUB -G teamtrynka # groupname for billing
#BSUB -cwd /lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/T_perturb/T_perturb # working directory
#BSUB -o logs/count/%J_$date_count.out # output file
#BSUB -e logs/count/%J_$date_count.err # error file
#BSUB -M 50000  # RAM memory part 2. Default: 100MB
#BSUB -R 'select[ngpus>0 && mem>50000] rusage[ngpus_physical=2.00,mem=50000] span[ptile=1]' # RAM memory part 1. Default: 100MB
#BSUB -W 12:00 # time for the job HH:MM:SS. Default: 1 min


# activate conda environment
source ~/.bashrc
source /lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/.tperturb/bin/activate
cwd=$(pwd)

export WANDB_DIR=$cwd/wandb
# run script
echo "--- Start computing model"
# # Run python script for rna
python3 $cwd/train.py \
--train_mode masking \
--num_cells 0 \
--src_dataset "pp/res/dataset/cytoimmgen_tokenised_degs_stratified_pairing_0h.dataset" \
--tgt_dataset "pp/res/dataset/cytoimmgen_tokenised_degs_stratified_pairing_16h.dataset" \
--src_adata_folder "pp/res/h5ad_pairing/cytoimmgen_tokenisation_degs_stratified_pairing_0h.h5ad" \
--tgt_adata_folder "pp/res/h5ad_pairing/cytoimmgen_tokenisation_degs_stratified_pairing_16h.h5ad" \
--batch_size 512 \
--epochs 5 \
--lr 0.001 \
--weight_decay 0.001 \
--n_workers 16 \
--loss_mode zinb \
--dropout 0.0 \
echo "--- Finished computing model"
