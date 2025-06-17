#make a date directory if it does not exist
#!/bin/bash
#BSUB -q gpu-lotfollahi # name of the partition to run job on (options: gpu-normal, gpu-huge, gpu-cellgeni-a100)
#BSUB -gpu 'mode=exclusive_process:num=1' # request for exclusive access to gpu
#BSUB -n 16 # number of cores
#BSUB -G teamtrynka # groupname for billing
#BSUB -cwd /lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/T_perturb/T_perturb # working directory
#BSUB -o logs/eb_masking_inter_s100_%J.out # output file
#BSUB -e logs/eb_masking_inter_s100_%J.err # error file
#BSUB -M 20000  # RAM memory part 2. Default: 100MB
#BSUB -R 'select[mem>20000] rusage[mem=20000]' # RAM memory part 1. Default: 100MB
#BSUB -J eb_masking_inter_s0 # job name

# load cuda
module load cuda-12.1.1

# activate python environment
source /lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/.cellgen_4096/bin/activate
cwd=$(pwd)

# # export WANDB_DIR=$cwd/wandb
# # Run python script to PETRA
# echo '--- Start computing model'

# # ----------------- Create folder to save results and copy the script -----------------
RES_DIR="/lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/T_perturb/T_perturb/iclr"
RES_NAME="eb/interpolation"
# if directory does not exist, create it with the name $RES_NAME
mkdir -p $RES_DIR/$RES_NAME
# Get the current timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
# copy the current script to the result directory
cp $0 $RES_DIR/$RES_NAME/2_run_train_masking_GF_frozen_interpolation_$TIMESTAMP.sh
echo "Copying script to $RES_DIR/$RES_NAME/2_run_train_masking_GF_frozen_interpolation_$TIMESTAMP.sh"

# export WANDB_DIR=$cwd/wandb
# Run python script to PETRA
echo '--- Start computing model'

# # interpolation
# python3 $cwd/train.py \
python3 /lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/T_perturb/T_perturb/train.py \
--train_mode masking \
--split False \
--splitting_mode random \
--output_dir $RES_DIR/$RES_NAME/res \
--src_dataset './T_perturb/T_perturb/pp/res/eb/dataset_hvg_subsetted_src/Day 00-03.dataset' \
--tgt_dataset_folder './T_perturb/T_perturb/pp/res/eb/dataset_hvg_subsetted_tgt' \
--src_adata './T_perturb/T_perturb/pp/res/eb/h5ad_pairing_hvg_src/Day 00-03.h5ad' \
--tgt_adata_folder './T_perturb/T_perturb/pp/res/eb/h5ad_pairing_hvg_tgt' \
--mapping_dict_path  './T_perturb/T_perturb/pp/res/eb/token_id_to_genename_hvg.pkl' \
--batch_size 64 \
--max_len 270 \
--epochs 50 \
--tgt_vocab_size 1730 \
--cellgen_lr 0.001 \
--cellgen_wd 0.0001 \
--mlm_prob 0.15 \
--n_workers 16 \
--num_layers 3 \
--d_ff 32 \
--time_steps 1 2 4 \
--seed 100 \
--var_list Time_point \
--mode GF_frozen \
--positional_encoding sin_learnt \
--mask_scheduler 'cosine'

echo '--- Finished computing model'
