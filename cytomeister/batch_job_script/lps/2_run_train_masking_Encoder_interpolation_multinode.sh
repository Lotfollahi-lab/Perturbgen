#make a date directory if it does not exist
#!/bin/bash
#BSUB -q gpu-parallel # name of the partition to run job on (options: gpu-normal, gpu-huge, gpu-lotfollahi)
#BSUB -gpu 'mode=exclusive_process:num=8' # request for exclusive access to gpu
#BSUB -n 32 # number of cores
#BSUB -R "span[ptile=16]"     # split X cores per host
#BSUB -G team361 # groupname for billing
#BSUB -cwd /lustre/scratch126/cellgen/team298/dv8/trace_paper/trace_final/T_perturb # working directory
#BSUB -o T_perturb/log/interpolation_6h_out_opt_hparam_2k%J.out # output file
#BSUB -e T_perturb/log/interpolation_6h_out_opt_hparam%J_2k.err # error file
#BSUB -M 140000  # RAM memory part 2. Default: 100MB
#BSUB -R 'select[mem>140000] rusage[mem=140000]' # RAM memory part 1. Default: 100MB
#BSUB -J interpolation # job name

set -eo pipefail

# initialize the module system
. /usr/share/modules/init/bash
module load ISG/openmpi
module load cuda-12.1.1

export NCCL_IB_HCA=^mlx5_bond
export NCCL_DEBUG=INFO
export NCCL_DEBUG_FILE=/lustre/scratch126/cellgen/team298/dv8/trace_paper/trace_final/T_perturb/cytomeister/log/nccl.%h.%p
export NCCL_IB_DISABLE=0  # disable infiniband to prevent annoying errors
export UCX_IB_MLX5_DEVX=n

# Get the number of hosts and GPUs from LSF
NUM_HOSTS=$(sed 's/ /\n/g' <<< $LSB_HOSTS  | sort | uniq | wc -l)
NUM_GPUS=$(bjobs -noheader -o 'gpu_num' "$LSB_JOBID")
GPU_PER_HOST=$((NUM_GPUS / NUM_HOSTS))

# activate pyenv
# source /lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/.cellgen_4096/bin/activate
cwd=$(pwd)
## source /lustre/scratch126/cellgen/lotfollahi/av13/scmaskgit/.venv/bin/activate

# export WANDB_DIR=$cwd/wandb
# run script
echo "--- Start computing model"


# # ----------------- Create folder to save results and copy the script -----------------
RES_DIR="/lustre/scratch126/cellgen/team298/dv8/trace_paper/trace_final/T_perturb/cytomeister/results"
RES_NAME="lps/interpolation_2k_alltps_cond_celltype"
# if directory does not exist, create it with the name $RES_NAME
mkdir -p $RES_DIR/$RES_NAME
# Get the current timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
# copy the current script to the result directory
cp $0 $RES_DIR/$RES_NAME/2_run_train_masking_GF_frozen_interpolation_$TIMESTAMP_$SLURM_JOB_ID.sh
echo "Copying script to $RES_DIR/$RES_NAME/2_run_train_masking_GF_frozen_interpolation_$TIMESTAMP.sh"

# export WANDB_DIR=$cwd/wandb
# Run python script to PETRA
echo '--- Start computing model'

# # interpolation
mpirun \
    -n ${NUM_GPUS} \
    --map-by "ppr:${GPU_PER_HOST}:node" \
    --display-allocation \
    python3 /lustre/scratch126/cellgen/team298/dv8/trace_paper/trace_final/T_perturb/cytomeister/train.py \
    --train_mode masking \
    --split False \
    --splitting_mode stratified \
    --split_obs cell_type_cellgen_harm \
    --output_dir $RES_DIR/$RES_NAME \
    --src_dataset "/lustre/scratch126/cellgen/team298/dv8/trace_paper/trace_final/T_perturb/tokenized_data/2k_hvg_ourMED_all_tps/dataset_2000_hvg_src/normal.dataset" \
    --tgt_dataset_folder "/lustre/scratch126/cellgen/team298/dv8/trace_paper/trace_final/T_perturb/tokenized_data/2k_hvg_ourMED_all_tps/dataset_2000_hvg_tgt" \
    --src_adata "/lustre/scratch126/cellgen/team298/dv8/trace_paper/trace_final/T_perturb/tokenized_data/2k_hvg_ourMED_all_tps/h5ad_pairing_2000_hvg_src/normal.h5ad" \
    --tgt_adata_folder "/lustre/scratch126/cellgen/team298/dv8/trace_paper/trace_final/T_perturb/tokenized_data/2k_hvg_ourMED_all_tps/h5ad_pairing_2000_hvg_tgt" \
    --mapping_dict_path "/lustre/scratch126/cellgen/team298/dv8/trace_paper/trace_final/T_perturb/tokenized_data/2k_hvg_ourMED_all_tps/token_id_to_genename_2000_hvg.pkl" \
    --batch_size 64 \
    --max_len 666 \
    --epochs 26 \
    --tgt_vocab_size 20274 \
    --cellgen_lr 0.0001 \
    --cellgen_wd 0.0001 \
    --mlm_prob 0.3 \
    --n_workers 16 \
    --num_layers 6 \
    --d_ff 32 \
    --pred_tps 1 3 \
    --var_list cell_type_cellgen_harm donor_cellgen_harm time_after_LPS \
    --cond_list cell_type_cellgen_harm \
    --encoder scmaskgit \
    --encoder_path "/lustre/scratch126/cellgen/lotfollahi/av13/scmaskgit/output2/checkpoints/20250620_1508_cellgen_train_masking_lr_5e-05_wd_1e-06_batch_64_ptime_pos_sin_m_pow_tp_1-2-3_s_42-epoch=07.ckpt" \
    --seed 42 \
    --context_mode True \
    --pos_encoding_mode time_pos_sin \
    --mask_scheduler 'cosine' \
    --d_model 768 \
    --num_node 2 \
    --d_model 768

echo '--- Finished computing model'
