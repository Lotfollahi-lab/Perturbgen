#make a date directory if it does not exist
#!/bin/bash
#BSUB -q gpu-huge # name of the partition to run job on (options: gpu-normal, gpu-huge, gpu-lotfollahi)
#BSUB -gpu 'mode=exclusive_process:num=2:gmodel=NVIDIAA100_SXM4_80GB' # request for exclusive access to gpu
#BSUB -n 4 # number of cores
#BSUB -R "span[ptile=4]"     # split X cores per host
#BSUB -G team361 # groupname for billing
#BSUB -cwd /lustre/scratch126/cellgen/lotfollahi/kl11/T_perturb/cytomeister # working directory
#BSUB -o logs/lps_count_extrapolation_s100_%J.out # output file
#BSUB -e logs/lps_count_extrapolation_s100_%J.err # error file
#BSUB -M 50000  # RAM memory part 2. Default: 100MB
#BSUB -R 'select[mem>50000] rusage[mem=50000]' # RAM memory part 1. Default: 100MB
#BSUB -J lps_count_extrapolation_s100 # job name

# load cuda
module load cuda-12.1.1

# activate conda environment
source /nfs/team361/cytomeister/.cytomeister/bin/activate
# results directory
RES_DIR="/lustre/scratch126/cellgen/lotfollahi/kl11/T_perturb/cytomeister/res"
RES_NAME="lps/pbmc_median/extrapolation"
# # if directory does not exist, create it with the name $RES_NAME
mkdir -p $RES_DIR/$RES_NAME

# export WANDB_DIR=$cwd/wandb
# run script
echo "--- Start computing model"

# # extrapolation
python3 /lustre/scratch126/cellgen/lotfollahi/kl11/T_perturb/cytomeister/train.py \
--train_mode count \
--split False \
--splitting_mode stratified \
--output_dir $RES_DIR/$RES_NAME \
--ckpt_masking_path "T_perturb/T_perturb/plt/res/lps/pbmc_median/extrapolation/res/checkpoints/20250512_2131_cellgen_train_masking_lr_0.0001_wd_0.0001_batch_64_ptime_pos_sin_m_pow_tp_1-2_s_100-epoch=19.ckpt" \
--src_dataset "/lustre/scratch126/cellgen/team298/dv8/trace_paper/trace_final/T_perturb/T_perturb/pp/res/2k_hvg_ourMED_all_tps/dataset_2000_hvg_src/normal.dataset" \
--tgt_dataset_folder "/lustre/scratch126/cellgen/team298/dv8/trace_paper/trace_final/T_perturb/T_perturb/pp/res/2k_hvg_ourMED_all_tps/dataset_2000_hvg_tgt" \
--src_adata "/lustre/scratch126/cellgen/team298/dv8/trace_paper/trace_final/T_perturb/T_perturb/pp/res/2k_hvg_ourMED_all_tps/h5ad_pairing_2000_hvg_src/normal.h5ad" \
--tgt_adata_folder "/lustre/scratch126/cellgen/team298/dv8/trace_paper/trace_final/T_perturb/T_perturb/pp/res/2k_hvg_ourMED_all_tps/h5ad_pairing_2000_hvg_tgt" \
--mapping_dict_path "/lustre/scratch126/cellgen/team298/dv8/trace_paper/trace_final/T_perturb/T_perturb/pp/res/2k_hvg_ourMED_all_tps/token_id_to_genename_2000_hvg.pkl" \
--batch_size 64 \
--max_len 666 \
--epochs 5 \
--tgt_vocab_size 1990 \
--count_lr 0.001 \
--cellgen_lr 0.0001 \
--cellgen_wd 0.0001 \
--count_wd 0.0001 \
--count_dropout 0.1 \
--n_workers 4 \
--num_layers 6 \
--d_ff 64 \
--loss_mode zinb \
--pred_tps 1 2 \
--var_list cell_type_cellgen_harm donor_cellgen_harm time_after_LPS cell_pairing_index  \
--cond_list time_after_LPS \
--encoder scmaskgit \
--encoder_path "/lustre/scratch126/cellgen/lotfollahi/av13/scmaskgit/output2/checkpoints/20250620_1508_cellgen_train_masking_lr_5e-05_wd_1e-06_batch_64_ptime_pos_sin_m_pow_tp_1-2-3_s_42-epoch=04.ckpt" \
--add_cell_time False \
--use_positional_encoding False \
--context_mode True \
--pos_encoding_mode time_pos_sin \
--mask_scheduler 'pow' \
--num_node 1 \
--d_model 768 \
--seed 100 \
--use_weighted_sampler False

echo '--- Finished computing model'
