#make a date directory if it does not exist
#!/bin/bash
#BSUB -q gpu-lotfollahi # name of the partition to run job on (options: gpu-normal, gpu-huge, gpu-lotfollahi)
#BSUB -gpu 'mode=exclusive_process:num=1' # request for exclusive access to gpu
#BSUB -n 32 # number of cores
#BSUB -R "span[ptile=32]"     # split X cores per host
#BSUB -G team361 # groupname for billing
#BSUB -cwd /lustre/scratch126/cellgen/team298/dv8/trace_paper/trace_final/T_perturb # working directory
#BSUB -o T_perturb/log/lps_generate_inter_s1.3k_%J.out # output file
#BSUB -e T_perturb/log/lps_generate_inter_s1.3k_%J.err # error file
#BSUB -M 250000  # RAM memory part 2. Default: 100MB
#BSUB -R 'select[mem>250000] rusage[mem=250000]' # RAM memory part 1. Default: 100MB
#BSUB -J lps_generate_inter_1.3k # job name

# activate pyenv
cwd=$(pwd)

RES_DIR="/lustre/scratch126/cellgen/team298/dv8/trace_paper/trace_final/T_perturb/cytomeister/results"
RES_NAME="lps/generate_6h_interpolation_opthp_ourMED_onST1_e6_b8_old_nompirun"
# if directory does not e
# echo create it with the name $RES_NAME
mkdir -p $RES_DIR/$RES_NAME
# Get the current timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
# copy the current script to the result directory
cp $0 $RES_DIR/$RES_NAME/4_run_val_generate_interpolation_$TIMESTAMP.sh
echo "Copying script to $RES_DIR/$RES_NAME/4_run_val_generate_interpolation_$TIMESTAMP.sh"
# run script
echo '--- Start computing model'

# interpolation

python3 /lustre/scratch126/cellgen/team298/dv8/trace_paper/trace_final/T_perturb/cytomeister/val.py \
--test_mode count \
--split False \
--splitting_mode stratified \
--generate True \
--output_dir $RES_DIR/$RES_NAME \
--ckpt_count_path '/lustre/scratch126/cellgen/team298/dv8/trace_paper/trace_final/T_perturb/cytomeister/results/lps/count_interpolation_ourMED_ws_on2k_e9_noSplit_nodropout_subset_cond/checkpoints/20250211_1446_cellgen_train_count_lr_0.001_wd_0.001_batch_16_zinb_tp_1-3_s_42_pos_time_pos_sin_m_cosine-epoch=07.ckpt' \
--src_dataset "/lustre/scratch126/cellgen/team298/dv8/trace_paper/trace_repo/T_perturb/tokenized_data/st1_ourMED_adib/dataset_hvg_subsetted_src/normal.dataset" \
--tgt_dataset_folder "/lustre/scratch126/cellgen/team298/dv8/trace_paper/trace_repo/T_perturb/tokenized_data/st1_ourMED_adib/dataset_hvg_subsetted_tgt" \
--src_adata "/lustre/scratch126/cellgen/team298/dv8/trace_paper/trace_repo/T_perturb/tokenized_data/st1_ourMED_adib/h5ad_pairing_1367_hvg_src/normal.h5ad" \
--tgt_adata_folder "/lustre/scratch126/cellgen/team298/dv8/trace_paper/trace_repo/T_perturb/tokenized_data/st1_ourMED_adib/h5ad_pairing_1367_hvg_tgt" \
--batch_size 8 \
--max_len 3500 \
--tgt_vocab_size 20274 \
--count_lr 0.001 \
--cellgen_lr 0.00001 \
--cellgen_wd 0.00001 \
--sequence_length 1367 \
--count_wd 0.001 \
--num_layers 6 \
--d_ff 64 \
--loss_mode zinb \
--n_workers 32 \
--pred_tps 2 \
--context_tps 1 3 \
--var_list cell_type_cellgen_harm donor_cellgen_harm time_after_LPS \
--encoder Transformer_encoder \
--encoder_path "/lustre/scratch126/cellgen/lotfollahi/av13/scmaskgit/output2/checkpoints/20250620_1508_cellgen_train_masking_lr_5e-05_wd_1e-06_batch_64_ptime_pos_sin_m_pow_tp_1-2-3_s_42-epoch=07.ckpt" \
--context_mode True \
--pos_encoding_mode time_pos_sin \
--mask_scheduler 'cosine' \
--num_node 1 \
--d_model 768
echo '--- Finished computing model'
