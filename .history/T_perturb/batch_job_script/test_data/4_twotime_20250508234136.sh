#make a date directory if it does not exist
#!/bin/bash
#BSUB -q gpu-lotfollahi # name of the partition to run job on (options: gpu-normal, gpu-huge, gpu-lotfollahi)
#BSUB -gpu 'mode=exclusive_process:num=1' # request for exclusive access to gpu, # request for exclusive access to gpu, 'mode=shared:num=2:gmem=10000'
#BSUB -n 32 # number of cores
#BSUB -R "span[ptile=32]"     # split X cores per host
#BSUB -G team361 # groupname for billing
#BSUB -cwd /lustre/scratch126/cellgen/team298/dv8/trace_paper/trace_final/T_perturb # working directory
#BSUB -o T_perturb/logs/val_pred_inter_2k_%J.out # output file
#BSUB -e T_perturb/logs/val_pred_inter_2k_%J.err # error file
#BSUB -M 50000  # RAM memory part 2. Default: 100MB
#BSUB -R 'select[mem>50000] rusage[mem=50000]' # RAM memory part 1. Default: 100MB
#BSUB -J val_pred_inter_1.3k # job name

# activate pyenv
cwd=$(pwd)

RES_DIR="/lustre/scratch126/cellgen/team298/dv8/trace_paper/trace_final/T_perturb/T_perturb/results"
RES_NAME="lps/count_prediction_valS1"
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


python3 /lustre/scratch126/cellgen/team298/dv8/trace_paper/trace_final/T_perturb/T_perturb/val.py \
--test_mode count \
--split False \
--splitting_mode stratified \
--generate False \
--output_dir $RES_DIR/$RES_NAME/res \
--ckpt_count_path '/lustre/scratch126/cellgen/team298/dv8/trace_paper/trace_final/T_perturb/T_perturb/results/lps/count_valS1_ourMED/checkpoints/20250508_1311_cellgen_train_count_lr_0.001_wd_0.001_batch_64_zinb_tp_1_s_42_pos_time_pos_sin_m_pow-epoch=09.ckpt' \
--src_dataset "/lustre/scratch126/cellgen/team298/dv8/trace_paper/trace_final/T_perturb/T_perturb/pp/res/validation_TrainDataS1_2kHVG_ourMED/dataset_2000_hvg_src/early.dataset" \
--tgt_dataset_folder "/lustre/scratch126/cellgen/team298/dv8/trace_paper/trace_final/T_perturb/T_perturb/pp/res/validation_TrainDataS1_2kHVG_ourMED/dataset_2000_hvg_tgt" \
--src_adata "/lustre/scratch126/cellgen/team298/dv8/trace_paper/trace_final/T_perturb/T_perturb/pp/res/validation_TrainDataS1_2kHVG_ourMED/h5ad_pairing_2000_hvg_src/early.h5ad" \
--tgt_adata_folder "/lustre/scratch126/cellgen/team298/dv8/trace_paper/trace_final/T_perturb/T_perturb/pp/res/validation_TrainDataS1_2kHVG_ourMED/h5ad_pairing_2000_hvg_tgt" \
--mapping_dict_path "/lustre/scratch126/cellgen/team298/dv8/trace_paper/trace_final/T_perturb/T_perturb/pp/res/validation_TrainDataS1_2kHVG_ourMED/token_id_to_genename_2000_hvg.pkl" \
--batch_size 64 \
--max_len 1175 \
--tgt_vocab_size 2100 \
--count_lr 0.001 \
--cellgen_lr 0.0001 \
--cellgen_wd 0.0001 \
--sequence_length 150 \
--count_wd 0.001 \
--num_layers 6 \
--d_ff 32 \
--loss_mode zinb \
--n_workers 32 \
--pred_tps 1 \
--var_list annotation_simplified target pairing replicate cell_pairing_index \
--encoder scmaskgit \
--encoder_path "/lustre/scratch126/cellgen/team361/av13/scmaskgit/scmaskgit/output3/checkpoints/20250113_1104_cellgen_train_masking_lr_5e-05_wd_1e-06_batch_64_ptime_pos_sin_m_pow_tp_1-2-3_s_42-epoch=06.ckpt" \
--add_cell_time False \
--use_positional_encoding False \
--layer_norm True \
--context_mode True \
--pos_encoding_mode time_pos_sin \
--mask_scheduler 'pow' \
--iterations 20 \
--n_samples 3 \
--num_node 1 \
--d_model 768 \

echo '--- Finished computing model'

#--encoder scmaskgit \
#--encoder_path "/lustre/scratch126/cellgen/team361/av13/scmaskgit/scmaskgit/output3/checkpoints/20250113_1104_cellgen_train_masking_lr_5e-05_wd_1e-06_batch_64_ptime_pos_sin_m_pow_tp_1-2-3_s_42-epoch=06.ckpt" \
# > T_perturb/logs/valid_count_pred_trainS1_$(date +%Y%m%d_%H%M%S).out \
# 2> T_perturb/logs/valid_count_pred_trainS1_$(date +%Y%m%d_%H%M%S).err