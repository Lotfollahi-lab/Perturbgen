#make a date directory if it does not exist
#!/bin/bash
#BSUB -q gpu-lotfollahi # name of the partition to run job on (options: gpu-normal, gpu-huge, gpu-lotfollahi)
#BSUB -gpu 'mode=exclusive_process:num=2' # request for exclusive access to gpu
#BSUB -n 8 # number of cores
#BSUB -R "span[ptile=8]"     # split X cores per host
#BSUB -G team361 # groupname for billing
#BSUB -cwd /lustre/scratch126/cellgen/team298/dv8/trace_paper/trace_final/T_perturb # working directory
#BSUB -o T_perturb/logs/count_interpolation_6h_out_%J.out # output file
#BSUB -e T_perturb/logs/count_interpolation_6h_out_%J.err # error file
#BSUB -M 150000  # RAM memory part 2. Default: 100MB
#BSUB -R 'select[mem>150000] rusage[mem=150000]' # RAM memory part 1. Default: 100MB
#BSUB -J count_interpolation # job name

# activate pyenv
# source /lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/.cellgen_4096/bin/activate
cwd=$(pwd)
## source /lustre/scratch126/cellgen/team361/av13/scmaskgit/.venv/bin/activate

# export WANDB_DIR=$cwd/wandb
# run script
echo "--- Start computing model"


# # ----------------- Create folder to save results and copy the script -----------------
RES_DIR="/lustre/scratch126/cellgen/team298/dv8/trace_paper/trace_final/T_perturb/T_perturb/results"
RES_NAME="lps/count_interpolation_ourMED_ws_on2k_alltps_ns3_withval"
# if directory does not exist, create it with the name $RES_NAME
mkdir -p $RES_DIR/$RES_NAME
# Get the current timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
# copy the current script to the result directory
cp $0 $RES_DIR/$RES_NAME/3_run_train_count_interpolation_$TIMESTAMP_$SLURM_JOB_ID.sh
echo "Copying script to $RES_DIR/$RES_NAME/3_run_train_count_interpolation_$TIMESTAMP.sh"

# export WANDB_DIR=$cwd/wandb
# Run python script to PETRA
echo '--- Start computing model'

# # interpolation
python3 /lustre/scratch126/cellgen/team298/dv8/trace_paper/trace_final/T_perturb/T_perturb/train.py \
--train_mode count \
--split True \
--splitting_mode stratified \
--output_dir $RES_DIR/$RES_NAME \
--ckpt_masking_path "/lustre/scratch126/cellgen/team298/dv8/trace_paper/trace_final/T_perturb/T_perturb/results/lps/interpolation_2k_all_tps_cond_celltype/res/checkpoints/20250216_1825_cellgen_train_masking_lr_0.0001_wd_0.0001_batch_64_ptime_pos_sin_m_cosine_tp_1-2-3_s_42-epoch=15.ckpt" \
--src_dataset "/lustre/scratch126/cellgen/team298/dv8/trace_paper/trace_final/T_perturb/T_perturb/pp/res/2k_hvg_ourMED_all_tps/dataset_2000_hvg_src/normal.dataset" \
--tgt_dataset_folder "/lustre/scratch126/cellgen/team298/dv8/trace_paper/trace_final/T_perturb/T_perturb/pp/res/2k_hvg_ourMED_all_tps/dataset_2000_hvg_tgt" \
--src_adata "/lustre/scratch126/cellgen/team298/dv8/trace_paper/trace_final/T_perturb/T_perturb/pp/res/2k_hvg_ourMED_all_tps/h5ad_pairing_2000_hvg_src/normal.h5ad" \
--tgt_adata_folder "/lustre/scratch126/cellgen/team298/dv8/trace_paper/trace_final/T_perturb/T_perturb/pp/res/2k_hvg_ourMED_all_tps/h5ad_pairing_2000_hvg_tgt" \
--mapping_dict_path "/lustre/scratch126/cellgen/team298/dv8/trace_paper/trace_final/T_perturb/T_perturb/pp/res/2k_hvg_ourMED_all_tps/token_id_to_genename_2000_hvg.pkl" \
--batch_size 16 \
--max_len 666 \
--epochs 16 \
--tgt_vocab_size 20274 \
--count_lr 0.001 \
--cellgen_lr 0.0001 \
--cellgen_wd 0.0001 \
--count_wd 0.001 \
--mlm_prob 0.30 \
--n_workers 32 \
--num_layers 6 \
--d_ff 32 \
--loss_mode zinb \
--pred_tps 1 2 3 \
--var_list cell_type_cellgen_harm donor_cellgen_harm time_after_LPS \
--cond_list cell_type_cellgen_harm \
--encoder scmaskgit \
--add_cell_time False \
--d_condc 64 \
--d_condt 768 \
--count_dropout 0.1 \
--use_positional_encoding False \
--layer_norm True \
--context_mode True \
--encoder_path "/lustre/scratch126/cellgen/lotfollahi/av13/scmaskgit/output2/checkpoints/20250620_1508_cellgen_train_masking_lr_5e-05_wd_1e-06_batch_64_ptime_pos_sin_m_pow_tp_1-2-3_s_42-epoch=04.ckpt" \
--pos_encoding_mode time_pos_sin \
--mask_scheduler 'cosine' \
--num_node 1 \
--d_model 768

echo '--- Finished computing model'

