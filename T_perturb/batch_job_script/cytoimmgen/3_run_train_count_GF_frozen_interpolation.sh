#make a date directory if it does not exist
#!/bin/bash
#BSUB -q gpu-lotfollahi # name of the partition to run job on (options: gpu-normal, gpu-huge, gpu-lotfollahi)
#BSUB -gpu 'mode=exclusive_process:num=2' # request for exclusive access to gpu
#BSUB -n 8 # number of cores
#BSUB -R "span[ptile=8]"     # split X cores per host
#BSUB -G team361 # groupname for billing
#BSUB -cwd /lustre/scratch126/cellgen/team298/dv8/trace_paper/trace_final/T_perturb # working directory
#BSUB -o T_perturb/logs/tcell_count_intrapolation_out_%J.out # output file
#BSUB -e T_perturb/logs/tcell_count_intrapolation_out_%J.err # error file
#BSUB -M 50000  # RAM memory part 2. Default: 100MB
#BSUB -R 'select[mem>50000] rusage[mem=50000]' # RAM memory part 1. Default: 100MB
#BSUB -J tcell_count_intra # job name

# load cuda
#module load cuda-12.1.1

# activate conda environment
#source /lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/.cellgen_4096/bin/activate
cwd=$(pwd)

export WANDB_DIR=$cwd/wandb
# Run python script to train count decoder
echo "--- Start computing model"

# ----------------- Create folder to save results and copy the script -----------------
RES_DIR="/lustre/scratch126/cellgen/team298/dv8/trace_paper/trace_final/T_perturb/T_perturb/results"
RES_NAME="cytoimmgen/context"
# if directory does not e
echo create it with the name $RES_NAME
mkdir -p $RES_DIR/$RES_NAME
# Get the current timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
# copy the current script to the result directory
cp $0 $RES_DIR/$RES_NAME/3_run_train_count_GF_frozen_interpolation_$TIMESTAMP.sh
echo "Copying script to $RES_DIR/$RES_NAME/3_run_train_count_GF_frozen_interpolation_$TIMESTAMP.sh"

# ----------------- Interpolation -----------------
# python3 $cwd/train.py \
python3 /lustre/scratch126/cellgen/team298/dv8/trace_paper/trace_final/T_perturb/T_perturb/train.py \
--train_mode count \
--split False \
--splitting_mode stratified \
--output_dir $RES_DIR/$RES_NAME/res \
--ckpt_masking_path "/lustre/scratch126/cellgen/team298/dv8/trace_paper/trace_final/T_perturb/T_perturb/results/cytoimmgen/pbmc_median/interpolation/checkpoints/20250427_1745_cellgen_train_masking_lr_1e-05_wd_1e-05_batch_64_ptime_pos_sin_m_pow_tp_1-3_s_100-epoch=09.ckpt" \
--src_dataset "/lustre/scratch126/cellgen/team361/kl11/t_generative/T_perturb/T_perturb/pp/res/cytoimmgen_pbmc_median/dataset_2000_hvg_src/0h.dataset" \
--tgt_dataset_folder "/lustre/scratch126/cellgen/team361/kl11/t_generative/T_perturb/T_perturb/pp/res/cytoimmgen_pbmc_median/dataset_2000_hvg_tgt" \
--src_adata "/lustre/scratch126/cellgen/team361/kl11/t_generative/T_perturb/T_perturb/pp/res/cytoimmgen_pbmc_median/h5ad_pairing_2000_hvg_src/0h.h5ad" \
--tgt_adata_folder "/lustre/scratch126/cellgen/team361/kl11/t_generative/T_perturb/T_perturb/pp/res/cytoimmgen_pbmc_median/h5ad_pairing_2000_hvg_tgt" \
--mapping_dict_path  "/lustre/scratch126/cellgen/team361/kl11/t_generative/T_perturb/T_perturb/pp/res/cytoimmgen_pbmc_median/token_id_to_genename_2000_hvg.pkl" \
--batch_size 64 \
--max_len 400 \
--epochs 2 \
--tgt_vocab_size 1360 \
--cellgen_lr 0.0001 \
--count_lr 0.005 \
--cellgen_wd 0.0001 \
--count_wd 0.001  \
--mlm_prob 0.15 \
--n_workers 32 \
--d_ff 64 \
--num_layers 6 \
--loss_mode zinb \
--condition_keys Cell_culture_batch \
--cond_list Time_point \
--pred_tps 1 3 \
--var_list Cell_population Cell_type Time_point Donor \
--encoder scmaskgit \
--encoder_path "/lustre/scratch126/cellgen/team361/av13/scmaskgit/scmaskgit/output3/checkpoints/20250113_1104_cellgen_train_masking_lr_5e-05_wd_1e-06_batch_64_ptime_pos_sin_m_pow_tp_1-2-3_s_42-epoch=06.ckpt" \
--pos_encoding_mode time_pos_sin \
--context_mode True \
--seed 42 \
--mask_scheduler 'pow' \
--d_model 768
echo "--- Finished computing model"
