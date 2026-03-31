#make a date directory if it does not exist
#!/bin/bash
#BSUB -q normal # run CPU job
#BSUB -n 1 # number of cores
#BSUB -G cellulargenetics-priority # groupname for billing
#BSUB -cwd /lustre/scratch126/cellgen/lotfollahi/kl11 # working directory
#BSUB -o TRACE-reproducibility/logs/plt_perturb_res_%J_%I.out # output file
#BSUB -e TRACE-reproducibility/logs/plt_perturb_res_%J_%I.err # error file
#BSUB -M 40000  # RAM memory part 2. Default: 100MB
#BSUB -R 'select[mem>40000] rusage[mem=40000]' # RAM memory part 1. Default: 100MB
#BSUB -J plt_perturb_res_[1-5]%100 # job name

# activate python environment
source /nfs/team361/cytomeister/.cytomeister/bin/activate
cwd=$(pwd)

PERTURBED_GENE=$(sed -n "${LSB_JOBINDEX}p" $cwd/T_perturb/perturbgen/configs/eval/HSPC/all_genes.txt)
echo "[$(date)] Starting job index $LSB_JOBINDEX"
echo "Current gene: $PERTURBED_GENE"


echo "--- Start plotting"

python3 /lustre/scratch126/cellgen/lotfollahi/kl11/TRACE-reproducibility/HSPC/3.1_multiple_perturbation.py \
--path 'T_perturb/res/hspc/perturbation_5k' \
--perturbed_gene $PERTURBED_GENE \
--output_dir 'T_perturb/res/hspc/perturbation_5k_res' \
--p_val_adj_threshold 0.05 \
--logfc_threshold 0.25 \
--perturb_genes_file 'T_perturb/res/hspc/perturbation/intermediate_top250_rank_genes_groups.csv' \
--return_lin_reg_summary True
echo "--- Finished plotting"
