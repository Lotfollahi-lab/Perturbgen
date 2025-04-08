#make a date directory if it does not exist
#!/bin/bash
#BSUB -q normal # run CPU job
#BSUB -n 4 # number of cores
#BSUB -G team361 # groupname for billing
#BSUB -cwd /lustre/scratch126/cellgen/team361/kl11/t_generative/ # working directory
#BSUB -o T_perturb/T_perturb/logs/plt_perturb_res_RUNX1_%J.out # output file
#BSUB -e T_perturb/T_perturb/logs/plt_perturb_res_RUNX1_%J.err # error file
#BSUB -M 50000  # RAM memory part 2. Default: 100MB
#BSUB -R 'select[mem>50000] rusage[mem=50000]' # RAM memory part 1. Default: 100MB
#BSUB -J plt_perturb_res_RUNX1 # job name

# activate python environment
source /lustre/scratch126/cellgen/team361/kl11/t_generative/.cellgen_4096/bin/activate
cwd=$(pwd)

echo "--- Start plotting"

python3 $cwd/TRACE-reproducibility/HSPC/2.2.1_multiple_perturbation.py \
--perturbed_gene 'RUNX1' \
--p_perturbation 'T_perturb/T_perturb/plt/res/hspc/pbmc_median/perturbation/20250407-20:37_minference_adata_gRUNX1_ssrc_tmask.h5ad'

echo "--- Finished plotting"
