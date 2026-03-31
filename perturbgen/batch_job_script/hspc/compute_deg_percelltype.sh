#make a date directory if it does not exist
#!/bin/bash
#BSUB -q normal # run CPU job
#BSUB -n 1 # number of cores
#BSUB -G team361 # groupname for billing
#BSUB -cwd /lustre/scratch126/cellgen/lotfollahi/kl11 # working directory
#BSUB -o TRACE-reproducibility/logs/plt_perturb_res_%J_%I.out # output file
#BSUB -e TRACE-reproducibility/logs/plt_perturb_res_%J_%I.err # error file
#BSUB -M 25000  # RAM memory part 2. Default: 100MB
#BSUB -R 'select[mem>25000] rusage[mem=25000]' # RAM memory part 1. Default: 100MB
#BSUB -J plt_perturb_res_[1-6000]%50 # job name

# activate python environment
source /nfs/team361/cytomeister/.cytomeister/bin/activate
cwd=$(pwd)

PERTURBED_FILE=$(
  find "$cwd/T_perturb/res/hspc/perturbation_5k" -maxdepth 1 -type f \
  | sort \
  | sed -n "${LSB_JOBINDEX}p"
)
echo "[$(date)] Starting job index $LSB_JOBINDEX"
echo "Current gene: $PERTURBED_FILE"


echo "--- Start DEG computation"

python3 /lustre/scratch126/cellgen/lotfollahi/kl11/TRACE-reproducibility/HSPC/compute_deg_for_webbrowser.py \
--h5ad_path ${PERTURBED_FILE} \
--h5ad_perturbation_path 'T_perturb/res/hspc/perturbation_5k_res/summary_plots/20250723_leiden_pert_cls.h5ad' \
--output_dir '/lustre/scratch126/cellgen/lotfollahi/kl11/T_perturb/res/hspc/webbrowser'
echo "--- Finished DEG computation"
