#make a date directory if it does not exist
#!/bin/bash
#BSUB -q normal # run CPU job
#BSUB -n 16 # number of cores
#BSUB -G teamtrynka # groupname for billing
#BSUB -cwd /lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/T_perturb/T_perturb # working directory
#BSUB -o logs/GF_tokenisation_mnc_%J.out # output file
#BSUB -e logs/GF_tokenisation_mnc_%J.err # error file
#BSUB -M 100000  # RAM memory part 2. Default: 100MB
#BSUB -R 'select[mem>100000] rusage[mem=100000]' # RAM memory part 1. Default: 100MB
#BSUB -J GF_tokenisation_mnc # job name

# activate python environment
source /lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/.petra_cuda12/bin/activate
cwd=$(pwd)

echo "--- Start tokenisation"

python3 $cwd/pp/GF_tokenisation.py \
--h5ad_path './data/20241026_HSPC/mnc.h5ad' \
--dataset mnc \
--gene_filtering_mode hvg \
--var_list assignment_id sex tissue phase\
celltype_v2 donor_tissue diff_state dataset\
 cell_pairing_index \
--pairing_mode mapping \
--nproc 16 \
--reference_time stem \
--time_point_order stem intermediate terminal \
