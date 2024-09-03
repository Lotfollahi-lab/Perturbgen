#make a date directory if it does not exist
#!/bin/bash
#BSUB -q normal # run CPU job
#BSUB -n 32 # number of cores
#BSUB -G team361 # groupname for billing
#BSUB -cwd /lustre/scratch126/cellgen/team205/bair/CellGen_HK/T_perturb/T_perturb # working directory
#BSUB -o logs/masking_%J.out # output file
#BSUB -e logs/masking_%J.err # error file
#BSUB -M 100000  # RAM memory part 2. Default: 100MB
#BSUB -R 'select[mem>100000] rusage[mem=100000]' # RAM memory part 1. Default: 100MB
#BSUB -J GF_tokenisation # job name

# activate python environment
source /lustre/scratch126/cellgen/team361/hk11/cytoimmgen/.venv/bin/activate
cwd=$(pwd)

echo "--- Start tokenisation"
echo $cwd

python3 $cwd/pp/GF_tokenisation.py \
--dataset 'cytoimmgen' \
--gene_filtering_mode 'hvg' \
--var_list Cell_population Cell_type Time_point Age \
Sex batch Cell_culture_batch Phase \
Donor cell_pairing_index \
--pairing_mode stratified \
--nproc 32 \
--h5ad_path /lustre/scratch126/cellgen/team205/av13/PETRA/data/h5d_files/cytoimmgen.h5ad \