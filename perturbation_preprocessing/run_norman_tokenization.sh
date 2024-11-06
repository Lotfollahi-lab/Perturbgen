#make a date directory if it does not exist
#!/bin/bash
#BSUB -q normal # run CPU job
#BSUB -n 32 # number of cores
#BSUB -G teamtrynka # groupname for billing
#BSUB -cwd /lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/T_perturb/T_perturb # working directory
#BSUB -o logs/random_pairing_GF_tokenisation_%J.out # output file
#BSUB -e logs/random_pairing_GF_tokenisation_%J.err # error file
#BSUB -M 1500000  # RAM memory part 2. Default: 100MB
#BSUB -R 'select[mem>1500000] rusage[mem=1500000]' # RAM memory part 1. Default: 100MB
#BSUB -J random_pairing_GF_tokenisation # job name

# activate python environment
# source /lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/.petra_cuda12/bin/activate
# cwd=$(pwd)

echo "--- Start tokenisation ---"

export PYTHONPATH=/lustre/scratch126/cellgen/team205/bair/cg/T_perturb:$PYTHONPATH

python3 /lustre/scratch126/cellgen/team205/bair/perturbench/perturbench_data/GF_tokenization_pb.py \
--gene_filtering_mode 'all' \
--exclude_non_GF_genes Tru
--var_list good_coverage cell_type perturbation_type celltype perturbation nperts ngenes condition  \
--pairing_mode random \
--nproc 32 \