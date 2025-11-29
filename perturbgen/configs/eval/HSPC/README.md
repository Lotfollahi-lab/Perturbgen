# HSPC Gene Lists

This directory contains gene lists used for perturbation experiments on HSPC (Hematopoietic Stem and Progenitor Cells) data.

## Gene List Files

- **perturb_TF.txt**: A curated list of 12 transcription factors important for hematopoiesis:
  - NFE2, KLF1, FOSL1, LMO2, SPI1, GFI1B, LDB1, TAL1, BCL11A, RUNX1, GATA1, MYB

- **perturb_stem.txt**: A list of 86 genes associated with stem cell identity and differentiation

- **genes_to_perturb.txt**: A list of 10 genes for general perturbation experiments

## Usage

### Reading Gene Lists in Python

You can use the `read_gene_list` utility function to load gene lists:

```python
from perturbgen.src.utils import read_gene_list

# Load transcription factors for perturbation
genes_to_perturb = read_gene_list('perturbgen/configs/eval/HSPC/perturb_TF.txt')

# Iterate through the genes
for gene in genes_to_perturb:
    print(f"Processing perturbation for gene: {gene}")
```

### Using Gene Lists with PerturberTrainer

```python
from perturbgen.Perturb.trainer import PerturberTrainer
from perturbgen.src.utils import read_gene_list

# Load gene list
genes_to_perturb = read_gene_list('perturbgen/configs/eval/HSPC/perturb_TF.txt')

# Use with trainer
trainer = PerturberTrainer(
    genes_to_perturb=genes_to_perturb,
    perturbation_mode='mask',
    perturbation_sequence='src',
    # ... other parameters
)
```

### Creating Your Own Gene List

To create your own gene list file:

1. Create a text file (e.g., `my_genes.txt`)
2. Add one gene name per line:
   ```
   GENE1
   GENE2
   GENE3
   ```
3. Load it using `read_gene_list('path/to/my_genes.txt')`

### Using Gene Lists in Shell Scripts

Gene lists can also be used in batch job arrays to iterate through genes:

```bash
# Read a specific gene based on job array index
PERTURBED_GENE=$(sed -n "${LSB_JOBINDEX}p" perturb_TF.txt)

# Process the gene
python3 val.py --config config.yaml --gene $PERTURBED_GENE
```

See `batch_job_script/hspc/3_perturb_job_array_mask.sh` for a complete example.
