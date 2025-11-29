"""
Example: Creating and Iterating Through Gene Lists

This example demonstrates how to create and use gene lists for perturbation experiments.
The gene names mentioned (NFE2, KLF1, FOSL1, etc.) are transcription factors important
for hematopoiesis and are included in the perturb_TF.txt file.
"""

from pathlib import Path
from typing import List


def create_gene_list_manually() -> List[str]:
    """
    Create a gene list manually in Python.
    
    This is useful when you want to define genes directly in your code.
    """
    genes = [
        'NFE2',
        'KLF1',
        'FOSL1',
        'LMO2',
        'SPI1',
        'GFI1B',
        'LDB1',
        'TAL1',
        'BCL11A',
        'RUNX1',
        'GATA1',
        'MYB',
    ]
    return genes


def read_gene_list_from_file(file_path: str) -> List[str]:
    """
    Read gene list from a text file.
    
    Parameters
    ----------
    file_path : str
        Path to the text file containing gene names (one per line)
    
    Returns
    -------
    List[str]
        List of gene names
    """
    with open(file_path, 'r') as f:
        genes = [line.strip() for line in f if line.strip()]
    return genes


def iterate_through_genes(genes: List[str]) -> None:
    """
    Example of iterating through a list of genes.
    
    Parameters
    ----------
    genes : List[str]
        List of gene names to process
    """
    print(f"Processing {len(genes)} genes:")
    for i, gene in enumerate(genes, 1):
        print(f"  {i}. {gene}")


def main():
    """Main example demonstrating different ways to work with gene lists."""
    
    print("=" * 60)
    print("Example 1: Creating a gene list manually")
    print("=" * 60)
    genes_manual = create_gene_list_manually()
    iterate_through_genes(genes_manual)
    
    print("\n" + "=" * 60)
    print("Example 2: Reading from existing file")
    print("=" * 60)
    # Path to the existing gene list file
    gene_file = Path(__file__).parent.parent / 'configs' / 'eval' / 'HSPC' / 'perturb_TF.txt'
    
    if gene_file.exists():
        genes_from_file = read_gene_list_from_file(str(gene_file))
        iterate_through_genes(genes_from_file)
    else:
        print(f"Gene file not found at: {gene_file}")
        print("Using manual list instead.")
    
    print("\n" + "=" * 60)
    print("Example 3: Using with perturbgen utilities")
    print("=" * 60)
    print("You can also use the built-in utility function:")
    print("")
    print("  from perturbgen.src.utils import read_gene_list")
    print("  genes = read_gene_list('configs/eval/HSPC/perturb_TF.txt')")
    print("  for gene in genes:")
    print("      # Process each gene")
    print("      pass")
    
    print("\n" + "=" * 60)
    print("Example 4: Using with PerturberTrainer")
    print("=" * 60)
    print("To use gene lists with the PerturberTrainer:")
    print("")
    print("  from perturbgen.Perturb.trainer import PerturberTrainer")
    print("  from perturbgen.src.utils import read_gene_list")
    print("")
    print("  genes_to_perturb = read_gene_list('configs/eval/HSPC/perturb_TF.txt')")
    print("  trainer = PerturberTrainer(")
    print("      genes_to_perturb=genes_to_perturb,")
    print("      perturbation_mode='mask',")
    print("      perturbation_sequence='src',")
    print("      # ... other parameters")
    print("  )")


if __name__ == '__main__':
    main()
