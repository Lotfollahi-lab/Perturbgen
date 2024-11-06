import pickle
import torch
from difflib import get_close_matches
from transformers import AutoTokenizer


# Load the token-to-gene dictionary
with open('/lustre/scratch126/cellgen/team205/bair/perturbench/perturbench_data/norman/token_id_to_genename_hvg.pkl', 'rb') as file:
    token_to_gene_dict = pickle.load(file)


def find_closest_match(gene, gene_to_token_dict):
    """
    Find the closest match for a given gene name using difflib's get_close_matches.

    Args:
        gene (str): The gene name to find the closest match for.
        gene_to_token_dict (dict): Dictionary mapping gene names to token IDs.

    Returns:
        str or None: The closest matching gene name or None if no close match is found.
    """
    matches = get_close_matches(gene, gene_to_token_dict.keys(), n=1, cutoff=0.6)
    return matches[0] if matches else None


def get_token_ids_for_genes(genes, token_to_gene_dict):
    """
    Retrieve token IDs for a list of gene names or a single concatenated gene string.

    Args:
        genes (list of str or str): List of gene names or a single concatenated string of gene names.
        token_to_gene_dict (dict): Dictionary mapping token IDs to gene names.

    Returns:
        list of int: List of token IDs corresponding to the input gene names.
    """
    # Convert the dictionary to map gene names to token IDs for easy lookup
    gene_to_token = {v: k for k, v in token_to_gene_dict.items()}

    # If input is a string, split by '+' to handle combinations of genes
    if isinstance(genes, str):
        genes = genes.split('+')

    token_ids = []
    for gene in genes:
        if gene in gene_to_token:
            print(f"Match found for '{gene}'")
            token_ids.append(gene_to_token[gene])
        else:
            print(f"Warning: No match found for '{gene}'. Skipping this gene.")
    
    print(token_ids)
    return token_ids


def get_gene_embeddings(token_ids, model_wrapper):
    """
    Retrieve gene embeddings using a Geneformer model wrapper.

    Args:
        token_ids (list of int): List of token IDs to get embeddings for.
        model_wrapper (Geneformerwrapper): The Geneformer model wrapper to use for extracting embeddings.

    Returns:
        torch.Tensor: A tensor containing gene embeddings with shape (1, number_of_genes, embedding_dim).
    """
    # Convert token IDs to tensor input format
    if not token_ids:
        print("No valid token IDs found. Returning default embedding.")
        return torch.zeros((1, model_wrapper.model.config.hidden_size))

    # Ensure input_ids are properly shaped for BERT model
    input_ids = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)  # Adds a batch dimension
    attention_mask = torch.ones_like(input_ids)  # All tokens are valid

    # Use the Geneformerwrapper to get embeddings
    with torch.no_grad():  # Ensure no gradients are computed
        embeddings = model_wrapper(src_input_id=input_ids, src_attention_mask=attention_mask)

    return embeddings


def extract_gene_embeddings(target_genes, token_to_gene_dict=None, model_wrapper=None):
    """
    Extract gene embeddings for a list of target genes.

    Args:
        target_genes (list of str): List of target gene names to extract embeddings for.
        token_to_gene_dict (dict, optional): Dictionary mapping token IDs to gene names. If None, a default dictionary is loaded.
        model_wrapper (Geneformerwrapper, optional): The Geneformer model wrapper to use. If None, a default wrapper is used.

    Returns:
        torch.Tensor: A tensor containing gene embeddings with shape (1, number_of_genes, embedding_dim).
    """
    # Load defaults if necessary
    if token_to_gene_dict is None:
        with open('/lustre/scratch126/cellgen/team205/bair/perturbench/perturbench_data/norman/token_id_to_genename_hvg.pkl', 'rb') as file:
            token_to_gene_dict = pickle.load(file)

    # Get the IDs for the target genes
    token_ids = get_token_ids_for_genes(target_genes, token_to_gene_dict)
    # print(f"Target genes: {target_genes}")

    # Extract embeddings using Geneformerwrapper
    gene_embeddings = get_gene_embeddings(token_ids, model_wrapper)
    
    return gene_embeddings