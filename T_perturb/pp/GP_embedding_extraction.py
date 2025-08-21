
import openai
from openai import OpenA
import numpy as np
import pickle
from tqdm import tqdm
import re


# Check gene programs from GO_Biological_Process_2025 and extract genes associated with each program.
filepath = "/lustre/scratch126/cellgen/lotfollahi/dv8/cellgen/GO_Biological_Process_2025_Delshad.txt"

gp_to_genes = {}
with open(filepath, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        match = re.match(r"(.+?)\s{2,}(.+)", line)
        if match:
            gp_name = match.group(1)
            genes = match.group(2).split()
            gp_to_genes[gp_name] = genes
        else:
            print(f"Could not parse line: {line}")

print(f"Loaded {len(gp_to_genes)} gene programs.")
# Print the first 3 GPs and a sample of genes for checking
for i, (gp, genes) in enumerate(gp_to_genes.items()):
    print(f"{gp}: {genes[:5]} ... ({len(genes)} genes)")
    if i > 2: break

client = OpenAI(api_key="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")
 # Function to get embedding from OpenAI
def get_embedding(text, model="text-embedding-3-large"):
    response = client.embeddings.create(input=text, model=model)
    return response.data[0].embedding

embeddings_dict = {}

for gp, genes in gp_to_genes.items():
    # Prepare input text for embedding (GP description + gene list)
    description = f"{gp}: {', '.join(genes)}"
    # Get embedding from OpenAI
    embedding = get_embedding(description)
    embeddings_dict[gp] = embedding
    print(f"Processed embedding for: {gp} (Dimension: {len(embedding)})")

# Save embeddings as NumPy binary format
np.save("gp_embeddings_partial.npy", embeddings_dict)

print("Partial embeddings saved.")

remaining_gps = {gp: genes for gp, genes in gp_to_genes.items() if gp not in embeddings_dict}

print(f"Remaining embeddings to generate: {len(remaining_gps)}")

import tiktoken

encoding = tiktoken.get_encoding("cl100k_base")
MAX_TOKENS = 8000  # A little under 8192 for safety

def truncate_text_if_needed(text, gp_name, max_tokens=MAX_TOKENS):
    tokens = encoding.encode(text)
    if len(tokens) > max_tokens:
        print(f"Truncating GP '{gp_name}' from {len(tokens)} tokens to {max_tokens} tokens.")
        truncated_tokens = tokens[:max_tokens]
        return encoding.decode(truncated_tokens)
    else:
        return text

# In your embedding loop:
for gp, genes in remaining_gps.items():
    description = f"{gp}: {', '.join(genes)}"
    safe_description = truncate_text_if_needed(description, gp)
    embedding = get_embedding(safe_description)
    embeddings_dict[gp] = embedding
    print(f"Processed embedding for: {gp} (Dimension: {len(embedding)})")