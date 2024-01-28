import matplotlib.pyplot as plt
import scanpy as sc

adata = sc.read_h5ad(
    '/lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/'
    'T_perturb/T_perturb/pp/res/cls_embeddings.h5ad'
)
# plot umap of cls embeddings
fig, ax = plt.subplots(figsize=(1, 1))
sc.pp.neighbors(adata, n_neighbors=15, use_rep='X')
sc.tl.umap(adata)
sc.pl.umap(adata, color='tgt_cell_type', ax=ax, show=False)
plt.savefig(
    '/lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/'
    'T_perturb/T_perturb/pp/res/cls_embeddings_umap.pdf',
    bbox_inches='tight',
)
