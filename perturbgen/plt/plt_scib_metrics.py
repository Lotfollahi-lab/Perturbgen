import scanpy as sc
from scib_metrics.benchmark import Benchmarker

adata = sc.read_h5ad(
    '/lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/T_perturb/'
    'T_perturb/plt/res/Cora/cls_embeddings_stratified_pairing_16h.h5ad'
)
sc.tl.pca(adata, svd_solver='arpack', n_comps=50)
adata.obsm['Unintergrated'] = adata.obsm['X_pca']
# rename embedding
adata.obsm['scConformer'] = adata.obsm['X_CLS_embeddings']
del adata.obsm['X_CLS_embeddings']
adata.obsm['GF_zero_shot'] = adata.obsm['X_GF_zero_shot']
del adata.obsm['X_GF_zero_shot']
bm = Benchmarker(
    adata,
    batch_key='Cell_culture_batch',
    label_key='Cell_population',
    embedding_obsm_keys=['Unintergrated', 'scConformer', 'GF_zero_shot'],
    n_jobs=8,
)
bm.benchmark()
bm.plot_results_table(
    save_dir='/lustre/scratch123/hgi/projects/healthy_imm_expr/'
    't_generative/T_perturb/perturbgen/plt/res'
)
bm.plot_results_table(
    min_max_scale=False,
    save_dir='/lustre/scratch123/hgi/projects/healthy_imm_expr/'
    't_generative/T_perturb/perturbgen/plt/res',
)
df = bm.get_results(min_max_scale=False)
df.to_csv(
    '/lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/'
    'T_perturb/cytomeister/plt/res/scib.csv'
)
# save results table
