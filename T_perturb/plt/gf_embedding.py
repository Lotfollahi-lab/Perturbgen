import os
from pathlib import Path

import geneformer.perturber_utils as pu
import pandas as pd
import scanpy as sc
import torch
from datasets import load_from_disk
from geneformer import EmbExtractor
from geneformer.emb_extractor import get_embs, label_cell_embs
from matplotlib import pyplot as plt
from matplotlib import style

style.use(
    '/lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/'
    'T_perturb/T_perturb/pp/mpl_style.mplstyle'
)
# from transformers import BertForSequenceClassification, BertForTokenClassification
dataset_name = 'cytoimmgen_tokenised_hvg_paired.dataset'

# Set default figure facecolor to white
plt.rcParams['figure.facecolor'] = 'white'
if os.getcwd().split('/')[-3] != 'T_perturb':
    # set working directory to root of repository
    os.chdir(
        '/lustre/scratch123/hgi/projects/healthy_imm_expr/'
        't_generative/T_perturb/T_perturb/plt'
    )
    print('Changed working directory to root of repository')

tokenized_dir = (
    '/lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/'
    'T_perturb/T_perturb/pp/res/dataset/'
    '{dataset_name}'
)
dataset = load_from_disk(tokenized_dir)
num_labels = len(set(dataset['Cell_type']))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# inherit EmbExtractor to avoid sorting of embs
class non_sorted_EmbExtractor(EmbExtractor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def extract_embs(
        self,
        model_directory,
        input_data_file,
        output_directory,
        output_prefix,
        output_torch_embs=False,
        cell_state=None,
    ):
        filtered_input_data = pu.load_and_filter(
            self.filter_data, self.nproc, input_data_file
        )
        if cell_state is not None:
            filtered_input_data = pu.filter_by_dict(
                filtered_input_data, cell_state, self.nproc
            )
        model = pu.load_model(self.model_type, self.num_classes, model_directory)
        layer_to_quant = pu.quant_layers(model) + self.emb_layer
        embs = get_embs(
            model,
            filtered_input_data,  # Remove downsampling code
            self.emb_mode,
            layer_to_quant,
            self.pad_token_id,
            self.forward_batch_size,
            self.summary_stat,
        )

        if self.emb_mode == 'cell':
            if self.summary_stat is None:
                embs_df = label_cell_embs(embs, filtered_input_data, self.emb_label)
            elif self.summary_stat is not None:
                embs_df = pd.DataFrame(embs.cpu().numpy()).T
        elif self.emb_mode == 'gene':
            if self.summary_stat is None:
                embs_df = self.label_gene_embs(
                    embs, filtered_input_data, self.token_gene_dict
                )
            elif self.summary_stat is not None:
                embs_df = pd.DataFrame(embs).T
                embs_df.index = [self.token_gene_dict[token] for token in embs_df.index]

        # save embeddings to output_path
        if cell_state is None:
            output_path = (Path(output_directory) / output_prefix).with_suffix('.csv')
            embs_df.to_csv(output_path)

        if self.exact_summary_stat == 'exact_mean':
            embs = embs.mean(dim=0)
            embs_df = pd.DataFrame(
                embs_df[0:255].mean(axis='rows'), columns=[self.exact_summary_stat]
            ).T
        elif self.exact_summary_stat == 'exact_median':
            embs = torch.median(embs, dim=0)[0]
            embs_df = pd.DataFrame(
                embs_df[0:255].median(axis='rows'), columns=[self.exact_summary_stat]
            ).T

        if cell_state is not None:
            return embs
        else:
            if output_torch_embs:
                return embs_df, embs
            else:
                return embs_df


embex = non_sorted_EmbExtractor(
    model_type='CellClassifier',
    num_classes=num_labels,  # number of cell types for unsupervised training
    emb_mode='cell',
    max_ncells=len(dataset),  # extract embeddings for all cells
    emb_layer=0,  # 0 = to last layer, -1 = second to last layer
    forward_batch_size=32,
    emb_label=['Cell_population', 'Cell_type'],
    labels_to_plot=['Cell_population', 'Cell_type'],
    nproc=8,
    summary_stat=None,
)

# embs = embex.extract_embs(
#     './res/Geneformer/240131_geneformer_CellClassifier_L2048_B32_LR5e-05_LSlinear_WU10000_E3_Oadamw_F5_16h/checkpoint-16272',
#     '/lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/'
#     'T_perturb/T_perturb/pp/res/dataset/cytoimmgen_degs_random_pairing_16h.dataset',
#     './res/Geneformer',
#     'cell_embeddings_finetuned_16h',
# )
embs = embex.extract_embs(
    '/lustre/scratch123/hgi/projects/healthy_imm_expr/'
    't_generative/generative_modelling_omic/Geneformer',
    tokenized_dir,
    './res/Geneformer',
    'cell_embeddings_zeroshot_16h',
)
# only keep numerical columns from embs
embs_embeddings = embs.select_dtypes(include='number')
# load adata
adata = sc.read_h5ad(
    '/lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/T_perturb/'
    'T_perturb/plt/res/Cora/cls_embeddings_stratified_pairing_16h.h5ad'
)
# check if Cell_population in adata.obs and embs_embeddings Cell_population are the same

assert adata.obs['Cell_population'].tolist() == embs['Cell_population'].tolist()

# convert embs into np.array
embs_embeddings = embs_embeddings.to_numpy()
adata.obsm['X_GF_zero_shot'] = embs_embeddings
# save adata
adata.write_h5ad(
    '/lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/T_perturb/'
    'T_perturb/plt/res/Cora/cls_embeddings_stratified_pairing_16h.h5ad'
)

sc.pp.neighbors(adata, use_rep='X_GF_zero_shot', n_neighbors=50)
sc.tl.umap(adata)
sc.pl.umap(
    adata,
    color=[
        'Cell_type',
        'Cell_population',
        'Cell_culture_batch',
        'Activation_level',
    ],
    wspace=0.5,
    ncols=2,
    frameon=False,
    show=False,
)
plt.savefig(
    './res/Geneformer/umap_cell_type_zeroshot_16h.pdf', dpi=200, bbox_inches='tight'
)
plt.close()
