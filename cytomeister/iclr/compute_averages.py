import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib import style

if os.getcwd().split('/')[-1] != 'healthy_imm_expr':
    # set working directory to root of repository
    os.chdir('/lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/')
    print('Changed working directory to root of repository')

style.use('default')
style.use(
    '/lustre/scratch123/hgi/projects/healthy_imm_expr/'
    't_generative/T_perturb/T_perturb/pp/mpl_style.mplstyle'
)
# Table 1: Benchmarking results for interpolation/extrapolation of single cell data
# ---------------------------------------------------------

# 1.1 Cytoimmgen
# ---------------------------------------------------------
# 1.1.1 Cytoimmgen interpolation (ours)
inter_seed_42_df = pd.read_csv(
    'T_perturb/T_perturb/iclr/cytoimmgen/interpolation/res/'
    '20240928-00:15_psin_learnt_mcosine_t1.0_i20_'
    's{self.sequence_length}_metrics.csv',
    index_col=0,
)
inter_seed_100_df = pd.read_csv(
    'T_perturb/T_perturb/iclr/cytoimmgen/interpolation/res/'
    '20240929-10:17_psin_learnt_mcosine_t1.0_i20_s100_metrics.csv',
    index_col=0,
)
inter_seed_0_df = pd.read_csv(
    'T_perturb/T_perturb/iclr/cytoimmgen/interpolation/res/'
    '20240929-10:19_psin_learnt_mcosine_t1.0_i20_s0_metrics.csv',
    index_col=0,
)
# concatenate the dataframes
inter_df = pd.concat([inter_seed_42_df, inter_seed_100_df, inter_seed_0_df])
# for each column calculate the mean and standard deviation
inter_mean = inter_df.mean()
# add column names
inter_mean.rename('inter_mean', inplace=True)
inter_std = inter_df.std()
inter_std.rename('inter_std', inplace=True)
inter_summary = pd.concat([inter_mean, inter_std], axis=1)


def calc_mean_std(df, prefix):
    mean = df.mean()
    mean.rename(f'{prefix}_mean', inplace=True)
    std = df.std()
    std.rename(f'{prefix}_std', inplace=True)
    return pd.concat([mean, std], axis=1)


# print the mean and standard deviation

# 1.1.2 Cytoimmgen extrapolation (ours)
extra_seed_42_df = pd.read_csv(
    'T_perturb/T_perturb/iclr/cytoimmgen/extrapolation/res/'
    '20240928-12:44_psin_learnt_mcosine_t1.0'
    '_i20_s{self.sequence_length}_metrics.csv',
    index_col=0,
)
extra_seed_100_df = pd.read_csv(
    './T_perturb/T_perturb/iclr/cytoimmgen/extrapolation/res/'
    '20240929-10:23_psin_learnt_mcosine_t1.0'
    '_i20_s100_metrics.csv',
    index_col=0,
)
extra_seed_0_df = pd.read_csv(
    'T_perturb/T_perturb/iclr/cytoimmgen/extrapolation/res/'
    '20240929-10:24_psin_learnt_mcosine_t1.0_i20_s0_metrics.csv',
    index_col=0,
)

extra_df = pd.concat([extra_seed_42_df, extra_seed_100_df, extra_seed_0_df])
extra_mean = extra_df.mean()
extra_mean.rename('extra_mean', inplace=True)
extra_std = extra_df.std()
extra_std.rename('extra_std', inplace=True)
extra_summary = pd.concat([extra_mean, extra_std], axis=1)

cytoimmgen_summary = pd.concat([inter_summary, extra_summary], axis=1)

cytoimmgen_summary.to_csv(
    'T_perturb/T_perturb/iclr/final_results/cytoimmgen_summary_ours.csv'
)

# 1.1.3 Cytoimmgen interpolation (CFM)
inter_seed_42_df = pd.read_csv(
    './benchmarking/scNODE/benchmark/res/cytoimmgen/'
    'cfm/metrics_train_tp_[0, 1, 3]_test_tp[2]_42.csv',
    index_col=0,
)
inter_seed_0_df = pd.read_csv(
    './benchmarking/scNODE/benchmark/res/cytoimmgen/'
    'cfm/metrics_train_tp_[0, 1, 3]_test_tp[2]_0.csv',
    index_col=0,
)
inter_seed_100_df = pd.read_csv(
    './benchmarking/scNODE/benchmark/res/cytoimmgen/'
    'cfm/metrics_train_tp_[0, 1, 3]_test_tp[2]_100.csv',
    index_col=0,
)
inter_df = pd.concat([inter_seed_42_df, inter_seed_0_df, inter_seed_100_df])
inter_summary = calc_mean_std(inter_df, 'inter')

# 1.1.4 Cytoimmgen extrapolation (CFM)
extra_seed_42_df = pd.read_csv(
    'benchmarking/scNODE/benchmark/res/cytoimmgen/'
    'cfm/metrics_train_tp_[0, 1, 2]_test_tp[3]_42.csv',
    index_col=0,
)
extra_seed_0_df = pd.read_csv(
    'benchmarking/scNODE/benchmark/res/cytoimmgen/'
    'cfm/metrics_train_tp_[0, 1, 2]_test_tp[3]_0.csv',
    index_col=0,
)
extra_seed_100_df = pd.read_csv(
    'benchmarking/scNODE/benchmark/res/cytoimmgen/'
    'cfm/metrics_train_tp_[0, 1, 2]_test_tp[3]_100.csv',
    index_col=0,
)
extra_df = pd.concat([extra_seed_42_df, extra_seed_0_df, extra_seed_100_df])
extra_summary = calc_mean_std(extra_df, 'extra')

cytoimmgen_summary_cfm = pd.concat([inter_summary, extra_summary], axis=1)
cytoimmgen_summary_cfm.to_csv(
    'T_perturb/T_perturb/iclr/final_results/cytoimmgen_summary_cfm.csv'
)

# 1.1.5 Cytoimmgen interpolation (MIOFlow)
inter_seed_42_df = pd.read_csv(
    'benchmarking/scNODE/benchmark/res/cytoimmgen/'
    'mioflow/metrics_train_tp_[0, 1, 3]_test_tp[2]_42.csv',
    index_col=0,
)
inter_seed_0_df = pd.read_csv(
    'benchmarking/scNODE/benchmark/res/cytoimmgen/'
    'mioflow/metrics_train_tp_[0, 1, 3]_test_tp[2]_0.csv',
    index_col=0,
)
inter_seed_100_df = pd.read_csv(
    'benchmarking/scNODE/benchmark/res/cytoimmgen/'
    'mioflow/metrics_train_tp_[0, 1, 3]_test_tp[2]_100.csv',
    index_col=0,
)
inter_df = pd.concat([inter_seed_42_df, inter_seed_0_df, inter_seed_100_df])
inter_summary = calc_mean_std(inter_df, 'inter')
extra_seed_42_df = pd.read_csv(
    'benchmarking/scNODE/benchmark/res/cytoimmgen/'
    'mioflow/metrics_train_tp_[0, 1, 2]_test_tp[3]_42.csv',
    index_col=0,
)
extra_seed_0_df = pd.read_csv(
    'benchmarking/scNODE/benchmark/res/cytoimmgen/'
    'mioflow/metrics_train_tp_[0, 1, 2]_test_tp[3]_0.csv',
    index_col=0,
)
extra_seed_100_df = pd.read_csv(
    'benchmarking/scNODE/benchmark/res/cytoimmgen/'
    'mioflow/metrics_train_tp_[0, 1, 2]_test_tp[3]_100.csv',
    index_col=0,
)
extra_df = pd.concat([extra_seed_42_df, extra_seed_0_df, extra_seed_100_df])
extra_summary = calc_mean_std(extra_df, 'extra')
cytoimmgen_summary_mioflow = pd.concat([inter_summary, extra_summary], axis=1)
cytoimmgen_summary_mioflow.to_csv(
    'T_perturb/T_perturb/iclr/final_results/cytoimmgen_summary_mioflow.csv'
)

# 1.2 EB
# ---------------------------------------------------------
# 1.2.1 EB interpolation
inter_seed_42_df = pd.read_csv(
    'T_perturb/T_perturb/iclr/eb/interpolation/res/'
    '20240929-11:42_psin_learnt_mcosine_t1.0_i20_s42_metrics.csv',
    index_col=0,
)
inter_seed_100_df = pd.read_csv(
    'T_perturb/T_perturb/iclr/eb/interpolation/res/'
    '20241001-00:11_psin_learnt_mcosine_t1.0_i20_s100_metrics.csv',
    index_col=0,
)
inter_seed_0_df = pd.read_csv(
    'T_perturb/T_perturb/iclr/eb/interpolation/res/'
    '20241001-00:03_psin_learnt_mcosine_t1.0_i20_s0_metrics.csv',
    index_col=0,
)
inter_df = pd.concat([inter_seed_42_df, inter_seed_100_df, inter_seed_0_df])
inter_summary = calc_mean_std(inter_df, 'inter')

# 1.2.2 EB extrapolation
extra_seed_42_df = pd.read_csv(
    'T_perturb/T_perturb/iclr/eb/extrapolation/'
    'res/20240929-11:41_psin_learnt_mcosine_t1.0_i20_s42_metrics.csv',
    index_col=0,
)
extra_seed_100_df = pd.read_csv(
    'T_perturb/T_perturb/iclr/eb/extrapolation/'
    'res/20241001-00:03_psin_learnt_mcosine_t1.0_i20_s0_metrics.csv',
    index_col=0,
)
extra_seed_0_df = pd.read_csv(
    'T_perturb/T_perturb/iclr/eb/extrapolation/res/'
    '20241001-00:03_psin_learnt_mcosine_t1.0_i20_s100_metrics.csv',
    index_col=0,
)
extra_df = pd.concat([extra_seed_42_df, extra_seed_100_df, extra_seed_0_df])
extra_summary = calc_mean_std(extra_df, 'extra')

eb_summary = pd.concat([inter_summary, extra_summary], axis=1)
eb_summary.to_csv('T_perturb/T_perturb/iclr/final_results/eb_summary_ours.csv')
# 1.2.3 EB interpolation (CFM)
inter_seed_42_df = pd.read_csv(
    'benchmarking/scNODE/benchmark/res/eb/'
    'cfm/metrics_train_tp_[0, 1, 2, 4]_test_tp[3]_42.csv',
    index_col=0,
)
inter_seed_0_df = pd.read_csv(
    'benchmarking/scNODE/benchmark/res/eb/'
    'cfm/metrics_train_tp_[0, 1, 2, 4]_test_tp[3]_0.csv',
    index_col=0,
)
inter_seed_100_df = pd.read_csv(
    'benchmarking/scNODE/benchmark/res/eb/'
    'cfm/metrics_train_tp_[0, 1, 2, 4]_test_tp[3]_100.csv',
    index_col=0,
)
inter_df = pd.concat([inter_seed_42_df, inter_seed_0_df, inter_seed_100_df])
inter_summary = calc_mean_std(inter_df, 'inter')

# 1.2.4 EB extrapolation (CFM)
extra_seed_42_df = pd.read_csv(
    'benchmarking/scNODE/benchmark/res/eb/'
    'cfm/metrics_train_tp_[0, 1, 2]_test_tp[3]_42.csv',
    index_col=0,
)
extra_seed_0_df = pd.read_csv(
    'benchmarking/scNODE/benchmark/res/eb/'
    'cfm/metrics_train_tp_[0, 1, 2]_test_tp[3]_0.csv',
    index_col=0,
)
extra_seed_100_df = pd.read_csv(
    'benchmarking/scNODE/benchmark/res/eb/'
    'cfm/metrics_train_tp_[0, 1, 2]_test_tp[3]_100.csv',
    index_col=0,
)
extra_df = pd.concat([extra_seed_42_df, extra_seed_0_df, extra_seed_100_df])
extra_summary = calc_mean_std(extra_df, 'extra')

eb_summary_cfm = pd.concat([inter_summary, extra_summary], axis=1)
eb_summary_cfm.to_csv('T_perturb/T_perturb/iclr/final_results/eb_summary_cfm.csv')

# 1.2.5 EB interpolation (MIOFlow)
inter_seed_42_df = pd.read_csv(
    'benchmarking/scNODE/benchmark/res/'
    'eb/mioflow/metrics_train_tp_'
    '[0, 1, 2, 4]_test_tp[3]_0.csv',
    index_col=0,
)
inter_seed_0_df = pd.read_csv(
    'benchmarking/scNODE/benchmark/res/'
    'eb/mioflow/metrics_train_tp_'
    '[0, 1, 2, 4]_test_tp[3]_0.csv',
    index_col=0,
)
inter_seed_100_df = pd.read_csv(
    'benchmarking/scNODE/benchmark/res/'
    'eb/mioflow/metrics_train_tp_'
    '[0, 1, 2, 4]_test_tp[3]_100.csv',
    index_col=0,
)
inter_df = pd.concat([inter_seed_42_df, inter_seed_0_df, inter_seed_100_df])
inter_summary = calc_mean_std(inter_df, 'inter')
extra_seed_42_df = pd.read_csv(
    'benchmarking/scNODE/benchmark/res/'
    'eb/mioflow/metrics_train_tp_'
    '[0, 1, 2]_test_tp[3]_42.csv',
    index_col=0,
)
extra_seed_0_df = pd.read_csv(
    'benchmarking/scNODE/benchmark/res/'
    'eb/mioflow/metrics_train_tp_'
    '[0, 1, 2]_test_tp[3]_0.csv',
    index_col=0,
)
extra_seed_100_df = pd.read_csv(
    'benchmarking/scNODE/benchmark/res/'
    'eb/mioflow/metrics_train_tp_'
    '[0, 1, 2]_test_tp[3]_100.csv',
    index_col=0,
)
extra_df = pd.concat([extra_seed_42_df, extra_seed_0_df, extra_seed_100_df])
extra_summary = calc_mean_std(extra_df, 'extra')
eb_summary_mioflow = pd.concat([inter_summary, extra_summary], axis=1)
eb_summary_mioflow.to_csv(
    'T_perturb/T_perturb/iclr/final_results/eb_summary_mioflow.csv'
)

# 1.2.6 EB interpolation (PRESCIENT)
inter_seed_42_df = pd.read_csv(
    'benchmarking/scNODE/benchmark/res/'
    'eb/prescient/metrics_train_tp_[0, 1, 2, 4]'
    '_test_tp_[3]_s42.csv',
    index_col=0,
)
inter_seed_0_df = pd.read_csv(
    'benchmarking/scNODE/benchmark/res/'
    'eb/prescient/metrics_train_tp_[0, 1, 2, 4]'
    '_test_tp_[3]_s0.csv',
    index_col=0,
)
inter_seed_100_df = pd.read_csv(
    'benchmarking/scNODE/benchmark/res/'
    'eb/prescient/metrics_train_tp_[0, 1, 2, 4]'
    '_test_tp_[3]_s100.csv',
    index_col=0,
)
inter_df = pd.concat([inter_seed_42_df, inter_seed_0_df, inter_seed_100_df])
inter_summary = calc_mean_std(inter_df, 'inter')
extra_seed_42_df = pd.read_csv(
    'benchmarking/scNODE/benchmark/res/'
    'eb/prescient/metrics_train_tp_[0, 1, 2]'
    '_test_tp_[3]_s42.csv',
    index_col=0,
)
extra_seed_0_df = pd.read_csv(
    'benchmarking/scNODE/benchmark/res/'
    'eb/prescient/metrics_train_tp_[0, 1, 2]'
    '_test_tp_[3]_s0.csv',
    index_col=0,
)
extra_seed_100_df = pd.read_csv(
    'benchmarking/scNODE/benchmark/res/'
    'eb/prescient/metrics_train_tp_[0, 1, 2]'
    '_test_tp_[3]_s100.csv',
    index_col=0,
)
extra_df = pd.concat([extra_seed_42_df, extra_seed_0_df, extra_seed_100_df])
extra_summary = calc_mean_std(extra_df, 'extra')
eb_summary_prescient = pd.concat([inter_summary, extra_summary], axis=1)
eb_summary_prescient.to_csv(
    'T_perturb/T_perturb/iclr/final_results/eb_summary_prescient.csv'
)

# 2. Encoder ablation
# ---------------------------------------------------------

gf_frozen_df = pd.read_csv(
    'T_perturb/T_perturb/iclr/cytoimmgen/interpolation/res/'
    '20240928-00:15_psin_learnt_mcosine_t1.0_i20_'
    's{self.sequence_length}_metrics.csv',
    index_col=0,
)
gf_frozen_df = gf_frozen_df.T
gf_frozen_df.rename(columns={0: 'gf_frozen'}, inplace=True)
gf_fine_tuned_df = pd.read_csv(
    'T_perturb/T_perturb/iclr/cytoimmgen/encoder/'
    'res/20240930-12:14_psin_learnt_mcosine_t1.0_i20'
    '_s42_gf_fine_tuned_metrics.csv',
    index_col=0,
)
gf_fine_tuned_df = gf_fine_tuned_df.T
gf_fine_tuned_df.rename(columns={0: 'gf_fine_tuned'}, inplace=True)
encoder_scratch_df = pd.read_csv(
    'T_perturb/T_perturb/iclr/cytoimmgen/encoder/'
    'res/20240930-12:11_psin_learnt_mcosine'
    '_t1.0_i20_s42_encoder_metrics.csv',
    index_col=0,
)
encoder_scratch_df = encoder_scratch_df.T
encoder_scratch_df.rename(columns={0: 'encoder_scratch'}, inplace=True)
encoder_summary = pd.concat(
    [gf_frozen_df, gf_fine_tuned_df, encoder_scratch_df], axis=1
)
encoder_summary.to_csv(
    'T_perturb/T_perturb/iclr/final_results/encoder_ablation_summary.csv'
)

# 3. Masking scheduler ablation
# ---------------------------------------------------------
cosine_df = pd.read_csv(
    'T_perturb/T_perturb/iclr/cytoimmgen/positional_encoding/'
    'res/20240927-12:44_psin_learnt_mcosine_metrics.csv',
    index_col=0,
)
cosine_df = cosine_df.T
cosine_df.rename(columns={0: 'cosine'}, inplace=True)
exp_df = pd.read_csv(
    'T_perturb/T_perturb/iclr/cytoimmgen/masking_scheduler/'
    'res/20240927-12:13_psin_learnt_mexp_metrics.csv',
    index_col=0,
)
exp_df = exp_df.T
exp_df.rename(columns={0: 'exp'}, inplace=True)
pow_df = pd.read_csv(
    'T_perturb/T_perturb/iclr/cytoimmgen/masking_scheduler/'
    'res/20240927-12:41_psin_learnt_mpow_metrics.csv',
    index_col=0,
)
pow_df = pow_df.T
pow_df.rename(columns={0: 'pow'}, inplace=True)
masking_scheduler_summary = pd.concat([cosine_df, exp_df, pow_df], axis=1)
masking_scheduler_summary.to_csv(
    'T_perturb/T_perturb/iclr/final_results/masking_scheduler_ablation_summary.csv'
)

# 4. Positional encoding ablation
# ---------------------------------------------------------
sin_learnt_df = pd.read_csv(
    'T_perturb/T_perturb/iclr/cytoimmgen/postional_encoding/'
    'res/20240926-00:19_psin_learnt_metrics.csv',
    index_col=0,
)
sin_learnt_df = sin_learnt_df.T
sin_learnt_df.rename(columns={0: 'sin_learnt'}, inplace=True)
time_pos_sin_df = pd.read_csv(
    'T_perturb/T_perturb/iclr/cytoimmgen/postional_encoding/'
    'res/20240926-00:19_ptime_pos_sin_metrics.csv',
    index_col=0,
)
time_pos_sin_df = time_pos_sin_df.T
time_pos_sin_df.rename(columns={0: 'time_pos_sin'}, inplace=True)
comb_sin_df = pd.read_csv(
    'T_perturb/T_perturb/iclr/cytoimmgen/postional_encoding/'
    'res/20240926-00:19_pcomb_sin_metrics.csv',
    index_col=0,
)
comb_sin_df = comb_sin_df.T
comb_sin_df.rename(columns={0: 'comb_sin'}, inplace=True)
positional_encoding_summary = pd.concat(
    [sin_learnt_df, time_pos_sin_df, comb_sin_df], axis=1
)
positional_encoding_summary.to_csv(
    'T_perturb/T_perturb/iclr/final_results/' 'positional_encoding_ablation_summary.csv'
)

# 6. Generation ablation
# ---------------------------------------------------------

t2_i5_df = pd.read_csv(
    'T_perturb/T_perturb/iclr/eb/generation/res/'
    '20240927-16:06_psin_learnt_mcosine_t2.0_i5_s150_metrics.csv',
    index_col=0,
)
t2_i5_df = t2_i5_df.T
i5_df = t2_i5_df.copy()
i5_df.rename(columns={0: 'i5'}, inplace=True)
t2_i10_df = pd.read_csv(
    'T_perturb/T_perturb/iclr/eb/generation/res/'
    '20240927-16:43_psin_learnt_mcosine_t2.0_i10_s150_metrics.csv',
    index_col=0,
)
t2_i10_df = t2_i10_df.T
i10_df = t2_i10_df.copy()
i10_df.rename(columns={0: 'i10'}, inplace=True)
t2_i15_df = pd.read_csv(
    'T_perturb/T_perturb/iclr/eb/generation/res/'
    '20240927-16:42_psin_learnt_mcosine_t2.0_i15_s150_metrics.csv',
    index_col=0,
)
t2_i15_df = t2_i15_df.T
i15_df = t2_i15_df.copy()
i15_df.rename(columns={0: 'i15'}, inplace=True)
t2_i20_df = pd.read_csv(
    'T_perturb/T_perturb/iclr/eb/generation/res/'
    '20240927-16:52_psin_learnt_mcosine_t2.0_i20_s150_metrics.csv',
    index_col=0,
)
t2_i20_df = t2_i20_df.T
i20_df = t2_i20_df.copy()
i20_df.rename(columns={0: 'i20'}, inplace=True)
# concatenate the dataframes
generation_summary = pd.concat([i5_df, i10_df, i15_df, i20_df], axis=1)
generation_summary.to_csv(
    'T_perturb/T_perturb/iclr/final_results/generation_ablation_summary.csv'
)
# convert into long format for plotting use one
# column iteration (i5, i10, i15, i20) and one column for each metric
generation_summary_long = generation_summary.stack().reset_index()
generation_summary_long.columns = ['metric', 'iteration', 'value']
# strip the 'i' from the iteration column
generation_summary_long['iteration'] = generation_summary_long['iteration'].str.strip(
    'i'
)
# plot x = iteration, y = value for rouge1_25, rouge1_100,
# rouge1_370 each with separate lines for each metric

rouge_summary_long = generation_summary_long[
    generation_summary_long['metric'].str.contains('rouge1')
]
rouge_summary_long['metric'] = rouge_summary_long['metric'].str.replace('rouge1_', '')
rouge_summary_long.rename(columns={'metric': 'rouge1_seq_length'}, inplace=True)
# size of the plot
sns.lineplot(
    data=rouge_summary_long,
    x='iteration',
    y='value',
    hue='rouge1_seq_length',
    linewidth=4,
)
# Use Matplotlib to remove duplicate legend entries
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(
    by_label.values(),
    by_label.keys(),
    title='rouge1 \nsequence length',
    loc='center left',
    bbox_to_anchor=(1, 0.5),
)
plt.savefig(
    'T_perturb/T_perturb/iclr/final_results/'
    'generation_ablation_summary_rouge_plot.pdf',
    bbox_inches='tight',
)
plt.close()

# sequence length ablation for emd and pearson_r
s_50_df = pd.read_csv(
    'T_perturb/T_perturb/iclr/eb/generation/res/'
    '20241002-07:43_psin_learnt_mcosine_t1.0_i20_s42_seq50_metrics.csv',
    index_col=0,
)
s_50_df = s_50_df.T
s_50_df.rename(columns={0: 's_50'}, inplace=True)
s_100_df = pd.read_csv(
    'T_perturb/T_perturb/iclr/eb/generation/res/'
    '20241002-07:50_psin_learnt_mcosine_t1.0_i20_s42_s100_metrics.csv',
    index_col=0,
)
s_100_df = s_100_df.T
s_100_df.rename(columns={0: 's_100'}, inplace=True)
s_150_df = pd.read_csv(
    'T_perturb/T_perturb/iclr/eb/generation/res/'
    '20241002-07:52_psin_learnt_mcosine_t1.0_i20_s42_s150_metrics.csv',
    index_col=0,
)
s_150_df = s_150_df.T
s_150_df.rename(columns={0: 's_150'}, inplace=True)
s_200_df = pd.read_csv(
    'T_perturb/T_perturb/iclr/eb/generation/res/'
    '20241002-07:54_psin_learnt_mcosine_t1.0_i20_s42_s200_metrics.csv',
    index_col=0,
)
s_200_df = s_200_df.T
s_200_df.rename(columns={0: 's_200'}, inplace=True)
# concatenate the dataframes
s_300_df = pd.read_csv(
    'T_perturb/T_perturb/iclr/eb/generation/res/'
    '20241002-07:56_psin_learnt_mcosine_t1.0_i20_s42_s300_metrics.csv',
    index_col=0,
)
s_300_df = s_300_df.T
s_300_df.rename(columns={0: 's_300'}, inplace=True)
sequence_length_summary = pd.concat(
    [s_50_df, s_100_df, s_150_df, s_200_df, s_300_df], axis=1
)
# creating the same summary for rouge1, emd and pearson_r
sequence_length_summary_long = sequence_length_summary.stack().reset_index()
sequence_length_summary_long.columns = ['metric', 'sequence_length', 'value']
# strip the 's' from the iteration column
sequence_length_summary_long['sequence_length'] = sequence_length_summary_long[
    'sequence_length'
].str.strip('s_')
# make the sequence_length column a float
sequence_length_summary_long['sequence_length'] = sequence_length_summary_long[
    'sequence_length'
].astype(float)
# plot x = iteration, y = value for rouge1_25,
# rouge1_100, rouge1_370 each with separate lines for each metric
# line plot for each metric
rouge_summary_long = sequence_length_summary_long[
    sequence_length_summary_long['metric'].str.contains('rouge1')
]
rouge_summary_long['metric'] = rouge_summary_long['metric'].str.replace('rouge1_', '')
rouge_summary_long.rename(columns={'metric': 'rouge1'}, inplace=True)
sns.lineplot(
    data=rouge_summary_long, x='sequence_length', y='value', hue='rouge1', linewidth=4
)
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(
    by_label.values(),
    by_label.keys(),
    title='rouge1 \nsequence length',
    loc='center left',
    bbox_to_anchor=(1, 0.5),
)
plt.savefig(
    'T_perturb/T_perturb/iclr/final_results/'
    'sequence_length_ablation_summary_rouge_plot.pdf',
    bbox_inches='tight',
)
plt.close()
# create a line plot for emd and pearson_r
emd_summary_long = sequence_length_summary_long[
    sequence_length_summary_long['metric'].str.contains('emd')
]
emd_summary_long.rename(columns={'metric': 'emd'}, inplace=True)
sns.lineplot(
    data=emd_summary_long, x='sequence_length', y='value', hue='emd', linewidth=4
)
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(
    by_label.values(),
    by_label.keys(),
    title='emd',
    loc='center left',
    bbox_to_anchor=(1, 0.5),
)
plt.savefig(
    'T_perturb/T_perturb/iclr/final_results/'
    'sequence_length_ablation_summary_emd_plot.pdf',
    bbox_inches='tight',
)
plt.close()
# plot for pearson_r
pearson_r_summary_long = sequence_length_summary_long[
    sequence_length_summary_long['metric'].str.contains('pearson_r')
]
pearson_r_summary_long.rename(columns={'metric': 'pearson_r'}, inplace=True)
sns.lineplot(
    data=pearson_r_summary_long,
    x='sequence_length',
    y='value',
    hue='pearson_r',
    linewidth=4,
)
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(
    by_label.values(),
    by_label.keys(),
    title='pearson_r',
    loc='center left',
    bbox_to_anchor=(1, 0.5),
)
plt.savefig(
    'T_perturb/T_perturb/iclr/final_results/'
    'sequence_length_ablation_summary_pearson_r_plot.pdf',
    bbox_inches='tight',
)

# plot EMD for each iteration
emd_summary_long = generation_summary_long[
    generation_summary_long['metric'].str.contains('emd')
]
emd_summary_long.rename(columns={'metric': 'emd'}, inplace=True)
sns.lineplot(data=emd_summary_long, x='iteration', y='value', hue='emd', linewidth=4)
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(
    by_label.values(), by_label.keys(), loc='center left', bbox_to_anchor=(1, 0.5)
)
plt.savefig(
    'T_perturb/T_perturb/iclr/final_results/generation_ablation_summary_emd_plot.pdf',
    bbox_inches='tight',
)
plt.close()

# plot
pearson_r_summary_long = generation_summary_long[
    generation_summary_long['metric'].str.contains('pearson_r')
]
pearson_r_summary_long.rename(columns={'metric': 'pearson_r'}, inplace=True)
sns.lineplot(
    data=pearson_r_summary_long, x='iteration', y='value', hue='pearson_r', linewidth=4
)
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(
    by_label.values(), by_label.keys(), loc='center left', bbox_to_anchor=(1, 0.5)
)
plt.savefig(
    'T_perturb/T_perturb/iclr/final_results/'
    'generation_ablation_summary_pearson_r_plot.pdf',
    bbox_inches='tight',
)
plt.close()

t_05_df = pd.read_csv(
    'T_perturb/T_perturb/iclr/eb/generation/res/'
    '20240927-16:58_psin_learnt_mcosine_t0.5_i20_s150_metrics.csv',
    index_col=0,
)
t_05_df = t_05_df.T
t_05_df.rename(columns={0: 't_05'}, inplace=True)
t1_df = pd.read_csv(
    'T_perturb/T_perturb/iclr/eb/generation/res/'
    '20240927-17:11_psin_learnt_mcosine_t1.0_i20_s150_metrics.csv',
    index_col=0,
)
t1_df = t1_df.T
t1_df.rename(columns={0: 't1'}, inplace=True)
t_15_df = pd.read_csv(
    'T_perturb/T_perturb/iclr/eb/generation/res/'
    '20240927-16:59_psin_learnt_mcosine_t1.5_i20_s150_metrics.csv',
    index_col=0,
)
t_15_df = t_15_df.T
t_15_df.rename(columns={0: 't_15'}, inplace=True)
t2_df = t2_i20_df.copy()
t2_df.rename(columns={0: 't2'}, inplace=True)
t3_df = pd.read_csv(
    'T_perturb/T_perturb/iclr/eb/generation/res/'
    '20240927-17:00_psin_learnt_mcosine_t3.0_i20_s150_metrics.csv',
    index_col=0,
)
t3_df = t3_df.T
t3_df.rename(columns={0: 't3'}, inplace=True)

# concatenate the dataframes
temperature_summary = pd.concat([t_05_df, t1_df, t_15_df, t2_df, t3_df], axis=1)
# creating the same summary for rouge1, emd and pearson_r
temperature_summary_long = temperature_summary.stack().reset_index()
temperature_summary_long.columns = ['metric', 'temperature', 'value']
# strip the 't' from the iteration column
temperature_summary_long['temperature'] = temperature_summary_long[
    'temperature'
].str.strip('t')
# replace temperature _05 with 0.5
temperature_summary_long['temperature'] = temperature_summary_long[
    'temperature'
].str.replace('_05', '0.5')
temperature_summary_long['temperature'] = temperature_summary_long[
    'temperature'
].str.replace('_15', '1.5')
# make the temperature column a float
temperature_summary_long['temperature'] = temperature_summary_long[
    'temperature'
].astype(float)
# plot x = iteration, y = value for rouge1_25,
# rouge1_100, rouge1_370 each with separate lines for each metric
# bar plot for each metric
rouge_summary_long = temperature_summary_long[
    temperature_summary_long['metric'].str.contains('rouge1')
]
rouge_summary_long['metric'] = rouge_summary_long['metric'].str.replace('rouge1_', '')
rouge_summary_long.rename(columns={'metric': 'rouge1'}, inplace=True)
sns.barplot(data=rouge_summary_long, x='rouge1', y='value', hue='temperature')
# x axis label
plt.xlabel('sequence length')
plt.ylabel('rouge1')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(
    by_label.values(),
    by_label.keys(),
    title='temperature',
    loc='center left',
    bbox_to_anchor=(1, 0.5),
)
plt.savefig(
    'T_perturb/T_perturb/iclr/final_results/temperature_ablation_summary_plot.pdf',
    bbox_inches='tight',
)
plt.close()
# create bar plot for emd and pearson_r in the same plot
emd_summary_long = temperature_summary_long[
    temperature_summary_long['metric'].str.contains('emd')
]
emd_summary_long.rename(columns={'metric': 'emd'}, inplace=True)
sns.barplot(data=emd_summary_long, x='emd', y='value', hue='temperature')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(
    by_label.values(),
    by_label.keys(),
    title='temperature',
    loc='center left',
    bbox_to_anchor=(1, 0.5),
)
plt.savefig(
    'T_perturb/T_perturb/iclr/final_results/temperature_ablation_summary_emd_plot.pdf',
    bbox_inches='tight',
)
plt.close()
