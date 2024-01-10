import os
from datasets import load_from_disk
import pandas as pd
import numpy as np
import tqdm

if os.getcwd().split("/")[-2] != "T_perturb":
    #set working directory to root of repository
    os.chdir("/lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/T_perturb/T_perturb/pp")
    print("Changed working directory to root of repository")

dataset = load_from_disk("./res/dataset/cytoimmgen_tokenised_degs.dataset")
#create dataframe for cell pairings including Donor, Cell_type, Time_point
metadata_df = pd.DataFrame({
    "Donor": dataset["Donor"],
    "Cell_type": dataset["Cell_type"],
    "Time_point": dataset["Time_point"]
    })
#drop Donor if they do not have Cell_type, Donor in all the Time_points
metadata_df_ = metadata_df[metadata_df.groupby(["Donor","Cell_type"])["Time_point"].transform("nunique") == 4]
print(f"dropped {metadata_df['Donor'].nunique() - metadata_df_['Donor'].nunique()} donors")
#choose 0h cells as resting
resting_cells = metadata_df_.loc[metadata_df_["Time_point"] == "0h",:]
#create dictionnary to store all cell pairings
cell_pairings = {
    '0h': [],
    '16h': [],
    '40h': [],
    '5d': []
}
grouped = metadata_df_.groupby(['Donor', 'Cell_type'])
for idx, resting in tqdm.tqdm(resting_cells.iterrows(), total=resting_cells.shape[0]):
    #choose a 16h cell and select that index
    group = grouped.get_group((resting["Donor"], resting["Cell_type"]))
    # Get the indices for the 16h, 40h, and 5d cells from the filtered DataFrame
    indices_16h = group[group["Time_point"] == "16h"].index
    indices_40h = group[group["Time_point"] == "40h"].index
    indices_5d = group[group["Time_point"] == "5d"].index
    # append the chosen index to the dictionary
    cell_pairings["0h"].append(idx)
    cell_pairings["16h"].append(np.random.choice(indices_16h))
    cell_pairings["40h"].append(np.random.choice(indices_40h))
    cell_pairings["5d"].append(np.random.choice(indices_5d))
#subset dataset to store separate dataset for each time point
dataset_0h = dataset.select(cell_pairings["0h"])
dataset_16h = dataset.select(cell_pairings["16h"])
dataset_40h = dataset.select(cell_pairings["40h"])
dataset_5d = dataset.select(cell_pairings["5d"])
dataset_0h.save_to_disk("./res/dataset/cytoimmgen_tokenised_degs_0h.dataset")
dataset_16h.save_to_disk("./res/dataset/cytoimmgen_tokenised_degs_16h.dataset")
dataset_40h.save_to_disk("./res/dataset/cytoimmgen_tokenised_degs_40h.dataset")
dataset_5d.save_to_disk("./res/dataset/cytoimmgen_tokenised_degs_5d.dataset")