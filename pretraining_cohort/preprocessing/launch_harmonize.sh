#!/bin/bash

CONFIG_FILE=$1  

if [ "$USER" == "am74" ]; then
    source /etc/profile.d/modules.sh
    module load cellgen/conda
    conda activate scanpy
elif [ "$USER" == "av13" ]; then
    source /etc/profile.d/modules.sh
    module load cellgen/conda
    conda activate scanpy
fi

python /nfs/team361/am74/Cytomeister/scripts/pp_cohort_2/code_harmonization/harmonize_bronze_to_silver.py --config_file ${CONFIG_FILE}
