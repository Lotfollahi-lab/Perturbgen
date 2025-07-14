#!/bin/bash
#BSUB -q gpu-lotfollahi # name of the partition to run job on (options: gpu-normal, gpu-huge, gpu-lotfollahi)
#BSUB -gpu "mode=exclusive_process:num=1" # request for exclusive access to gpu  :gmodel=NVIDIA_H100_HBM3_80GB
#BSUB -J hspc_perturb_cluster_[1-278]%4 # job array with 35 jobs, max 4 running at the same time
#BSUB -n 4 # number of cores
#BSUB -G cellulargenetics-priority # groupname for billing team361
#BSUB -cwd /lustre/scratch126/cellgen/lotfollahi/kl11/ # working directory
#BSUB -o T_perturb/cytomeister/logs/perturb_cluster_DEG_%J_%I.out # output file
#BSUB -e T_perturb/cytomeister/logs/perturb_cluster_DEG_%J_%I.err # error file
#BSUB -M 50000  # RAM memory part 2. Default: 100MB
#BSUB -R 'select[mem>50000] rusage[mem=50000]' # RAM memory part 1. Default: 100MB

# load cuda
module load cuda-12.1.1

# activate pyenv
source /nfs/team361/cytomeister/.cytomeister/bin/activate
cwd=$(pwd)

export WANDB_DIR=$cwd/wandb
# run script
echo "--- Start computing model"

GENES=(
  TOP3A DKC1 TERC TERT TINF2 NOP10 USB1 WRAP53 NHP2 FANCA FANCB FANCC FANCD2
  FANCE FANCF FANCG FANCI FANCL FANCM PALB2 RAD51C SLX4 BRCA1 BRCA2 BRIP1 ERCC4
  UBE2T MAD2L2 XRCC2 ATM BLM GATA2 SAMD9 SAMD9L SRP72 DDX41 DNAJC21 CTC1 ADH5
  ERCC6L2 AK2 EFL1 MASTL MYSM1 NPM1 RAP1B STN1 LIG4 PSMB8 IFNG TCN2 GATA1 CDAN1
  COX4I2 KIF23 KLF1 LPIN2 C15orf41 PDGFRA FLT3 ABL1 ACSL6 ARHGAP26 ASXL1 BCOR
  BCORL1 BRAF CBLB CBLC CDKN2A CUX1 DNMT3A EZH2 FBXW7 GNAS HRAS IDH1 IDH2 IRF1
  JAK3 KDM6A KIT KMT2A KRAS MYD88 NF1 NOTCH1 NRAS PHF6 PTEN RAD21 SETBP1 SH3BP1
  SMC1A SMC3 SRSF2 STAG2 TET2 U2AF1 WT1 ZRSR2 HBB HBA1 HBA2 HBG1 HBG2 ATRX HBE1
  HBD AK1 G6PD GCLC GPI GSR GSS HK1 NT5C3A PKLR TPI1 PGK1 GPX1 ALDOA ENO1 ANK1
  SLC4A1 SPTB EPB42 SPTA1 EPB41 CD59 PFKM PIGA PIGT PIEZO1 KCNN4 RHAG SLC2A1
  PRF1 UNC13D STX11 STXBP-2 STXBP2 SLC11A2 STEAP3 TF TMPRSS6 RPS20 RPL13 RPL19
  RPL35 RPS19 RPL5 RPS26 RPL11 RPL35a RPS10 RPS24 RPS17 RPL15 RPS28 RPS29 RPS7
  RPS15 RPS27a RPS27 RPL9 RPL18 RPL26 RPL27 RPL31 TSR2 ADA2 EPO CEBPA ETV6 RUNX1
  ACD CHEK2 RTEL1 PARN NBN NAF1 PAX5 MLH1 MSH2 MSH6 ITK AMN CUBN DHFR MTR MTRR
  SLC19A2 UMPS JAK2 MPL THPO CALR SH2B3 SBDS HAX1 G6PC3 ELANE SLC37A4 GFI1 JAGN1
  AP3B1 VPS45 LAMTOR2 TCIRG1 WDR1 CXCR4 WAS TAZ VPS13B DBF4 CSF3R ACKR1 SRP54
  CLPB HTRA2 NCF4 CYBB MPO ACTN1 ANO6 AP3D1 ARPC1B BLOC1S3 BLOC1S6 DTNBP1 FERMT3
  GFI1B GP6 GP9 HPS1 HPS3 HPS4 HPS5 HPS6 ITGA2B ITGB3 LYST P2RY12 PLA2G4A PLAU
  PRKACG RASGRP2 TBXA2R TBXAS1 VIPAS39 VPS33B VWF BLOC1S5 P2RX1 PTGS1 PTPN11
  SLC45A2 ITGA2 CD36 XK CYB5R3 CYB5A UROS SF3B1 ALAS2 GLRX5 HSPA9 PUS1 SEC23B
  SLC25A38 YARS2 LARS2 NDUFB11 ABCB7 TRNT1 ANKRD26 FLI1 FYB GP1BA GP1BB HOXA11
  MYH9 NBEAL2
)



INDEX=$((LSB_JOBINDEX - 1))
PERTURBED_GENE=${GENES[$INDEX]}

echo "[$(date)] Starting job index $LSB_JOBINDEX"
echo "Current gene to be perturbed: $PERTURBED_GENE"

CONFIG_PATH="T_perturb/cytomeister/configs/eval/HSPC/mask_src_inference_perturbation_jobarray.yaml"
TMP_CONFIG_PATH="T_perturb/cytomeister/configs/eval/HSPC/mask_src_generate_perturbation_${PERTURBED_GENE}_${LSB_JOBID}.yaml"

sed "s/{{PERTURBED_GENE}}/${PERTURBED_GENE}/g" $CONFIG_PATH > $TMP_CONFIG_PATH

if [[ $INDEX -ge ${#GENES[@]} ]]; then
  echo "Error: LSB_JOBINDEX ($LSB_JOBINDEX) out of range for GENES array" >&2
  exit 1
fi

# python3 $cwd/train.py \
python3 /lustre/scratch126/cellgen/lotfollahi/kl11/T_perturb/cytomeister/Perturb/val.py \
--config $TMP_CONFIG_PATH 

# Clean up temporary config file
trap "rm -f $TMP_CONFIG_PATH" EXIT
