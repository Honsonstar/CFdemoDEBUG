#!/bin/bash

DATA_ROOT_DIR='/root/autodl-tmp/newcfdemo/CFdemo_gene_text_copy/datasets_csv/reports_clean' # where are the TCGA features stored?
BASE_DIR="/root/autodl-tmp/newcfdemo/CFdemo_gene_text_copy"            # where is the repo cloned?

# STUDY="blca"
STUDY="brca" 
# STUDY="hnsc" 
# STUDY="stad"
# STUDY="coadread"

TYPE_OF_PATH="combine" # what type of pathways? 
MODEL="snn" # what type of model do you want to train?

#改omics_dir
CUDA_VISIBLE_DEVICES=0 conda run -n causal python main.py \
    --label_file datasets_csv/clinical_data/tcga_${STUDY}_clinical.csv \
    --study tcga_${STUDY} \
    --split_dir splits \
    --data_root_dir $DATA_ROOT_DIR \
    --task survival \
    --which_splits 5foldcv_ramdom \
    --omics_dir /root/autodl-tmp/newcfdemo/CFdemo_gene_text_copy/preprocessing/CPCG_algo/raw_data/finalstage_result_/tcga_${STUDY}/tcga_${STUDY}_M2M3base_0916.csv \
    --results_dir "case_results_dbiase001" \
    --batch_size 1 \
    --lr 0.0005 \
    --opt radam \
    --reg 0.0001 \
    --alpha_surv 0.5 \
    --weighted_sample \
    --max_epochs 20 \
    --label_col survival_months_dss \
    --k 5 \
    --bag_loss nll_surv \
    --type_of_path custom \
    --modality $MODEL \
    --enable_multitask \
    --multitask_weight 0.12
    # --multitask_weight 0.12