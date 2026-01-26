import os
import random
import screen as scr
import numpy as np
import pandas as pd
import cum_hazard as ch

import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':
    
    # the number of genes in candidate gene I
    num_candidate_I = 100

    save_path = './parametric_result_20250916_n100'
    data_path = r'../raw_data'#改：将文件名更改为我运行后保存数据的文件名
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # Protein coding gene
    # pc_symbol = pd.read_csv(r'./pc_symbol.csv', index_col=0).gene_name.values.tolist()
    pc_symbol = pd.read_csv(r'../raw_data/pc_symbol.csv').gene_name.values.tolist()
    filenames=os.listdir(data_path)
    
    for fl in filenames:
        
        if not os.path.isdir(os.path.join(data_path, fl)):
            continue
        print(fl)

        # 读取临床数据（含OS和Censor_OD等字段）
        clinical_data = pd.read_csv(os.path.join(data_path, fl, 'clinical.CSV'), keep_default_na=False)
        # 读取基因表达数据（含gene_name字段）
        exp_data = pd.read_csv(os.path.join(data_path, fl, 'data.csv'))
        exp_data.index = exp_data['gene_name'].values
        # 保留蛋白质编码基因：筛选出表达数据中既存在于当前样本中，又是蛋白质编码基因的基因。
        exp_data = exp_data.loc[np.unique(list(set(exp_data.gene_name.tolist()) & set(pc_symbol))).tolist(), :]
        print("len of exp_data: ", len(exp_data))

        # 只保留存活时间不为0的样本：去除 OS=0 的临床样本（可能为无效数据）。
        # clinical_data = clinical_data.query("OS > 0")  # 明确筛选存活时间>0的样本
        clinical_data = clinical_data[clinical_data.OS != 0]
        
        # process hazard by using exponential proportional hazard model
        # 计算累积风险（使用指数比例风险模型）
        clinical_final = ch.cum_hazard(clinical_data)
        # 复制 Censor_OD 列作为 Censor 列
        clinical_final['Censor'] = clinical_final['Censor_OD']
        
        # screen gene in candidate gene I
        result = scr.screen_step_1(clinical_final, exp_data, h_type = 'hazard_OD', threshold = num_candidate_I)
        
        # save result
        os.makedirs(os.path.join(save_path, fl), exist_ok=True)
        clinical_final.to_csv(os.path.join(save_path, fl, 'clinical_final.CSV'), sep=',', header=True, index=False)
        result.to_csv(os.path.join(save_path, fl, 'result.csv'),sep=',',index=True,header=True)

