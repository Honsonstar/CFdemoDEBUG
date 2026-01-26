import os
import random
import numpy as np
import pandas as pd
import screen as scr
import pandas as pd
import cum_hazard as ch

if __name__ == '__main__':
    
    # the number of genes in candidate gene I
    num_candidate_I = 100

    save_path = './semi-parametric_result_20250916_n100'
    data_path = r'../raw_data'#改为我自己保存数据的路径
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # Protein coding gene
    pc_symbol = pd.read_csv(r'../raw_data/pc_symbol.csv').gene_name.values.tolist()
    filenames=os.listdir(data_path)
         
    for fl in filenames:
        
        if not os.path.isdir(os.path.join(data_path, fl)):
            continue
        print(fl)
        clinical_data = pd.read_csv(os.path.join(data_path, fl, 'clinical.CSV'), keep_default_na=False)
        exp_data = pd.read_csv(os.path.join(data_path, fl, 'data.csv')); exp_data.index = exp_data['gene_name'].values;

        # 保留蛋白质编码基因：筛选出表达数据中既存在于当前样本中，又是蛋白质编码基因的基因。
        exp_data = exp_data.loc[np.unique(list(set(exp_data.gene_name.tolist()) & set(pc_symbol))).tolist(), :]

        clinical_data = clinical_data[clinical_data.OS != 0]
        # calculated risk
        clinical_final = ch.cum_hazard(clinical_data)
        
        # screen gene in candidate gene I
        result = scr.screen_step_2(clinical_final, exp_data, h_type = 'hazard_OD', threshold = num_candidate_I)
        
        # save result
        os.makedirs(os.path.join(save_path, fl), exist_ok=True)
        clinical_final.to_csv(os.path.join(save_path, fl, 'clinical_final.CSV'), sep=',', header=True, index=False)
        result.to_csv(os.path.join(save_path, fl, 'result.csv'),sep=',',index=True,header=True)

