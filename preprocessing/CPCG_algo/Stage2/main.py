import os

import itertools
import time

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import f_oneway
from pingouin import partial_corr
from itertools import combinations
import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Optional
from lifelines import KaplanMeierFitter, ExponentialFitter


def skeleton(data, alpha: float = 0.05, max_l: int = 2):
    n_nodes = data.shape[1]
    labels = data.columns.to_list()
    O = [[[] for _ in range(n_nodes)] for _ in range(n_nodes)]
    G = [[i != j for i in range(n_nodes)] for j in range(n_nodes)]
    pairs = [(i, j) for i in range(n_nodes) for j in range(i + 1, n_nodes)]
    l = 0
    while l < max_l and any([any(row) for row in G]):
        for x, y in pairs:
            if G[x][y]:
                neighbors = [i for i in range(n_nodes) if G[x][i] and i != y]
                if len(neighbors) >= l:
                    for K in combinations(neighbors, l):
                        cc = [labels[k] for k in K]
                        p_value = partial_corr(data=data, x=labels[x], y=labels[y], covar=cc).loc['pearson', 'p-val']
                        if p_value >= alpha:
                            G[x][y] = G[y][x] = False
                            O[x][y] = O[y][x] = list(K)
                            break
        l += 1
    return np.asarray(G, dtype=int), O


def cs_step_2(result_cs1, hazard_type):
    data = result_cs1.copy()
    labels = data.columns.to_list()
    G, O = skeleton(data, alpha=0.05, max_l=2)
    G_nx = nx.from_numpy_array(np.array(G))
    OS_idx = labels.index(hazard_type)
    neighbors = list(nx.single_source_shortest_path_length(G_nx, OS_idx, cutoff=2).keys())  # 扩展到距离2
    c_label = [labels[i] for i in neighbors]
    c_data = data.loc[:, c_label]
    print(f"{fl}: number of gene = {len(c_label) - 1}")  # -1 是去掉 OS 本身
    return c_data


if __name__ == "__main__":
    # raw data path
    data_path = r'../raw_data'#改为我自己的数据的文件名
    # parametric data path
    P_data_path = '../raw_data/para_result'
    # semi-parametric data path
    SP_data_path = '../raw_data/semi_result'
    save_path = r'./result_m2m3_base_0916_n100'

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    cs_filenames = os.listdir(P_data_path)
    fcs_filenames = os.listdir(SP_data_path)
    filenames = list(set(cs_filenames) & set(fcs_filenames))

    for fl in filenames:

        if not os.path.isdir(os.path.join(P_data_path, fl)) or \
           not os.path.isdir(os.path.join(SP_data_path, fl)):
            continue

        print(fl)
        if not os.path.exists(os.path.join(save_path, fl)):
            os.makedirs(os.path.join(save_path, fl))
        start = time.time()

        # Combine candidate gene I and candidate gene II
        P_data = pd.read_csv(os.path.join(P_data_path, fl, 'result.csv'), index_col=0)
        SP_data = pd.read_csv(os.path.join(SP_data_path, fl, 'result.csv'), index_col=0)
        gene_list = list((set(P_data.columns.tolist()) | set(SP_data.columns.tolist())) - set(['hazard_OD', 'OS']))
        print(f"number of init gene = {len(gene_list)}")

        # get raw data
        clinical_data = pd.read_csv(os.path.join(data_path, fl, 'clinical.CSV'), keep_default_na=False)
        clinical_data.index = clinical_data['case_submitter_id'].values;
        clinical_data = clinical_data[clinical_data.Censor == 1]
        exp_data = pd.read_csv(os.path.join(data_path, fl, 'data.csv'));
        exp_data.index = exp_data['gene_name'].values;
        exp_data = exp_data.loc[gene_list, :];
        exp_data = exp_data.drop(columns='gene_name');
        exp_data = exp_data.T

        # combine data
        data = pd.merge(clinical_data.OS, exp_data, right_index=True, left_index=True)
        # data.to_csv(os.path.join(save_path, fl, fl + '_data.csv'))


        data = data.loc[:, data.corr()['OS'].abs().sort_values(ascending=False).index.tolist()]

        # output skeleton
        result = cs_step_2(data, hazard_type='OS')
        # save result
        result.to_csv(os.path.join(save_path, fl, fl + '_M2M3base_0916.csv'), sep=',', index=True, header=True)
        end = time.time()

        print('time:{}'.format(end - start))


