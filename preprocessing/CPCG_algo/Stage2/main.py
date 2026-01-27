import os
import numpy as np
import pandas as pd
from pingouin import partial_corr
from itertools import combinations
import networkx as nx

def skeleton(data, alpha: float = 0.05, max_l: int = 2):
    n_nodes = data.shape[1]
    labels = data.columns.to_list()
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
                        try:
                            # 增加异常处理，防止计算偏相关时出错
                            stats = partial_corr(data=data, x=labels[x], y=labels[y], covar=cc)
                            p_value = stats.loc['pearson', 'p-val']
                            if p_value >= alpha:
                                G[x][y] = G[y][x] = False
                                break
                        except:
                            continue
        l += 1
    return np.asarray(G, dtype=int)

def cs_step_2(result_cs1, hazard_type):
    data = result_cs1.copy()
    # 确保数据是数值型
    data = data.select_dtypes(include=[np.number])
    labels = data.columns.to_list()
    
    try:
        # 1. 构建骨架
        G = skeleton(data, alpha=0.05, max_l=2)
        G_nx = nx.from_numpy_array(np.array(G))
        
        # 2. 找到与 OS (hazard_type) 相关的邻居
        if hazard_type in labels:
            OS_idx = labels.index(hazard_type)
            # 扩展到距离2的邻居（Markov Blanket 近似）
            neighbors = list(nx.single_source_shortest_path_length(G_nx, OS_idx, cutoff=2).keys())
            c_label = [labels[i] for i in neighbors]
            
            # 【修复点】移除错误的 print(fl...)，改为通用日志
            print(f"    [Stage2] PC算法筛选完成: 从 {len(labels)-1} -> {len(c_label)-1} 个基因")
            
            return data.loc[:, c_label]
        else:
            print(f"    [Stage2] 警告: 数据中找不到 {hazard_type} 列，返回原始数据")
            return data
            
    except Exception as e:
        print(f"    [Stage2] 算法运行出错: {e}，启用兜底策略(返回输入数据)")
        return data

if __name__ == "__main__":
    pass
