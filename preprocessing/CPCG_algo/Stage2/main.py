import os
import numpy as np
import pandas as pd
from pingouin import partial_corr
from itertools import combinations
import networkx as nx
from joblib import Parallel, delayed
from scipy import stats as scipy_stats

def skeleton(data, alpha: float = 0.05, max_l: int = 2):
    """
    PC 算法骨架构建（优化版）

    优化点：
    1. Depth 0 使用 df.corr() 向量化计算
    2. Depth 1+ 使用 joblib.Parallel 并行化条件独立性测试
    """
    n_nodes = data.shape[1]
    labels = data.columns.to_list()
    G = [[i != j for i in range(n_nodes)] for j in range(n_nodes)]
    pairs = [(i, j) for i in range(n_nodes) for j in range(i + 1, n_nodes)]

    # ============================================================
    # 【优化1】Depth 0: 使用 df.corr() 向量化计算
    # ============================================================
    if max_l >= 0:
        print(f"    [Depth 0] 向量化计算皮尔逊相关性矩阵...")

        # 向量化计算所有对的皮尔逊相关系数
        corr_matrix = data.corr(method='pearson')

        # 将相关系数转换为 p-value (t-test)
        n_samples = len(data)
        for x, y in pairs:
            if G[x][y]:
                r = corr_matrix.iloc[x, y]
                if pd.isna(r):
                    G[x][y] = G[y][x] = False
                    continue

                # 计算 t 统计量
                t_stat = r * np.sqrt((n_samples - 2) / (1 - r**2))
                p_value = 2 * (1 - scipy_stats.t.cdf(abs(t_stat), df=n_samples - 2))

                if p_value >= alpha:
                    G[x][y] = G[y][x] = False

        print(f"    [Depth 0] 向量化计算完成，处理 {len(pairs)} 对变量")

    # ============================================================
    # 主循环: l = 0, 1, 2, ...
    # ============================================================
    l = 0
    while l < max_l and any([any(row) for row in G]):
        print(f"    [Depth {l}] 正在处理 {sum(sum(1 for j in row if j) for row in G)//2} 条边...")

        # ============================================================
        # 【优化2】Depth 1+: 使用 joblib.Parallel 并行化
        # ============================================================
        if l >= 1:
            # 收集所有需要测试的 (x, y, K) 组合
            test_tasks = []
            for x, y in pairs:
                if G[x][y]:
                    neighbors = [i for i in range(n_nodes) if G[x][i] and i != y]
                    if len(neighbors) >= l:
                        for K in combinations(neighbors, l):
                            test_tasks.append((x, y, K))

            if test_tasks:
                # 并行执行所有条件独立性测试
                def test_ci(task):
                    x, y, K = task
                    cc = [labels[k] for k in K]
                    try:
                        stats = partial_corr(data=data, x=labels[x], y=labels[y], covar=cc)
                        p_value = stats.loc['pearson', 'p-val']
                        return (x, y, p_value)
                    except:
                        return (x, y, 0.0)  # 出错时保持连接

                results = Parallel(n_jobs=-1, verbose=5)(
                    delayed(test_ci)(task) for task in test_tasks
                )

                # 根据测试结果更新图
                for x, y, p_value in results:
                    if p_value >= alpha:
                        G[x][y] = G[y][x] = False

        else:
            # Depth 0 已在上方处理
            pass

        l += 1

    return np.asarray(G, dtype=int)

def cs_step_2(result_cs1, hazard_type):
    data = result_cs1.copy()
    # 确保数据是数值型
    data = data.select_dtypes(include=[np.number])
    labels = data.columns.to_list()

    try:
        # 1. 构建骨架
        G = skeleton(data, alpha=0.10, max_l=2)
        G_nx = nx.from_numpy_array(np.array(G))
        
        # 2. 找到与 OS (hazard_type) 相关的邻居
        if hazard_type in labels:
            OS_idx = labels.index(hazard_type)
            # 扩展到距离2的邻居（Markov Blanket 近似）
            neighbors = list(nx.single_source_shortest_path_length(G_nx, OS_idx, cutoff=2).keys())
            c_label = [labels[i] for i in neighbors]

            # 计算原始基因数（排除 OS）
            original_genes = len(labels) - 1

            # 统计保留的基因数
            retained_genes = len(c_label) - 1 if hazard_type in c_label else len(c_label)

            print(f"    [Stage2] PC算法骨架边数: {G.sum()//2}")
            print(f"    [Stage2] PC算法筛选完成: 从 {original_genes} -> {retained_genes} 个基因")

            return data.loc[:, c_label]
        else:
            print(f"    [Stage2] 警告: 数据中找不到 {hazard_type} 列，返回原始数据")
            return data
            
    except Exception as e:
        print(f"    [Stage2] 算法运行出错: {e}，启用兜底策略(返回输入数据)")
        return data

if __name__ == "__main__":
    pass
