import os
import argparse
import sys
import pandas as pd
import numpy as np
from scipy import stats
from pingouin import partial_corr
from lifelines.statistics import logrank_test
from joblib import Parallel, delayed
import warnings
import time

warnings.filterwarnings('ignore')
np.seterr(divide='ignore', invalid='ignore')

def _process_single_gene_semi(gene_data, cd, h_type, gene_name):
    """
    单个基因的处理函数 (无状态，适合并行)
    """
    try:
        # 1. 数据准备
        # 转置并重命名，减少 merge 开销，直接赋值
        # 注意: gene_data 是 (1, N_samples)
        series_gene = gene_data.iloc[0]
        
        # 此时 cd 已经有了 case_submitter_id 作为 index
        # 我们需要确保 series_gene 的 index (sample_id) 与 cd 的 index 对齐
        
        # 快速合并：利用 Pandas 索引对齐
        # 先创建一个只包含该基因的 Series，索引为样本ID
        
        # 检查重叠样本
        common_indices = cd.index.intersection(series_gene.index)
        if len(common_indices) < 10: # 样本太少直接跳过
            return (gene_name, None, None)
            
        # 提取对应数据
        sub_cd = cd.loc[common_indices].copy()
        sub_gene_vals = series_gene.loc[common_indices].astype(float)
        
        # 赋值
        sub_cd[gene_name] = sub_gene_vals
        
        # 2. 缺失值清洗
        sub_cd = sub_cd.dropna(subset=[gene_name, 'OS', 'Censor'])
        if len(sub_cd) == 0:
            return (gene_name, None, None)

        # 3. 中位数分组
        median_val = sub_cd[gene_name].median()
        d_l = sub_cd[sub_cd[gene_name] <= median_val]
        d_h = sub_cd[sub_cd[gene_name] > median_val]

        # 分组样本检查
        if len(d_l) < 6 or len(d_h) < 6:
            return (gene_name, None, None)

        # 4. Logrank Test
        results = logrank_test(d_l['OS'], d_h['OS'], d_l['Censor'], d_h['Censor'])
        if results.p_value > 0.05:
            return (gene_name, None, None)
            
        logrank_p = results.p_value / 2

        # 5. 偏相关分析 (Partial Correlation)
        # 注意: pingouin 可能会在极少数情况下报错或卡住，增加保护
        try:
            corr_pd = partial_corr(data=sub_cd, x=gene_name, y=h_type)
            if corr_pd is not None and 'pearson' in corr_pd.index:
                corr_value = np.abs(corr_pd.loc['pearson', 'r'])
                return (gene_name, logrank_p, corr_value)
        except Exception:
            pass # 偏相关计算失败，视为无效
            
        return (gene_name, logrank_p, None)

    except Exception as e:
        # 捕获所有未知错误，防止中断进程
        return (gene_name, None, None)

def screen_step_2(clinical_final, exp_data, h_type, threshold=100, n_jobs=-1):
    """
    Stage 1 半参数化筛选 (强制并行版)
    """
    print(f"🔄 Stage1 Semi-Parametric 启动 (Target n_jobs={n_jobs})")
    
    # 预处理临床数据
    cd = clinical_final.copy()
    # 确保使用 case_submitter_id 作为索引，方便后续对齐
    if 'case_submitter_id' in cd.columns:
        cd.index = cd['case_submitter_id'].values
    elif 'case_id' in cd.columns:
        cd.index = cd['case_id'].values
        
    # 准备表达数据
    ed = exp_data.copy()
    # 确保 gene_name 是列而不是索引 (如果是索引，reset一下)
    if ed.index.name == 'gene_name':
        ed.reset_index(inplace=True)
        
    gene_names = ed['gene_name'].tolist()
    print(f"📊 待处理基因总数: {len(gene_names)}")

    # ---------------------------------------------------------
    # 并行执行核心
    # ---------------------------------------------------------
    # 使用 joblib 的 verbose 来显示进度，backend='loky' 通常最稳定
    # pre_dispatch 控制任务分发，'2*n_jobs' 可以防止内存爆满
    
    # 为了减少序列化开销，我们不直接传 ed[aa:aa+1]，而是只传 numpy array 或者 series?
    # 但为了保持逻辑简单且兼容旧代码结构，我们还是传切片，但要注意内存。
    
    # 如果 n_jobs 为 -1，但在容器中可能识别错误，建议限制最大值 (如 16)
    # 这里我们信任用户设置，但增加 batch_size 优化
    
    results = Parallel(n_jobs=n_jobs, verbose=5, pre_dispatch='2*n_jobs')(
        delayed(_process_single_gene_semi)(
            ed.iloc[aa:aa+1], # 传入单行 DataFrame
            cd,               # 传入临床数据 (所有进程共享内存)
            h_type, 
            gene_names[aa]
        ) 
        for aa in range(len(gene_names))
    )

    # ---------------------------------------------------------
    # 结果汇总
    # ---------------------------------------------------------
    table = pd.DataFrame(index=gene_names, columns=['corr', 'logrank'])
    valid_count = 0

    for gene_name, logrank_p, corr_value in results:
        if logrank_p is not None and corr_value is not None:
            table.loc[gene_name, 'logrank'] = logrank_p
            table.loc[gene_name, 'corr'] = corr_value
            valid_count += 1

    print(f"✅ 处理完成. 有效基因数: {valid_count}/{len(gene_names)}")

    # 筛选逻辑
    table = table.dropna()
    
    if table.empty:
        print("❌ 警告: 没有基因通过筛选 (可能阈值过严或数据问题)")
        return pd.DataFrame() # 返回空

    if table.shape[0] < threshold:
        print(f"⚠️  警告: 有效基因不足 ({table.shape[0]} < {threshold})，全部保留")
        threshold = table.shape[0]

    # 排序取 Top N
    # 优先级: 相关性 abs(corr) 越大越好
    table['corr'] = table['corr'].astype(float)
    corr_index = table.sort_values(by='corr', ascending=False).head(threshold).index.tolist()

    # 构建返回结果
    # 需要将选中的基因表达量合并回 clinical data
    # ed 需要设回 index
    ed_indexed = ed.set_index('gene_name')
    
    # 提取选中基因的表达量 (转置: 行=样本, 列=基因)
    selected_exp = ed_indexed.loc[corr_index].T
    
    # 合并
    result = pd.merge(cd, selected_exp, left_index=True, right_index=True, how='inner')
    
    print(f"✅ Stage1 Semi-Parametric 筛选结束，输出形状: {result.shape}")
    return result

if __name__ == '__main__':
    pass
