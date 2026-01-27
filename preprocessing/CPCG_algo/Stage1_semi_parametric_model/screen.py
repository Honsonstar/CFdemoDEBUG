import os
import argparse
import sys
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import f_oneway
from pingouin import partial_corr
import statsmodels.api as sm
from lifelines.statistics import logrank_test
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings('ignore')

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("⚠️  tqdm未安装，跳过进度条")

np.seterr(divide='ignore',invalid='ignore')

def _process_single_gene_semi(gene_data, cd, h_type, gene_name):
    """
    并行处理单个基因的筛选（半参数化版本）

    Args:
        gene_data: 单个基因的表达数据
        cd: 临床数据
        h_type: 生存类型
        gene_name: 基因名

    Returns:
        tuple: (gene_name, logrank_p, corr_value) 或 (gene_name, None, None) 如果基因被跳过
    """
    try:
        # 合并基因表达数据
        temp_data = gene_data.T.copy()
        temp_data.columns = [gene_name]
        temp_data = temp_data.drop(['gene_name'])

        cd_copy = cd.copy()
        cd_copy = cd_copy.merge(temp_data, how='left', left_index=True, right_index=True)

        # 检查数据类型并转换
        if gene_name not in cd_copy.columns:
            return (gene_name, None, None)

        try:
            cd_copy[gene_name] = cd_copy[gene_name].astype(float)
        except (KeyError, ValueError):
            return (gene_name, None, None)

        # 检查缺失值
        cd_copy = cd_copy.dropna(subset=[gene_name, 'OS', 'Censor'])

        if len(cd_copy) == 0:
            return (gene_name, None, None)

        # 中位数分组
        median_val = cd_copy[gene_name].median()
        d_l = cd_copy[cd_copy[gene_name] <= median_val].copy()
        d_h = cd_copy[cd_copy[gene_name] > median_val].copy()

        # 检查分组样本数
        if len(d_l) < 6 or len(d_h) < 6:
            return (gene_name, None, None)

        # Logrank test
        results = logrank_test(d_l['OS'], d_h['OS'], d_l['Censor'], d_h['Censor'])

        if results.p_value > 0.01:
            return (gene_name, None, None)

        logrank_p = results.p_value / 2

        # 偏相关分析
        corr_pd = partial_corr(data=cd_copy, x=gene_name, y=h_type)
        if corr_pd is not None and 'pearson' in corr_pd.index and 'r' in corr_pd.columns:
            corr_value = np.abs(corr_pd.loc['pearson', 'r'])
            return (gene_name, logrank_p, corr_value)
        else:
            return (gene_name, logrank_p, None)

    except Exception as e:
        return (gene_name, None, None)

def screen_step_2(clinical_final, exp_data, h_type, threshold=100, n_jobs=-1):
    """
    Stage1半参数化筛选（并行化版本）

    Args:
        clinical_final: 临床数据
        exp_data: 表达数据
        h_type: 生存类型
        threshold: 基因筛选阈值
        n_jobs: 并行作业数，-1表示使用所有CPU核心

    Returns:
        筛选后的结果数据
    """
    print(f"🔄 Stage1 Semi-Parametric筛选启动 (并行作业数: {n_jobs if n_jobs != -1 else '所有核心'})")

    cd = clinical_final.copy()
    ed = exp_data.copy()

    cd.index = cd['case_submitter_id'].values

    # 准备基因列表
    gene_names = ed['gene_name'].tolist()

    # 并行处理所有基因
    print(f"📊 正在并行处理 {len(gene_names)} 个基因...")

    # 使用tqdm进度条（如果有的话）
    if HAS_TQDM:
        pbar = tqdm(total=len(gene_names), desc="筛选基因", unit="个")
        results = []
        for aa in range(len(gene_names)):
            results.append(_process_single_gene_semi(ed[aa:aa+1], cd, h_type, gene_names[aa]))
            pbar.update(1)
        pbar.close()
    else:
        results = Parallel(n_jobs=n_jobs)(
            delayed(_process_single_gene_semi)(ed[aa:aa+1], cd, h_type, gene_names[aa])
            for aa in range(len(gene_names))
        )

    # 整理结果
    table = pd.DataFrame(index=gene_names, columns=['corr', 'logrank'])
    valid_count = 0

    for gene_name, logrank_p, corr_value in results:
        if logrank_p is not None and corr_value is not None:
            table.loc[gene_name, 'logrank'] = logrank_p
            table.loc[gene_name, 'corr'] = corr_value
            valid_count += 1

    print(f"✅ 并行处理完成，有效基因: {valid_count}/{len(gene_names)}")

    # 排序并筛选
    table = table.dropna(axis=0, how='all')
    table['corr'] = table['corr'].astype(float)
    table['logrank'] = table['logrank'].astype(float)

    if table.shape[0] < threshold:
        print(f'⚠️  有效基因数({table.shape[0]}) < 阈值({threshold})，调整阈值')
        threshold = table.shape[0]

    if threshold == 0:
        print("❌ 没有基因通过筛选，返回空结果")
        return pd.DataFrame()

    # 按相关性排序并筛选前threshold个
    corr_index = table.sort_values(by='corr', ascending=False).iloc[0:threshold, :].index.tolist()

    # 构建最终结果
    ed.index = ed['gene_name'].values

    result = pd.DataFrame()
    result.index = cd.index
    result = pd.merge(result, cd[['Censor', h_type, 'OS']], how='left', left_index=True, right_index=True)
    result = pd.merge(
        result,
        ed.loc[corr_index, :].drop(columns='gene_name').T,
        how='left',
        left_index=True,
        right_index=True
    )

    print(f"✅ Stage1 Semi-Parametric筛选完成，保留 {len(corr_index)} 个基因")
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stage1 Semi-Parametric Screen')
    parser.add_argument('--clinical', required=True, help='Clinical data file')
    parser.add_argument('--exp', required=True, help='Expression data file')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--h_type', default='OS', help='Hazard type')

    args = parser.parse_args()

    print("开始Stage1 Semi-Parametric筛选...")
    print(f"  临床文件: {args.clinical}")
    print(f"  表达文件: {args.exp}")
    print(f"  输出目录: {args.output}")

    # 读取数据
    clinical_final = pd.read_csv(args.clinical)
    exp_data = pd.read_csv(args.exp)

    # 运行筛选
    result = screen_step_2(clinical_final, exp_data, args.h_type)

    # 保存结果
    import os
    os.makedirs(args.output, exist_ok=True)
    output_file = os.path.join(args.output, 'stage1_parametric_result.csv')
    result.to_csv(output_file)

    print(f"✅ Stage1 Semi-Parametric完成!")
    print(f"  输出文件: {output_file}")
    print(f"  筛选基因数: {result.shape[1] - 2}")
    print(f"  样本数: {result.shape[0]}")
