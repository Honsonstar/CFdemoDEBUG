import os
import argparse
import sys
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from scipy.stats import f_oneway
from pingouin import partial_corr
from lifelines.statistics import logrank_test

np.seterr(divide='ignore',invalid='ignore')

def screen_step_1(clinical_final, exp_data, h_type, threshold = 100):

    cd = clinical_final.copy()
    ed = exp_data.copy()

    cd = cd[cd['Censor']==1]
    cd.index = cd['case_submitter_id'].values

    table = pd.DataFrame(index = exp_data.gene_name.tolist(), columns = ['corr'])

    for aa in range(ed.shape[0]):
        temp_data = ed[aa:aa+1].T.copy()
        name_gene = temp_data.loc['gene_name'].values
        temp_data.columns = name_gene
        temp_data = temp_data.drop(['gene_name'])

        cd = cd.merge(temp_data, how='left', left_index=True, right_index=True)
        try:
            cd[name_gene[0]] = cd[name_gene[0]].astype(float)
        except KeyError:
            cd = cd.drop(columns=name_gene[0])
            continue

        median_val = cd[name_gene[0]].median()
        d_l = cd[cd[name_gene[0]] <= median_val].copy()
        d_h = cd[cd[name_gene[0]] > median_val].copy()

        # 检查分组是否样本偏差过大
        if len(d_l) < 6 or len(d_h) < 6:
            print(f"基因 {name_gene[0]} 某组样本数过少，跳过")
            cd = cd.drop(columns=name_gene[0])
            continue

        # Logrank test
        results = logrank_test(d_l['OS'], d_h['OS'], d_l['Censor'], d_h['Censor'])

        if results.p_value > 0.01:
            cd = cd.drop(columns=name_gene[0])
            continue

        corr_pd = partial_corr(data=cd[cd['Censor']==1], x=name_gene[0], y=h_type).loc['pearson','r']
        table.loc[name_gene[0], 'corr'] = np.abs(corr_pd)

        cd = cd.drop(columns=name_gene[0])

    table = table.dropna(axis=0,how='all')
    table['corr'] = table['corr'].astype(float)
    table = table.sort_values(by = 'corr', ascending=False)

    if table.shape[0] < threshold:
        print('table.shape[0] < threshold')
        threshold = table.shape[0]
    corr_index = table.iloc[0:threshold,:].index.tolist()

    ed.index = ed['gene_name'].values

    result = pd.DataFrame(); result.index = cd[cd['Censor']==1].index
    result = pd.merge(result, cd[cd['Censor']==1][h_type], how='left', left_index=True, right_index=True)
    result = pd.merge(result, ed.loc[corr_index, :].drop(columns = 'gene_name').T, how='left', left_index=True, right_index=True)

    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stage1 Parametric Screen')
    parser.add_argument('--clinical', required=True, help='Clinical data file')
    parser.add_argument('--exp', required=True, help='Expression data file')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--h_type', default='OS', help='Hazard type')

    args = parser.parse_args()

    print("开始Stage1 Parametric筛选...")
    print(f"  临床文件: {args.clinical}")
    print(f"  表达文件: {args.exp}")
    print(f"  输出目录: {args.output}")

    # 读取数据
    clinical_final = pd.read_csv(args.clinical)
    exp_data = pd.read_csv(args.exp)

    # 运行筛选
    result = screen_step_1(clinical_final, exp_data, args.h_type)

    # 保存结果
    import os
    os.makedirs(args.output, exist_ok=True)
    output_file = os.path.join(args.output, 'stage1_parametric_result.csv')
    result.to_csv(output_file)

    print(f"✅ Stage1 Parametric完成!")
    print(f"  输出文件: {output_file}")
    print(f"  筛选基因数: {result.shape[1] - 1}")
    print(f"  样本数: {result.shape[0]}")
