"""
================================================================================
find_genes_stable.py - 鍩哄洜绋冲畾鎬ч獙璇佷笌棰戠巼鎺掑簭 Pipeline
================================================================================

銆愭枃浠朵綔鐢ㄣ€?瀵规瘡鎶樿缁冮泦杩涜澶氭鎶芥牱杩愯 find_genes_gci2锛堥┈灏旂澶鎵╁睍鐗堬級锛?缁熻鍩哄洜鍑虹幇棰戠巼骞舵帓搴忚緭鍑?top100锛堜笉杩涜闄嶇淮锛岀洿鎺ヤ娇鐢ㄥ師濮嬫暟鎹級

銆愰噸瑕佹洿鏂?- 2026-03-20銆?- 鍘熺増鏈皟鐢?find_genes_gci (绗竴鐗?
- 鏂扮増鏈皟鐢?find_genes_gci2 (椹皵绉戝か姣墿灞曠増)
- find_genes_gci2 鍦?find_genes_gci 鐨勭粨鏋滃熀纭€涓婏紝寰€澶栧鍋氫竴灞傞┈灏旂澶锛?  1. 鍏堣繍琛?find_genes_gci 寰楀埌鍒濆鍥犳灉鍩哄洜
  2. 杞祦灏嗙瓫閫夊嚭鏉ョ殑鍩哄洜浣滀负鐩爣鍙橀噺杩涜绛涢€?  3. 灏嗘墍鏈夊緱鍒扮殑鍩哄洜鐨勫苟闆嗕綔涓?found_genes 杈撳嚭

銆愮畻娉曟祦绋嬨€?1. 鍔犺浇姣忔姌鐨勮缁冩暟鎹?(train_fold*.mat)
2. 瀵硅fold杩涜N娆℃娊鏍凤紙Bootstrap鎴朢andom锛?3. 杩愯 find_genes_gci2 绛涢€夊洜鏋滃熀鍥?4. 浠呭熀浜庤fold鐨凬娆¤繍琛岀粨鏋滐紝缁熻鍩哄洜鍑虹幇棰戠巼骞舵帓搴?5. 瀵规瘡涓猣old鍒嗗埆杈撳嚭Top K鍩哄洜

銆愪娇鐢ㄦ柟娉曘€?```bash
python find_genes_stable.py
```

銆愰厤缃弬鏁般€?- SAMPLE_MODE: 'bootstrap' (鏈夋斁鍥? / 'random' (鏃犳斁鍥? / 'partitioned' (鍒嗗尯鎶芥牱)
- SAMPLE_RATIO: 鎶芥牱姣斾緥 (浠?random 妯″紡鐢熸晥)
- NUM_BOOTSTRAP: 杩唬娆℃暟 (bootstrap/random 妯″紡)
- NUM_PARTITIONS: 鍒嗗尯鏁伴噺 (浠?partitioned 妯″紡)
- TOP_K: 杈撳嚭鍓岾涓熀鍥?
銆愯緭鍑烘枃浠躲€?- stable_genes_fold{fold}_top100.mat: 璇old鐨勫墠100鍩哄洜 (MATLAB鏍煎紡)
- stable_genes_fold{fold}_top100.txt: 璇old鐨勫墠100鍩哄洜 (鏂囨湰鏍煎紡)

銆愪緷璧栥€?- numpy
- scipy.io
- find_genes_gci2 (鍚岀洰褰曚笅)

================================================================================
"""

import os
import sys
import datetime
import numpy as np
from scipy.io import savemat, loadmat
from collections import Counter
import time
import pandas as pd
from multiprocessing import Pool, cpu_count

# 娣诲姞褰撳墠鐩綍鍒拌矾寰勶紝鐢ㄤ簬瀵煎叆find_genes_gci妯″潡
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# ================================================================================
# 鍘熷鐗堟湰浣跨敤 find_genes_gci (绗竴鐗?
# from find_genes_gci import find_genes_gci, load_data
#
# 鏂扮増鏈娇鐢?find_genes_gci2 (椹皵绉戝か姣墿灞曠増)
# 鏀瑰姩璇存槑锛?#   - find_genes_gci2 鍦?find_genes_gci 鐨勭粨鏋滃熀纭€涓婏紝寰€澶栧鍋氫竴灞傞┈灏旂澶
#   - 杞祦灏嗙瓫閫夊嚭鏉ョ殑鍩哄洜浣滀负鐩爣鍙橀噺杩涜绛涢€?#   - 灏嗘墍鏈夊緱鍒扮殑鍩哄洜鐨勫苟闆嗕綔涓?found_genes 杈撳嚭
# ================================================================================
# 鍔ㄦ€佸鍏ワ細鏍规嵁 MARKOV_BLANKET_LAYER 閰嶇疆閫夋嫨浣跨敤鍝釜鍑芥暟
from find_genes_gci import load_data


# ================================================================================
# ========== 閰嶇疆鍖哄煙 =============
# ================================================================================

# 鐧岀棁绫诲瀷 (brca, blca, hnsc, stad, coadread)
CANCER_TYPE = 'coadread' 


# 鏁版嵁鐩綍
DATA_DIR = rf'/root/autodl-tmp/newcfdemo/CFdemo_gene_text_copy/splits/CGI_nested_cv/{CANCER_TYPE}'
CURRENT_DATE = datetime.datetime.now().strftime('%Y-%m-%d')

# 鑾峰彇褰撳墠鏃ユ湡锛岀敤浜庤緭鍑虹洰褰?CURRENT_DATE = datetime.datetime.now().strftime('%Y-%m-%d')

# 杈撳嚭鐩綍
OUTPUT_DIR = rf'/root/autodl-tmp/newcfdemo/CFdemo_gene_text_copy/preprocessing/CGI_py2/plot_cgi/{CANCER_TYPE}'

# CSV鐗瑰緛鏂囦欢杈撳嚭鐩綍
CSV_OUTPUT_DIR = rf'/root/autodl-tmp/newcfdemo/CFdemo_gene_text_copy/preprocessing/CGI_py2/features/stable/{CANCER_TYPE}'

# 浜ゅ弶楠岃瘉鎶樻暟
NUM_FOLDS = 5

# 鎶芥牱妯″紡: 'bootstrap' (鏈夋斁鍥? / 'random' (鏃犳斁鍥? / 'partitioned' (鍒嗗尯鎶芥牱)
SAMPLE_MODE = 'random'

# 姣忔鎶藉彇鐨勬瘮渚?(浠呭湪 random 妯″紡涓嬬敓鏁?
# bootstrap 妯″紡涓嬪拷鐣ユ鍙傛暟锛屽浐瀹氫负1.0
SAMPLE_RATIO = 0.9

# 杩唬娆℃暟 (bootstrap/random 妯″紡)
NUM_BOOTSTRAP = 50
NUM_PARTITIONS = 20
TOP_K = 100
GENE_FREQ_THRESHOLD = 1

# 鍒嗗尯鏁伴噺 (浠?partitioned 妯″紡鐢熸晥)
# 灏嗘暟鎹垎鎴恘浠斤紝姣忔鍙杗-1浠斤紝寰幆n杞?NUM_PARTITIONS = 20

# 杈撳嚭鍓岾涓熀鍥?TOP_K = 100

# 鍩哄洜棰戞闃堝€硷紙鐢ㄤ簬鏈€缁堢壒寰佸鍑鸿鍒欙級
# - <= 1: 鎸夐娆℃帓搴忚緭鍑哄墠 TOP_K 涓熀鍥?# - > 1 : 杈撳嚭鈥滃嚭鐜伴娆?>= 璇ラ槇鍊尖€濈殑鎵€鏈夊熀鍥?GENE_FREQ_THRESHOLD = 1

# 闅忔満绉嶅瓙
RANDOM_SEED = 42
ALPHA = 0.05

# 鏄捐憲鎬ф按骞?ALPHA = 0.05

# 椹皵绉戝か姣眰鏁? 1 (涓€灞? 鎴?2 (涓ゅ眰锛岄┈灏旂澶鎵╁睍鐗?
MARKOV_BLANKET_LAYER = 1

# 骞惰杩涚▼鏁?(0 鎴?None 琛ㄧず浣跨敤鎵€鏈夊彲鐢–PU鏍稿績)
NUM_PROCESSES = 0

# 鏍规嵁閰嶇疆鍔ㄦ€佸鍏ュ搴旂殑鍑芥暟
if MARKOV_BLANKET_LAYER == 1:
    from find_genes_gci import find_genes_gci as find_genes_gci_func
    print(f"浣跨敤涓€灞傞┈灏旂澶 (find_genes_gci)")
elif MARKOV_BLANKET_LAYER == 2:
    from find_genes_gci2 import find_genes_gci2 as find_genes_gci_func
    print(f"浣跨敤涓ゅ眰椹皵绉戝か姣?(find_genes_gci2)")
else:
    raise ValueError(f"MARKOV_BLANKET_LAYER 蹇呴』鏄?1 鎴?2锛屽綋鍓嶅€? {MARKOV_BLANKET_LAYER}")

# ================================================================================


def sample_data(data: np.ndarray, mode: str, ratio: float = 1.0, seed: int = None,
                iteration: int = 0, n_partitions: int = 10) -> np.ndarray:
    """
    Data sampling helper.
    Supports: bootstrap / random / partitioned.
    """
    n_samples = data.shape[0]

    if seed is not None:
        np.random.seed(seed)

    if mode == 'bootstrap':
        n_select = n_samples
        indices = np.random.choice(n_samples, size=n_select, replace=True)
    elif mode == 'random':
        n_select = int(n_samples * ratio)
        indices = np.random.choice(n_samples, size=n_select, replace=False)
    elif mode == 'partitioned':
        np.random.seed(seed if seed is not None else 42)
        shuffled_indices = np.random.permutation(n_samples)
        chunk_size = n_samples // n_partitions

        val_start = iteration * chunk_size
        val_end = val_start + chunk_size if iteration < n_partitions - 1 else n_samples
        train_indices = np.concatenate([
            shuffled_indices[:val_start],
            shuffled_indices[val_end:]
        ])
        return data[train_indices]
    else:
        raise ValueError(
            f"Invalid sampling mode: {mode}. Must be 'bootstrap', 'random', or 'partitioned'."
        )

    return data[indices]

_POOL_CONTEXT = {}


def _init_pool_worker(data: np.ndarray, sample_mode: str, sample_ratio: float,
                      num_partitions: int, alpha: float):
    """Initialize per-process context to reduce argument passing overhead."""
    global _POOL_CONTEXT
    _POOL_CONTEXT = {
        'data': data,
        'sample_mode': sample_mode,
        'sample_ratio': sample_ratio,
        'num_partitions': num_partitions,
        'alpha': alpha
    }


def _run_iteration_task(task: tuple) -> dict:
    """
    Execute one iteration task in worker process.
    task = (fold, iteration, iter_display, total_iterations, iter_seed)
    """
    fold, iteration, iter_display, total_iterations, iter_seed = task
    sampled_data = sample_data(
        _POOL_CONTEXT['data'],
        mode=_POOL_CONTEXT['sample_mode'],
        ratio=_POOL_CONTEXT['sample_ratio'],
        seed=iter_seed,
        iteration=iteration,
        n_partitions=_POOL_CONTEXT['num_partitions']
    )
    iter_start = time.time()
    results = find_genes_gci_func(sampled_data, alpha=_POOL_CONTEXT['alpha'])
    iter_time = time.time() - iter_start
    found_genes = results['found_genes']
    return {
        'fold': fold,
        'iteration': iteration,
        'iter_display': iter_display,
        'total_iterations': total_iterations,
        'iter_seed': iter_seed,
        'sample_size': sampled_data.shape[0],
        'found_genes': found_genes,
        'iter_time': iter_time
    }


def load_train_sample_ids(cancer_type: str, fold: int, data_dir: str) -> list:
    """
    浠?nested_splits_{fold}.csv 鏂囦欢涓鍙栬缁冮泦鏍锋湰ID

    鍙傛暟:
        cancer_type: 鐧岀棁绫诲瀷
        fold: 鎶樻暟
        data_dir: 鏁版嵁鐩綍

    杩斿洖:
        list: 璁粌闆嗘牱鏈琁D鍒楄〃
    """
    splits_file = os.path.join(data_dir, f'nested_splits_{fold}.csv')
    if not os.path.exists(splits_file):
        print(f"    璀﹀憡: 鏈壘鍒版牱鏈琁D鏂囦欢 {splits_file}")
        return None

    df = pd.read_csv(splits_file)
    train_samples = df['train'].dropna().tolist()
    return train_samples


def load_all_sample_ids(cancer_type: str, fold: int, data_dir: str) -> list:
    """
    浠?nested_splits_{fold}.csv 鏂囦欢涓鍙栨墍鏈夋牱鏈琁D锛坱rain + val + test锛?
    鍙傛暟:
        cancer_type: 鐧岀棁绫诲瀷
        fold: 鎶樻暟
        data_dir: 鏁版嵁鐩綍

    杩斿洖:
        list: 鎵€鏈夋牱鏈琁D鍒楄〃锛堟寜 train -> val -> test 椤哄簭锛?    """
    splits_file = os.path.join(data_dir, f'nested_splits_{fold}.csv')
    if not os.path.exists(splits_file):
        print(f"    璀﹀憡: 鏈壘鍒版牱鏈琁D鏂囦欢 {splits_file}")
        return None

    df = pd.read_csv(splits_file)
    # 渚濇璇诲彇 train, val, test 鍒楋紝鎷兼帴鎴愬畬鏁村垪琛?    all_samples = []
    for col in ['train', 'val', 'test']:
        samples = df[col].dropna().tolist()
        all_samples.extend(samples)

    return all_samples


def load_full_data(cancer_type: str) -> tuple:
    """
    鍔犺浇瀹屾暣鏁版嵁锛堝寘鍚?train + val + test 鎵€鏈夋牱鏈級

    鍙傛暟:
        cancer_type: 鐧岀棁绫诲瀷

    杩斿洖:
        tuple: (data_matrix, gene_names, patient_ids)
            - data_matrix: 瀹屾暣鏁版嵁鐭╅樀 (n_samples, n_genes+1)锛屾渶鍚庝竴鍒楁槸time
            - gene_names: 鍩哄洜鍚嶅垪琛?            - patient_ids: 涓?data_matrix 琛屼竴涓€瀵瑰簲鐨?patient_id 鍒楄〃
    """
    # 瀹屾暣鏁版嵁鏂囦欢璺緞锛堟敞鎰忥細鍦?coadread 瀛愮洰褰曚笅锛?    data_dir = '/root/autodl-tmp/newcfdemo/CFdemo_gene_text_copy/preprocessing/CGI/data'
    full_data_path = os.path.join(data_dir, cancer_type, f'{cancer_type}_data_with_id.csv')

    if not os.path.exists(full_data_path):
        print(f"    璀﹀憡: 鏈壘鍒板畬鏁存暟鎹枃浠?{full_data_path}")
        return None, None, None

    df = pd.read_csv(full_data_path)

    # 鎻愬彇鍩哄洜鍚嶏紙璺宠繃绗竴鍒?patient_id 鍜屾渶鍚庝竴鍒?time锛?    gene_names = df.columns[1:-1].tolist()
    patient_ids = df['patient_id'].astype(str).tolist()

    # 鎻愬彇鏁版嵁鐭╅樀锛堣烦杩?patient_id 鍒楋紝淇濈暀鍩哄洜鍜?time锛?    data_matrix = df.iloc[:, 1:].values  # shape: (n_samples, n_genes+1)

    print(f"    宸插姞杞藉畬鏁存暟鎹? {data_matrix.shape[0]} 鏍锋湰, {len(gene_names)} 鍩哄洜")
    return data_matrix, gene_names, patient_ids


def load_gene_names(cancer_type: str) -> list:
    """
    浠庡師濮婥SV鏂囦欢鍔犺浇鍩哄洜鍚嶅垪琛?
    鍙傛暟:
        cancer_type: 鐧岀棁绫诲瀷

    杩斿洖:
        list: 鍩哄洜鍚嶅垪琛紙鎸夊垪绱㈠紩椤哄簭锛?    """
    # 鍘熷鏁版嵁鏂囦欢璺緞
    original_csv_path = f'/root/autodl-tmp/newcfdemo/CFdemo_gene_text_copy/preprocessing/CGI/data/{cancer_type}/{cancer_type}_data_with_id.csv'

    if not os.path.exists(original_csv_path):
        print(f"    璀﹀憡: 鏈壘鍒板師濮嬫暟鎹枃浠?{original_csv_path}")
        return None

    df = pd.read_csv(original_csv_path, nrows=0)
    gene_names = df.columns[1:-1].tolist()  # 璺宠繃patient_id鍜宼ime
    print(f"    Loaded gene names: {len(gene_names)}")
    return gene_names


def save_stable_genes_csv(data: np.ndarray, gene_indices: list, sample_ids: list,
                          output_file: str, gene_names: list = None,
                          index_name: str = 'gene_name'):
    """
    灏嗗熀鍥犺〃杈炬暟鎹繚瀛樹负 CSV 鏍煎紡

    鍙傛暟:
        data: 鍘熷鏁版嵁鐭╅樀 (n_samples, n_genes)
        gene_indices: 瑕佷繚瀛樼殑鍩哄洜绱㈠紩鍒楄〃
        sample_ids: 鏍锋湰ID鍒楄〃锛堢敤浜庡垪鍚嶏級
        output_file: 杈撳嚭鏂囦欢璺緞
        gene_names: 鍩哄洜鍚嶇О鍒楄〃锛堝彲閫夛紝鐢ㄤ簬琛屽悕锛?        index_name: 绱㈠紩鍚嶇О锛堥粯璁?'gene_name'锛?    """
    # 鏋勫缓鏁版嵁
    if sample_ids is None:
        sample_ids = [f'sample_{i}' for i in range(data.shape[0])]

    # 鍒涘缓DataFrame
    # 琛? 鍩哄洜, 鍒? 鏍锋湰
    rows = []
    for gene_idx in gene_indices:
        gene_data = data[:, gene_idx]
        if gene_names and gene_idx < len(gene_names):
            gene_name = gene_names[gene_idx]
        else:
            gene_name = f'gene_{gene_idx}'
        row = [gene_name] + list(gene_data)
        rows.append(row)

    # 鏋勫缓鍒楀悕
    columns = ['gene_name'] + sample_ids

    # 鍒涘缓DataFrame
    df = pd.DataFrame(rows, columns=columns)

    # 璁剧疆绱㈠紩鍚嶇О
    df.index.name = index_name

    # 淇濆瓨CSV锛堝寘鍚储寮曞悕绉帮級
    df.to_csv(output_file, index=True)
    print(f"    宸蹭繚瀛楥SV: {output_file}")


def run_stable_genes_pipeline():
    """
    涓诲嚱鏁帮細杩愯鍩哄洜绋冲畾鎬ч獙璇乸ipeline
    """
    # 纭繚杈撳嚭鐩綍瀛樺湪
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 鏍规嵁鎶芥牱妯″紡纭畾杩唬娆℃暟
    if SAMPLE_MODE == 'partitioned':
        num_iterations = NUM_PARTITIONS
        sample_ratio_display = f"{NUM_PARTITIONS - 1}/{NUM_PARTITIONS}"
    elif SAMPLE_MODE == 'random':
        num_iterations = NUM_BOOTSTRAP
        sample_ratio_display = f"{int(SAMPLE_RATIO * 100)}%"
    else:  # bootstrap
        num_iterations = NUM_BOOTSTRAP
        sample_ratio_display = "100% (bootstrap)"

    include_full_iteration = (SAMPLE_MODE == 'random')
    total_iterations = num_iterations + (1 if include_full_iteration else 0)

    print("=" * 70)
    print(f"  鍩哄洜绋冲畾鎬ч獙璇?Pipeline - {CANCER_TYPE}")
    print("=" * 70)
    print(f"  鎶芥牱妯″紡: {SAMPLE_MODE}")
    if SAMPLE_MODE == 'partitioned':
        print(f"  鍒嗗尯鏁伴噺: {NUM_PARTITIONS}, 姣忔鍙? {sample_ratio_display}")
    elif SAMPLE_MODE == 'random':
        print(f"  鎶芥牱姣斾緥: {sample_ratio_display} (鏃犳斁鍥?")
    else:
        print(f"  鎶芥牱姣斾緥: {sample_ratio_display}")
    print(f"  浜ゅ弶楠岃瘉鎶樻暟: {NUM_FOLDS}")
    if include_full_iteration:
        print(f"  杩唬娆℃暟: {total_iterations} (鍏ㄩ泦1娆?+ 闅忔満鎶芥牱{num_iterations}娆?")
    else:
        print(f"  杩唬娆℃暟: {num_iterations}")
    if GENE_FREQ_THRESHOLD <= 1:
        print(f"  Gene selection: top {TOP_K} by frequency")
    else:
        print(f"  Gene selection: frequency >= {GENE_FREQ_THRESHOLD}")
    print(f"  杈撳嚭鐩綍: {OUTPUT_DIR}")
    print("=" * 70)

    total_start_time = time.time()
    available_cpus = cpu_count()
    if NUM_PROCESSES is None or NUM_PROCESSES <= 0:
        effective_num_processes = available_cpus
    else:
        effective_num_processes = min(NUM_PROCESSES, available_cpus)
    print(f"  CPU cores: {available_cpus}, worker processes: {effective_num_processes}")

    # 閬嶅巻姣忎釜fold
    for fold in range(NUM_FOLDS):
        print(f"\n{'='*60}")
        print(f"  Fold {fold + 1}/{NUM_FOLDS}")
        print(f"{'='*60}")

        mat_file = os.path.join(DATA_DIR, f'train_fold{fold}.mat')
        print(f"  鍔犺浇鏁版嵁: {mat_file}")

        if not os.path.exists(mat_file):
            print(f"  璀﹀憡: 鏁版嵁鏂囦欢涓嶅瓨鍦紝璺宠繃 fold {fold}")
            continue

        data = load_data(mat_file)
        n_samples, n_cols = data.shape
        n_genes = n_cols - 1
        print(f"  鍘熷鏁版嵁: {n_samples} 鏍锋湰 脳 {n_genes} 鍩哄洜")

        fold_all_genes = []

        # random 妯″紡涓嬪厛璺戜竴娆″叏闆嗚凯浠ｏ紙涓嶆娊鏍凤級
        if include_full_iteration:
            print(f"\n  Fold {fold}, Iter 1/{total_iterations} (full training set)")
            sampled_data = data
            print(f"    鏍锋湰鏁? {sampled_data.shape[0]}")

            iter_start = time.time()
            results = find_genes_gci_func(sampled_data, alpha=ALPHA)
            iter_time = time.time() - iter_start

            found_genes = results['found_genes']
            print(f"    鍙戠幇鍩哄洜鏁? {len(found_genes)}, 鑰楁椂: {iter_time:.2f}s")
            fold_all_genes.extend(found_genes)

        # 杩唬鎶芥牱
        iteration_tasks = []
        for iteration in range(num_iterations):
            iter_seed = RANDOM_SEED + fold * 10000 + iteration if RANDOM_SEED else None
            iter_display = iteration + 1 + (1 if include_full_iteration else 0)
            iteration_tasks.append((fold, iteration, iter_display, total_iterations, iter_seed))

        if effective_num_processes > 1 and len(iteration_tasks) > 1:
            with Pool(
                processes=effective_num_processes,
                initializer=_init_pool_worker,
                initargs=(data, SAMPLE_MODE, SAMPLE_RATIO, NUM_PARTITIONS, ALPHA)
            ) as pool:
                iter_results = pool.map(_run_iteration_task, iteration_tasks)
        else:
            _init_pool_worker(data, SAMPLE_MODE, SAMPLE_RATIO, NUM_PARTITIONS, ALPHA)
            iter_results = [_run_iteration_task(task) for task in iteration_tasks]

        for iter_result in iter_results:
            print(
                f"\n  Fold {iter_result['fold']}, Iter "
                f"{iter_result['iter_display']}/{iter_result['total_iterations']} "
                f"(seed={iter_result['iter_seed']})"
            )
            print(f"    Sample count: {iter_result['sample_size']}")
            print(
                f"    Found genes: {len(iter_result['found_genes'])}, "
                f"time: {iter_result['iter_time']:.2f}s"
            )
            fold_all_genes.extend(iter_result['found_genes'])

        # 璇old鐨勭粺璁′笌杈撳嚭
        print(f"\n{'='*60}")
        print(f"  Fold {fold} 缁熻缁撴灉")
        print(f"{'='*60}")

        gene_counts = Counter(fold_all_genes)
        total_unique_genes = len(gene_counts)

        print(f"  杩唬娆℃暟: {total_iterations}")
        print(f"  璇old绛涢€夊嚭鐨勫熀鍥犳€绘暟锛堝惈閲嶅锛? {len(fold_all_genes)}")
        print(f"  鍞竴鍩哄洜鏁? {total_unique_genes}")

        sorted_genes = sorted(gene_counts.items(), key=lambda x: -x[1])
        if GENE_FREQ_THRESHOLD <= 1:
            selected_genes = sorted_genes[:TOP_K]
            select_rule_desc = f"Top {TOP_K} genes by frequency"
        else:
            selected_genes = [g for g in sorted_genes if g[1] >= GENE_FREQ_THRESHOLD]
            select_rule_desc = f"Genes with frequency >= {GENE_FREQ_THRESHOLD}"

        if len(selected_genes) == 0:
            raise ValueError(
                f"Fold {fold} 鍦ㄥ綋鍓嶉槇鍊奸厤缃笅鏈瓫鍑轰换浣曞熀鍥? GENE_FREQ_THRESHOLD={GENE_FREQ_THRESHOLD}"
            )

        print(f"\n  Fold {fold} {select_rule_desc}:")
        print(f"  {'鎺掑悕':<6} {'鍩哄洜绱㈠紩':<12} {'鍑虹幇娆℃暟':<12} {'鍑虹幇棰戠巼':<10}")
        print(f"  {'-'*45}")

        for rank, (gene_idx, count) in enumerate(selected_genes, 1):
            freq = count / total_iterations * 100
            print(f"  {rank:<6} {gene_idx:<12} {count:<12} {freq:.1f}%")

        # 淇濆瓨缁撴灉
        # 1. MATLAB鏍煎紡
        top_genes_mat = {
            'top_genes_indices': [g[0] for g in selected_genes],
            'top_genes_counts': [g[1] for g in selected_genes],
            'all_gene_counts': gene_counts,
            'num_iterations': total_iterations,
            'num_bootstrap_iterations': num_iterations,
            'include_full_iteration': include_full_iteration,
            'sample_mode': SAMPLE_MODE,
            'sample_ratio': SAMPLE_RATIO if SAMPLE_MODE == 'random' else (NUM_PARTITIONS - 1) / NUM_PARTITIONS if SAMPLE_MODE == 'partitioned' else 1.0,
            'num_partitions': NUM_PARTITIONS if SAMPLE_MODE == 'partitioned' else 0,
            'cancer_type': CANCER_TYPE,
            'fold': fold,
            'gene_freq_threshold': GENE_FREQ_THRESHOLD,
            'top_k_when_threshold_disabled': TOP_K
        }
        mat_output_file = os.path.join(OUTPUT_DIR, f'stable_genes_fold{fold}_top100.mat')
        savemat(mat_output_file, top_genes_mat)
        print(f"\n  宸蹭繚瀛? {mat_output_file}")

        # 2. 鏂囨湰鏍煎紡 (鏂板疄楠屽湪涓婇潰)
        txt_output_file = os.path.join(OUTPUT_DIR, f'stable_genes_fold{fold}_top100.txt')
        if SAMPLE_MODE == 'random':
            sample_mode_desc = f"Random (鏃犳斁鍥炴娊鏍? 姣斾緥={SAMPLE_RATIO})"
        elif SAMPLE_MODE == 'partitioned':
            sample_mode_desc = f"Partitioned ({NUM_PARTITIONS} parts, use {NUM_PARTITIONS - 1} each run)"
        else:
            sample_mode_desc = "Bootstrap (鏈夋斁鍥炴娊鏍? 姣斾緥=1.0)"

        content_lines = []
        content_lines.append(f"# 瀹為獙鏃堕棿: {CURRENT_DATE}")
        content_lines.append(f"# 鍩哄洜绋冲畾鎬ч獙璇佺粨鏋?- Fold {fold} - {CANCER_TYPE}")
        content_lines.append(f"# 鎶芥牱妯″紡: {sample_mode_desc}")
        content_lines.append(f"# 杩唬娆℃暟: {total_iterations}")
        if include_full_iteration:
            content_lines.append(f"# 鍏朵腑闅忔満鎶芥牱娆℃暟: {num_iterations} (鍙﹀惈鍏ㄩ泦1娆?")
        content_lines.append(f"# 鍞竴鍩哄洜鏁? {total_unique_genes}")
        content_lines.append(f"# 棰戞闃堝€? {GENE_FREQ_THRESHOLD}")
        content_lines.append(f"# TopK(闃堝€?=1鏃剁敓鏁?: {TOP_K}")
        content_lines.append(f"#")
        content_lines.append(f"# 鎺掑悕\t鍩哄洜绱㈠紩\t鍑虹幇娆℃暟\t鍑虹幇棰戠巼")
        content_lines.append(f"{'='*50}")
        for rank, (gene_idx, count) in enumerate(selected_genes, 1):
            freq = count / total_iterations * 100
            content_lines.append(f"{rank}\t{gene_idx}\t{count}\t{freq:.1f}%")

        new_content = '\n'.join(content_lines) + '\n'

        # 濡傛灉鏂囦欢宸插瓨鍦紝灏嗘柊鍐呭娣诲姞鍒版渶鍓嶉潰
        if os.path.exists(txt_output_file):
            with open(txt_output_file, 'r') as f:
                existing_content = f.read()
            with open(txt_output_file, 'w') as f:
                f.write(new_content)
                f.write(f"\n{'='*70}\n")
                f.write(existing_content)
        else:
            with open(txt_output_file, 'w') as f:
                f.write(new_content)
        print(f"  宸蹭繚瀛? {txt_output_file}")

        # 3. CSV鐗瑰緛鏂囦欢鏍煎紡锛堝寘鍚畬鏁存暟鎹泦 train + val + test锛?        # 纭繚CSV杈撳嚭鐩綍瀛樺湪
        os.makedirs(CSV_OUTPUT_DIR, exist_ok=True)

        sample_ids = load_all_sample_ids(CANCER_TYPE, fold, DATA_DIR)

        # 鍔犺浇瀹屾暣鏁版嵁锛堝寘鍚墍鏈夋牱鏈級
        full_data, gene_names_all, patient_ids_all = load_full_data(CANCER_TYPE)

        if sample_ids is not None and full_data is not None and patient_ids_all is not None:
            # 鑾峰彇top鍩哄洜绱㈠紩鍒楄〃
            top_gene_indices = [g[0] for g in selected_genes]

            data_for_csv = full_data[:, :-1]
            patient_to_idx = {pid: idx for idx, pid in enumerate(patient_ids_all)}
            missing_sample_ids = [sid for sid in sample_ids if sid not in patient_to_idx]
            if missing_sample_ids:
                preview = ', '.join(missing_sample_ids[:5])
                raise KeyError(
                    f"Fold {fold} has {len(missing_sample_ids)} sample IDs missing from full data. "
                    f"Examples: {preview}"
                )

            aligned_data = np.zeros((len(sample_ids), data_for_csv.shape[1]), dtype=data_for_csv.dtype)
            for i, sample_id in enumerate(sample_ids):
                aligned_data[i, :] = data_for_csv[patient_to_idx[sample_id], :]

            # 淇濆瓨CSV
            csv_file = os.path.join(CSV_OUTPUT_DIR, f'fold_{fold}_genes.csv')
            save_stable_genes_csv(
                aligned_data,
                top_gene_indices,
                sample_ids,
                csv_file,
                gene_names_all
            )

        print(f"\n  Fold {fold} 瀹屾垚!")

    total_time = time.time() - total_start_time
    print(f"\n{'='*70}")
    print("  All folds summary")
    print(f"{'='*70}")
    print(f"  鎬籉old鏁? {NUM_FOLDS}")
    print(f"  姣廎old杩唬娆℃暟: {total_iterations}")
    print(f"  鎬昏繍琛屾鏁? {NUM_FOLDS * total_iterations}")
    print(f"  鎬昏€楁椂: {total_time:.2f} 绉?({total_time/60:.2f} 鍒嗛挓)")

    print(f"\n{'='*70}")
    print(f"  Pipeline 瀹屾垚!")
    print(f"{'='*70}")
    print(f"\n  杈撳嚭鏂囦欢:")
    print(f"  - {OUTPUT_DIR}/stable_genes_fold{{fold}}_top100.mat")
    print(f"  - {OUTPUT_DIR}/stable_genes_fold{{fold}}_top100.txt")
    print(f"{'='*70}")


if __name__ == '__main__':
    run_stable_genes_pipeline()
