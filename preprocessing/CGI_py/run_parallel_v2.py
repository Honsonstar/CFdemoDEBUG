"""
================================================================================
run_parallel_v2.py - CGI 完整算法内存优化并行版本
================================================================================

【文件作用】
完整实现 CGI 因果基因发现算法的并行版本，包含 0阶、1阶、2阶条件独立性测试
以及回归分析确定因果方向。相比 v1 版本，使用共享内存优化内存使用，
适合处理大规模数据集。

【使用方法】
```python
import numpy as np
from run_parallel_v2 import find_genes_gci_parallel_v2
from scipy.io import loadmat

# 加载数据
data = loadmat('gene_expression.mat')['d']

# 运行完整算法（内存优化版）
result = find_genes_gci_parallel_v2(
    data,
    alpha=0.05,
    n_jobs=8,      # 并行进程数
    verbose=True    # 打印进度
)

# 获取结果
causal_genes = result['found_genes']
print(f"发现 {len(causal_genes)} 个因果基因")
```

【算法流程】
1. 0阶 CI 测试：检验每个基因与目标变量的无条件独立性
2. 1阶 CI 测试：控制一个基因后检验
3. 2阶 CI 测试：控制两个基因后检验
4. 回归分析：确定因果方向，找出真正的因果基因

【参数说明】
- data: 基因表达数据，shape=(n_samples, n_genes+1)，最后一列是目标变量
- alpha: 显著性水平，默认 0.05
- n_jobs: 并行进程数
- verbose: 是否打印进度信息

【内存优化】
- 使用共享内存 (shared_memory) 在进程间共享数据
- 减少内存复制，提高大规模数据处理效率

【依赖】
- numpy
- scipy
- multiprocessing

================================================================================
"""

import sys
sys.path.insert(0, '/root/autodl-tmp')

import numpy as np
from scipy.io import loadmat
from multiprocessing import Pool, shared_memory
from itertools import combinations
import os
import warnings
warnings.filterwarnings('ignore')

# 全局变量
X_GLOBAL = None
DATA_GLOBAL = None

def init_worker(x_shm_name, data_shm_name, x_shape, data_shape):
    """初始化 worker 进程 - 使用共享内存"""
    global X_GLOBAL, DATA_GLOBAL
    existing_x_shm = shared_memory.SharedMemory(name=x_shm_name)
    existing_data_shm = shared_memory.SharedMemory(name=data_shm_name)
    X_GLOBAL = np.ndarray(x_shape, dtype=np.float64, buffer=existing_x_shm.buf)
    DATA_GLOBAL = np.ndarray(data_shape, dtype=np.float64, buffer=existing_data_shm.buf)


def run_0order_test(i, alpha):
    """0-order CI 测试"""
    from CGI_py.algo import paco_test, kcit
    ind1 = paco_test(X_GLOBAL, DATA_GLOBAL[:, i], np.array([]), alpha)
    if ind1:
        ind2, _, _ = kcit(X_GLOBAL, DATA_GLOBAL[:, i], np.array([[]]), alpha=alpha)
        if ind2:
            return i
    return None


def run_1order_test(args):
    """1-order CI 测试"""
    from CGI_py.algo import paco_test, kcit, fit_gpr
    j, k, alpha, cov, hyp, Ncg = args
    try:
        y = DATA_GLOBAL[:, j]
        z = DATA_GLOBAL[:, k]
        ind1 = paco_test(X_GLOBAL, y, z, alpha)
        if ind1:
            try:
                xf = fit_gpr(z, X_GLOBAL, cov, hyp, Ncg)
                res1 = xf - X_GLOBAL
                yf = fit_gpr(z, y, cov, hyp, Ncg)
                res2 = yf - y
                ind2, _, _ = kcit(res1, res2, np.array([[]]), alpha=alpha)
                if ind2:
                    return (j, res1.copy())
            except:
                pass
    except:
        pass
    return None


def run_2order_test(args):
    """2-order CI 测试"""
    from CGI_py.algo import paco_test, kcit, fit_gpr
    j, k, alpha, cov, hyp, Ncg = args
    try:
        y = DATA_GLOBAL[:, j]
        z1 = DATA_GLOBAL[:, k[0]]
        z2 = DATA_GLOBAL[:, k[1]]
        z = np.column_stack([z1, z2])

        ind = paco_test(X_GLOBAL, y, z, alpha)
        if ind:
            try:
                xf = fit_gpr(z, X_GLOBAL, cov, hyp, Ncg)
                res1 = xf - X_GLOBAL
                ind2, _, _ = kcit(res1, z1.reshape(-1, 1), np.array([[]]), alpha=alpha)
                ind3, _, _ = kcit(res1, z2.reshape(-1, 1), np.array([[]]), alpha=alpha)
                if ind2 or ind3:
                    return j
            except:
                pass
    except:
        pass
    return None


def find_genes_gci_full(data: np.ndarray, alpha: float = 0.05,
                        cov: str = 'covSEiso', Ncg: int = 100,
                        hyp: np.ndarray = None, n_jobs: int = 8) -> dict:
    """完整的 CGI 算法 - 内存优化并行版本"""
    import uuid

    n = data.shape[1] - 1
    x = data[:, -1]
    data = data[:, :n]

    # 标准化
    for i in range(data.shape[1]):
        col = data[:, i]
        std = np.std(col)
        if std > 0:
            data[:, i] = (col - np.mean(col)) / std

    if hyp is None:
        # 必须使用 log(4.0) 以匹配 MATLAB 的行为
        hyp = np.array([np.log(4.0), np.log(4.0), np.log(np.sqrt(0.01))])

    # 使用唯一名称避免冲突
    unique_id = str(uuid.uuid4())[:8]
    x_shm_name = f'x_shm_{unique_id}'
    data_shm_name = f'data_shm_{unique_id}'

    # 创建共享内存
    x_shm = shared_memory.SharedMemory(create=True, size=x.nbytes, name=x_shm_name)
    data_shm = shared_memory.SharedMemory(create=True, size=data.nbytes, name=data_shm_name)
    x_shared = np.ndarray(x.shape, dtype=np.float64, buffer=x_shm.buf)
    data_shared = np.ndarray(data.shape, dtype=np.float64, buffer=data_shm.buf)
    x_shared[:] = x[:]
    data_shared[:] = data[:]

    non = []
    temp_res = []  # 保存残差
    temp_z = []    # 保存条件变量

    # ========== 0-order CI tests ==========
    print('--------------- 0-order CI tests (parallel)')
    with Pool(n_jobs, initializer=init_worker,
              initargs=(x_shm_name, data_shm_name, x.shape, data.shape)) as pool:
        tasks = [(i, alpha) for i in range(n)]
        results = pool.starmap(run_0order_test, tasks)

    for res in results:
        if res is not None:
            non.append(res)
    print(f'  Found {len(non)} non-causal genes')

    # ========== 1-order CI tests ==========
    print('--------------- 1-order CI tests (parallel)')
    idx1 = [i for i in range(n) if i not in non]
    len1 = len(idx1)
    print(f'  Testing {len1} genes...')

    tasks = []
    for j_idx in range(len1):
        for k_idx in range(len1):
            if j_idx != k_idx and idx1[k_idx] not in non:
                j = idx1[j_idx]
                k = idx1[k_idx]
                tasks.append((j, k, alpha, cov, hyp, Ncg))

    print(f'  Total tasks: {len(tasks)}')

    with Pool(n_jobs, initializer=init_worker,
              initargs=(x_shm_name, data_shm_name, x.shape, data.shape)) as pool:
        results = pool.map(run_1order_test, tasks)

    added_1order = 0
    for res in results:
        if res is not None:
            j, res1 = res
            if j not in non:
                non.append(j)
                temp_res.append(res1)
                temp_z.append([j, 0])
                added_1order += 1

    print(f'  Added {added_1order} to non-causal list')

    # ========== 2-order CI tests ==========
    print('--------------- 2-order CI tests (parallel)')
    idx2 = [i for i in range(n) if i not in non]
    len2 = len(idx2)
    print(f'  Testing {len2} genes with 2-order conditions...')

    tasks = []
    for j_idx in range(len2):
        for pair in combinations(range(len2), 2):
            if pair[0] != j_idx and pair[1] != j_idx:
                j = idx2[j_idx]
                k = (idx2[pair[0]], idx2[pair[1]])
                tasks.append((j, k, alpha, cov, hyp, Ncg))

    print(f'  Total tasks: {len(tasks)}')

    with Pool(n_jobs, initializer=init_worker,
              initargs=(x_shm_name, data_shm_name, x.shape, data.shape)) as pool:
        results = pool.map(run_2order_test, tasks)

    added_2order = 0
    for res in results:
        if res is not None and res not in non:
            non.append(res)
            added_2order += 1

    print(f'  Added {added_2order} to non-causal list')

    # ========== Find causal genes ==========
    print('--------------- find genes by regression')
    pa = [i for i in range(n) if i not in non]
    print(f'  Potential causal genes: {len(pa)}')

    # 1st order regression
    found_genes_1st = []
    if temp_res and temp_z:
        temp_res = np.array(temp_res)
        temp_z = np.array(temp_z)
        from CGI_py.algo import kcit
        for k in range(len(temp_z)):
            z_idx = int(temp_z[k, 0])
            if z_idx in pa and k < temp_res.shape[1]:
                ind = kcit(DATA_GLOBAL[:, z_idx], temp_res[:, k],
                          np.array([[]]), alpha=alpha)[0]
                if ind:
                    found_genes_1st.append(z_idx)

    print(f'  Found genes by 1st order: {len(found_genes_1st)}')

    # 2nd order regression
    found_genes_2nd = []
    if len(pa) >= 2:
        pairs = list(combinations(pa, 2))
        for p_idx, (g1, g2) in enumerate(pairs):
            if p_idx % 500 == 0:
                print(f'    Processing {p_idx}/{len(pairs)}')
            try:
                from CGI_py.algo import fit_gpr, kcit
                z = DATA_GLOBAL[:, [g1, g2]]
                xf = fit_gpr(z, X_GLOBAL, cov, hyp, Ncg)
                res1 = xf - X_GLOBAL

                ind3, _, _ = kcit(res1, z[:, 0:1], np.array([[]]), alpha=alpha)
                ind4, _, _ = kcit(res1, z[:, 1:2], np.array([[]]), alpha=alpha)

                if ind3 and g1 not in found_genes_2nd:
                    found_genes_2nd.append(g1)
                if ind4 and g2 not in found_genes_2nd:
                    found_genes_2nd.append(g2)
            except:
                pass

    print(f'  Found genes by 2nd order: {len(found_genes_2nd)}')

    # Combine results
    found_genes = list(set(found_genes_1st + found_genes_2nd))

    # 清理共享内存
    x_shm.close()
    x_shm.unlink()
    data_shm.close()
    data_shm.unlink()

    return {
        'non': non,
        'pa': pa,
        'found_genes': found_genes,
        'found_genes_1st': found_genes_1st,
        'found_genes_2nd': found_genes_2nd
    }


def run_dataset(data_name: str, mat_file: str, n_jobs: int = 8):
    """运行数据集"""
    print(f"\n{'='*50}")
    print(f"Running on {data_name}")
    print(f"{'='*50}")

    data = loadmat(mat_file)['d']
    print(f"Data shape: {data.shape}")
    print(f"Using {n_jobs} workers")

    result = find_genes_gci_full(data, alpha=0.05, n_jobs=n_jobs)

    save_path = f'/root/autodl-tmp/CGI_py/result/{data_name.lower()}_result.npy'
    np.save(save_path, result)

    print(f"\nResults for {data_name}:")
    print(f"  Non-causal genes: {len(result['non'])}")
    print(f"  Potential causal genes: {len(result['pa'])}")
    print(f"  Found causal genes: {len(result['found_genes'])}")
    print(f"  Causal gene indices: {result['found_genes']}")
    print(f"  Saved to: {save_path}")

    return result


if __name__ == '__main__':
    n_jobs = min(32, os.cpu_count() or 4)
    print(f"Using {n_jobs} workers")

    run_dataset('Leukemia', '/root/autodl-tmp/CGI/normalized_Leukemia.mat', n_jobs)
    run_dataset('Prostate', '/root/autodl-tmp/CGI/normalized_Prostate.mat', n_jobs)

    print("\nAll done!")
