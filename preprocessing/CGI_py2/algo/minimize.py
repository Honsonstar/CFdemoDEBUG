import numpy as np

def rasmussen_minimize(X, f, length, *args):
    """
    Robust Python translation of minimize.m
    (with Zero-Division Protection)
    """
    INT = 0.1; EXT = 3.0; MAX = 20; RATIO = 10; SIG = 0.1; RHO = SIG / 2.0
    X = np.array(X, dtype=float).flatten()
    i = 0
    ls_failed = 0

    try:
        f0, df0 = f(X, *args)
        f0 = float(f0); df0 = np.array(df0).flatten()
    except Exception:
        return X # 如果初始点就崩溃，安全返回

    s = -df0
    d0 = -np.dot(s, s)
    x3 = 1.0 / (1.0 - d0) if d0 != 1.0 else 1.0

    while i < length:
        i += 1
        X0 = X.copy(); f00 = f0; df00 = df0.copy()
        M = MAX
        while True:
            x2 = 0.0; f2 = f0; d2 = d0; f3 = f0; df3 = df0.copy()
            success = False
            while not success and M > 0:
                try:
                    M -= 1
                    f3, df3 = f(X + x3 * s, *args)
                    f3 = float(f3); df3 = np.array(df3).flatten()
                    if np.isnan(f3) or np.isinf(f3) or np.any(np.isnan(df3)) or np.any(np.isinf(df3)):
                        raise ValueError("Math Error") # 主动抛错，触发回溯
                    success = True
                except Exception:
                    x3 = (x2 + x3) / 2.0 # 核心：报错时缩小步长重试！

            if f3 < f0:
                X0 = X + x3 * s; f00 = f3; df00 = df3.copy()

            d3 = np.dot(df3, s)
            if d3 > SIG * d0 or f3 > f0 + x3 * RHO * d0 or M == 0:
                break

            x1 = x2; f1 = f2; d1 = d2
            x2 = x3; f2 = f3; d2 = d3
            A = 6.0 * (f1 - f2) + 3.0 * (d2 + d1) * (x2 - x1)
            B = 3.0 * (f2 - f1) - (2.0 * d1 + d2) * (x2 - x1)

            # --- 修复核心：安全计算分母防除零 ---
            denom = B + np.sqrt(max(B * B - A * d1 * (x2 - x1), 0.0))
            if denom == 0 or np.isnan(denom):
                x3 = x2 * EXT
            else:
                x3 = x1 - d1 * (x2 - x1)**2 / denom
            # ------------------------------------

            if not np.isreal(x3) or np.isnan(x3) or np.isinf(x3) or x3 < 0:
                x3 = x2 * EXT
            elif x3 > x2 * EXT:
                x3 = x2 * EXT
            elif x3 < x2 + INT * (x2 - x1):
                x3 = x2 + INT * (x2 - x1)

        while (abs(d3) > -SIG * d0 or f3 > f0 + x3 * RHO * d0) and M > 0:
            if d3 > 0 or f3 > f0 + x3 * RHO * d0:
                x4 = x3; f4 = f3; d4 = d3
            else:
                x2 = x3; f2 = f3; d2 = d3
            if f4 > f0:
                denom2 = (f4 - f2 - d2 * (x4 - x2))
                if denom2 == 0: denom2 = 1e-16
                x3 = x2 - (0.5 * d2 * (x4 - x2)**2) / denom2
            else:
                A = 6.0 * (f2 - f4) / (x4 - x2) + 3.0 * (d4 + d2)
                B = 3.0 * (f4 - f2) - (2.0 * d2 + d4) * (x4 - x2)
                if A == 0: A = 1e-16
                x3 = x2 + (np.sqrt(max(B * B - A * d2 * (x4 - x2)**2, 0.0)) - B) / A

            if np.isnan(x3) or np.isinf(x3):
                x3 = (x2 + x4) / 2.0
            x3 = max(min(x3, x4 - INT * (x4 - x2)), x2 + INT * (x4 - x2))

            try:
                f3, df3 = f(X + x3 * s, *args)
                f3 = float(f3); df3 = np.array(df3).flatten()
                M -= 1
            except Exception:
                break

            if f3 < f0:
                X0 = X + x3 * s; f00 = f3; df00 = df3.copy()
            d3 = np.dot(df3, s)

        if abs(d3) < -SIG * d0 and f3 < f0 + x3 * RHO * d0:
            X = X + x3 * s; f0 = f3
            s = (np.dot(df3, df3) - np.dot(df0, df3)) / max(np.dot(df0, df0), 1e-16) * s - df3
            df0 = df3.copy()
            d3 = d0; d0 = np.dot(df0, s)
            if d0 > 0:
                s = -df0; d0 = -np.dot(s, s)
            x3 = x3 * min(RATIO, d3 / (d0 - 1e-16))
            ls_failed = 0
        else:
            X = X0; f0 = f00; df0 = df00
            if ls_failed or i > length:
                break
            s = -df0; d0 = -np.dot(s, s)
            x3 = 1.0 / (1.0 - d0) if d0 != 1.0 else 1.0
            ls_failed = 1

    return X

# ========== 在文件最底部加上这行，桥接 __init__.py 的导入 ==========
minimize = rasmussen_minimize
