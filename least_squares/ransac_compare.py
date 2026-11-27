"""练习3 完整版：离群点数量 vs 拟合误差

固定"只观测 1/4 圆弧"这个最难的工况，逐步增加离群点，
对比 普通代数最小二乘 与 RANSAC 的圆心/半径误差。

运行：
    python ransac_compare.py
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ---------------- 基础算法 ----------------

def fit_circle_ls(x, y):
    """代数最小二乘（线性化）拟合圆，返回 (cx, cy, r)"""
    A = np.column_stack([2 * x, 2 * y, np.ones(len(x))])
    b = x ** 2 + y ** 2
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    a, bb, c = sol
    r = np.sqrt(max(c + a ** 2 + bb ** 2, 0.0))
    return a, bb, r


def circle_from_3pts(p):
    """三点定圆（外接圆）。三点共线时返回 None"""
    (x1, y1), (x2, y2), (x3, y3) = p
    d = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    if abs(d) < 1e-10:
        return None
    ux = ((x1**2 + y1**2) * (y2 - y3) +
          (x2**2 + y2**2) * (y3 - y1) +
          (x3**2 + y3**2) * (y1 - y2)) / d
    uy = ((x1**2 + y1**2) * (x3 - x2) +
          (x2**2 + y2**2) * (x1 - x3) +
          (x3**2 + y3**2) * (x2 - x1)) / d
    return ux, uy, np.hypot(x1 - ux, y1 - uy)


def ransac_circle(x, y, thr=1.0, iters=800, seed=1):
    """RANSAC 圆拟合，返回 (cx, cy, r, 内点数, 内点布尔数组)"""
    pts = np.column_stack([x, y])
    rng = np.random.default_rng(seed)
    best_inl, best_cnt = None, -1
    for _ in range(iters):
        idx = rng.choice(len(pts), 3, replace=False)
        m = circle_from_3pts(pts[idx])
        if m is None:
            continue
        cx, cy, r = m
        if not (0.1 < r < 1e4):
            continue
        d = np.abs(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) - r)
        inl = d < thr
        if inl.sum() > best_cnt:
            best_cnt, best_inl = int(inl.sum()), inl
    if best_inl is None:
        return None
    cx, cy, r = fit_circle_ls(x[best_inl], y[best_inl])
    return cx, cy, r, best_cnt, best_inl


# ---------------- 生成一组"干净"数据 ----------------

def make_data(arc=True, seed=0):
    """arc=True 只用 1/4 圆弧；False 用完整圆"""
    np.random.seed(seed)
    theta = (np.linspace(0, np.pi / 2, 10) if arc
             else np.linspace(0, 2 * np.pi, 10, endpoint=False))
    x = 2 + 5 * np.cos(theta) + np.random.randn(10) * 0.5
    y = 3 + 5 * np.sin(theta) + np.random.randn(10) * 0.5
    return x, y


def add_outliers(x, y, n_out, seed=42):
    """随机挑 n_out 个点，推到远处（沿随机方向推 25~35）"""
    rng = np.random.default_rng(seed)
    x2, y2 = x.copy(), y.copy()
    if n_out > 0:
        idx = rng.choice(len(x), n_out, replace=False)
        ang = rng.uniform(0, 2 * np.pi, n_out)
        dist = rng.uniform(25, 35, n_out)
        x2[idx] += dist * np.cos(ang)
        y2[idx] += dist * np.sin(ang)
    return x2, y2


# ---------------- 主实验 ----------------

def run(arc=True, tag=""):
    print("=" * 78)
    print(f"工况：{'1/4 弧' if arc else '完整圆'} {tag}")
    print("=" * 78)
    print(f"{'离群点':>6} | {'lstsq 圆心误差':>15} | {'RANSAC 圆心误差':>16} | "
          f"{'lstsq r误差':>12} | {'RANSAC r误差':>13} | {'RANSAC剔除':>10}")
    print("-" * 78)

    base_x, base_y = make_data(arc=arc)
    rows = []
    for n_out in [0, 1, 2, 3, 4]:
        ls_errs, ra_errs = [], []
        ls_rerrs, ra_rerrs = [], []
        removed = 0
        # RANSAC 有随机性，跑 10 次取中位数，避免单次偶然
        for trial in range(10):
            x, y = add_outliers(base_x, base_y, n_out, seed=100 + trial)
            a1, b1, r1 = fit_circle_ls(x, y)
            res = ransac_circle(x, y, thr=1.0, seed=trial)
            if res is None:
                continue
            a2, b2, r2, cnt, inl = res
            ls_errs.append(np.hypot(a1 - 2, b1 - 3))
            ra_errs.append(np.hypot(a2 - 2, b2 - 3))
            ls_rerrs.append(abs(r1 - 5))
            ra_rerrs.append(abs(r2 - 5))
            if trial == 0:
                removed = int((~inl).sum())

        med = np.median
        print(f"{n_out:>6} | {med(ls_errs):>15.3f} | {med(ra_errs):>16.3f} | "
              f"{med(ls_rerrs):>12.3f} | {med(ra_rerrs):>13.3f} | "
              f"{removed:>10}")
        rows.append((n_out, med(ls_errs), med(ra_errs),
                     med(ls_rerrs), med(ra_rerrs)))
    print("（表中数字为 10 次随机实验的中位数）\n")
    return rows


if __name__ == "__main__":
    arc_rows = run(arc=True, tag="（工业常见：只能看到部分弧）")
    full_rows = run(arc=False, tag="（理想：能看到完整圆）")

    # 画误差曲线
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, rows, title in (
        (axes[0], arc_rows, "1/4 弧（视角不足）"),
        (axes[1], full_rows, "完整圆（视角充足）"),
    ):
        n = [r[0] for r in rows]
        ax.plot(n, [r[1] for r in rows], 'r-o', lw=2, label='普通最小二乘')
        ax.plot(n, [r[2] for r in rows], 'b-s', lw=2, label='RANSAC')
        ax.set_yscale('log')
        ax.set_xlabel('离群点个数')
        ax.set_ylabel('圆心误差（对数轴）')
        ax.set_title(title)
        ax.grid(alpha=.3, which='both')
        ax.legend()
    fig.suptitle('离群点数量 vs 圆心误差', fontsize=14)
    plt.tight_layout()
    plt.show()
