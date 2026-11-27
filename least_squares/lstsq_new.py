import numpy as np
import cv2
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def fit_circle_ls(x, y):
    """代数最小二乘（线性化）"""
    A = np.column_stack([2*x, 2*y, np.ones(len(x))])
    b = x**2 + y**2
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    a, bb, c = sol
    r = np.sqrt(max(c + a**2 + bb**2, 0.0))
    return a, bb, r

def circle_from_n_pts(pts_subset):
    """用任意个点（>=3）拟合圆"""
    return fit_circle_ls(pts_subset[:, 0], pts_subset[:, 1])


def circle_from_3pts(p):
    """三点定圆（外接圆），共线返回 None"""
    (x1, y1), (x2, y2), (x3, y3) = p
    d = 2*(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))
    if abs(d) < 1e-10:
        return None
    ux = ((x1**2+y1**2)*(y2-y3) + (x2**2+y2**2)*(y3-y1) + (x3**2+y3**2)*(y1-y2)) / d
    uy = ((x1**2+y1**2)*(x3-x2) + (x2**2+y2**2)*(x1-x3) + (x3**2+y3**2)*(x2-x1)) / d
    return ux, uy, np.hypot(x1-ux, y1-uy)


def ransac_circle(x, y, thr=1.0, iters=500, seed=1):
    """RANSAC 圆拟合：随机 3 点定圆 → 数内点 → 取最好 → 内点重拟合"""
    pts = np.column_stack([x, y])
    rng = np.random.default_rng(seed)
    best_inl, best_cnt = None, -1
    for _ in range(iters):
        idx = rng.choice(len(pts), 5, replace=False)
        m = circle_from_n_pts(pts[idx])
        if m is None:                       # 三点共线，退化
            continue
        cx, cy, r = m
        if not (0.1 < r < 1e4):             # 半径荒谬，丢弃
            continue
        d = np.abs(np.hypot(pts[:, 0]-cx, pts[:, 1]-cy) - r)
        inl = d < thr
        if inl.sum() > best_cnt:
            best_cnt, best_inl = inl.sum(), inl
    if best_inl is None:
        return None
    a, b, r = fit_circle_ls(x[best_inl], y[best_inl])   # 用内点重拟合
    return a, b, r, int(best_cnt), best_inl


# ---- 数据 ----
np.random.seed(0)
theta = np.linspace(0, np.pi/2, 10)
x = 2 + 5*np.cos(theta) + np.random.randn(10)*0.5
y = 3 + 5*np.sin(theta) + np.random.randn(10)*0.5
y[3] += 30

a1, b1, r1 = fit_circle_ls(x, y)
pts32 = np.column_stack([x, y]).astype(np.float32)
(c2, c3), r2 = cv2.minEnclosingCircle(pts32)
a3, b3, r3, cnt, inl = ransac_circle(x, y, thr=1.0)

print(f"真值         : (2.000, 3.000)  r=5.000")
print(f"lstsq        : ({a1:6.2f}, {b1:6.2f})  r={r1:6.2f}   圆心误差 {np.hypot(a1-2, b1-3):6.2f}")
print(f"minEnclosing : ({c2:6.2f}, {c3:6.2f})  r={r2:6.2f}   圆心误差 {np.hypot(c2-2, c3-3):6.2f}")
print(f"RANSAC       : ({a3:6.2f}, {b3:6.2f})  r={r3:6.2f}   圆心误差 {np.hypot(a3-2, b3-3):6.2f}")
print(f"               内点 {cnt}/{len(x)}，剔除 {[i for i, v in enumerate(inl) if not v]}")

# ---- 画图 ----
t = np.linspace(0, 2*np.pi, 400)
fig, ax = plt.subplots(figsize=(8, 7))
ax.scatter(x, y, c='k', s=50, zorder=5, label='观测点')
ax.scatter(x[~inl], y[~inl], c='m', s=120, marker='x',
           zorder=6, label='RANSAC 判定的离群点')
ax.plot(2+5*np.cos(t), 3+5*np.sin(t), 'g--', lw=1.2, label='真值圆')
ax.plot(a1+r1*np.cos(t), b1+r1*np.sin(t), 'r-', lw=1.2, label='lstsq')
ax.plot(a3+r3*np.cos(t), b3+r3*np.sin(t), 'b-', lw=1.8, label='RANSAC')
ax.set_aspect('equal'); ax.legend(); ax.grid(alpha=.3)
ax.set_title('RANSAC 剔除离群点（1/4 弧 + 1 个离群点）')
plt.tight_layout(); plt.show()