"""Day3：几何拟合与尺寸测量

从点云里"量出尺寸"是工业 3D 检测的核心动作。三步：
    1. 找到面（平面拟合）
    2. 找到轴（圆柱拟合）
    3. 量尺寸（面间距、直径、高度）

三个实验：
    实验1  平面拟合 —— 最小二乘 vs RANSAC（加离群点看谁扛得住）
    实验2  圆柱拟合 —— 轴沿 Z 时退化成 2D 圆拟合（复用最小二乘）
    实验3  尺寸测量 —— 从点云量立方体边长、圆柱直径与高度

全部和合成真值对比，能报误差。

跑法：
    python day3_fitting.py
    python day3_fitting.py --no-vis
    python day3_fitting.py --exp 1
"""
import argparse
import os
import time

import numpy as np
import open3d as o3d

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")


# ------------------------------------------------------------------ 平面拟合

def fit_plane_lsq(pcd):
    """普通最小二乘平面拟合（PCA 法）

    取点云协方差矩阵最小特征值对应的特征向量作为法向。
    等价于最小化所有点到平面的距离平方和 —— 和标定时那个最小二乘同一个东西。
    缺点：离群点会被平方放大，一个飞点能带偏整个平面。
    """
    pts = np.asarray(pcd.points)
    c = pts.mean(axis=0)
    cov = np.cov((pts - c).T)
    w, v = np.linalg.eigh(cov)
    normal = v[:, 0]              # 最小特征值 -> 法向
    normal /= np.linalg.norm(normal)
    d = -float(normal @ c)        # 平面方程 n·x + d = 0
    return normal, d


def fit_plane_ransac(pcd, thr=1.0, iters=1000):
    """RANSAC 平面拟合（Open3D 内置）"""
    t0 = time.perf_counter()
    model, inliers = pcd.segment_plane(
        distance_threshold=thr, ransac_n=3, num_iterations=iters)
    dt = (time.perf_counter() - t0) * 1000
    normal = np.array(model[:3], float)
    normal /= np.linalg.norm(normal)
    d = float(model[3]) / np.linalg.norm(np.array(model[:3], float))
    return normal, d, np.asarray(inliers), dt


def plane_angle_err(n1, n2):
    """两个法向的夹角（度），方向无关"""
    c = abs(float(n1 @ n2))
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def plane_point_dist(pcd, normal, d):
    """点到平面的距离统计"""
    pts = np.asarray(pcd.points)
    dist = np.abs(pts @ normal + d)
    return float(dist.mean()), float(dist.max())


# ------------------------------------------------------------------ 圆拟合

def fit_circle_2d(x, y):
    """2D 圆拟合（线性化，复用标定那一节的推导）

        2a·x + 2b·y + c = x² + y²,   c = r² - a² - b²
    解完反推 r = sqrt(c + a² + b²)
    """
    A = np.column_stack([2 * x, 2 * y, np.ones(len(x))])
    b = x ** 2 + y ** 2
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    a, bb, c = sol
    r = np.sqrt(max(c + a ** 2 + bb ** 2, 0.0))
    return float(a), float(bb), float(r)


def fit_cylinder_z(pcd):
    """轴沿 Z 的圆柱拟合：投影到 XY 做 2D 圆拟合 + 统计 Z 范围

    真实圆柱轴向任意，需要先估轴（PCA / 法向聚类），复杂得多。
    这里先做最简单的轴对齐情形，够用来验证尺寸测量链路。
    """
    pts = np.asarray(pcd.points)
    cx, cy, r = fit_circle_2d(pts[:, 0], pts[:, 1])
    zmin, zmax = float(pts[:, 2].min()), float(pts[:, 2].max())
    return cx, cy, r, zmin, zmax


# ------------------------------------------------------------------ 造测试数据

def plane_with_outliers(pcd_plane, n_out=0, seed=0):
    """在平面里塞飞点"""
    pts = np.asarray(pcd_plane.points).copy()
    if n_out > 0:
        rng = np.random.default_rng(seed)
        xy = np.column_stack([
            rng.uniform(pts[:, 0].min(), pts[:, 0].max(), n_out),
            rng.uniform(pts[:, 1].min(), pts[:, 1].max(), n_out),
        ])
        z = rng.uniform(8, 30, n_out)      # 故意偏离平面
        pts = np.vstack([pts, np.column_stack([xy, z])])
    out = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(pts)
    return out


# ------------------------------------------------------------------ 实验

def exp1_plane(no_vis=False):
    print("\n" + "=" * 78)
    print("实验1：平面拟合 —— 最小二乘 vs RANSAC")
    print("=" * 78)
    print("  真值：Z=0 平面（法向 (0,0,1)，过原点）")
    print("  噪声：σ=0.5mm；离群点：随机塞 8~30mm 高的飞点\n")

    base = o3d.io.read_point_cloud(os.path.join(DATA, "plane.ply"))
    print(f"  {'离群点数':>8} | {'最小二乘':>32} | {'RANSAC':>28}")
    print(f"  {'':>8} | {'法向误差':>12} {'平均距离':>10} {'最大':>8} | "
          f"{'法向误差':>8} {'平均距离':>8} {'耗时':>8}")
    print("  " + "-" * 84)

    rows = []
    for n_out in (0, 50, 200, 800):
        pcd = plane_with_outliers(base, n_out, seed=n_out)

        n1, d1 = fit_plane_lsq(pcd)
        a1 = plane_angle_err(n1, np.array([0, 0, 1.0]))
        m1, x1 = plane_point_dist(base, n1, d1)     # 只在真实平面上评估

        n2, d2, inl, dt = fit_plane_ransac(pcd, thr=1.0)
        a2 = plane_angle_err(n2, np.array([0, 0, 1.0]))
        m2, x2 = plane_point_dist(base, n2, d2)

        rows.append((n_out, a1, m1, x1, a2, m2, dt))
        print(f"  {n_out:>8d} | {a1:>10.3f}° {m1:>10.3f} {x1:>8.3f} | "
              f"{a2:>7.3f}° {m2:>8.3f} {dt:>7.0f}ms")

    print("\n  看点：")
    print("    - 无离群点时两者都准（最小二乘甚至更快）")
    print("    - 离群点一多，最小二乘被拉偏，RANSAC 基本不动")
    print("    - 这就是为什么工业检测里平面分割默认用 RANSAC")
    return rows


def exp2_cylinder(no_vis=False):
    print("\n" + "=" * 78)
    print("实验2：圆柱拟合 —— 轴沿 Z 时退化成 2D 圆拟合")
    print("=" * 78)
    print("  真值：半径 40mm，高 120mm，轴沿 Z\n")

    pcd = o3d.io.read_point_cloud(os.path.join(DATA, "cylinder.ply"))
    pts = np.asarray(pcd.points)
    print(f"  点云 {len(pts)} 点，Z 范围 [{pts[:,2].min():.1f}, {pts[:,2].max():.1f}]")

    t0 = time.perf_counter()
    cx, cy, r, zmin, zmax = fit_cylinder_z(pcd)
    dt = (time.perf_counter() - t0) * 1000

    print(f"\n  拟合结果：")
    print(f"    圆心 ({cx:7.3f}, {cy:7.3f})   半径 {r:7.3f} mm")
    print(f"    高度 {zmax - zmin:7.3f} mm   ({zmin:.2f} ~ {zmax:.2f})")
    print(f"    耗时 {dt:.1f} ms")
    print(f"\n  与真值对比：")
    print(f"    圆心误差 {np.hypot(cx, cy):7.4f} mm   (真值 0,0)")
    print(f"    半径误差 {abs(r - 40):7.4f} mm      (真值 40)")
    print(f"    高度误差 {abs((zmax - zmin) - 120):7.4f} mm   (真值 120)")

    print("\n  看点：")
    print("    - 轴对齐时圆柱拟合就是 2D 圆拟合，直接复用标定那节的线性化公式")
    print("    - 但端点靠 Z 的最大最小值，噪声会让高度偏大（极值对噪声敏感）")
    return (cx, cy, r, zmin, zmax)


def fit_box_faces(pcd, thr=1.5, max_planes=8, min_inliers=500):
    """RANSAC 逐个分割平面，返回到 [(normal, d, n_inliers), ...]

    做法：找一个平面 -> 把它的内点移走 -> 再找下一个。
    最后按法向分组，同一方向的一对平面间距就是一条边长。
    """
    work = o3d.geometry.PointCloud(pcd)
    planes = []
    for _ in range(max_planes):
        if len(work.points) < min_inliers:
            break
        model, inl = work.segment_plane(
            distance_threshold=thr, ransac_n=3, num_iterations=1000)
        if len(inl) < min_inliers:
            break
        n = np.array(model[:3], float)
        nrm = np.linalg.norm(n)
        n = n / nrm
        d = float(model[3]) / nrm
        planes.append((n, d, len(inl)))
        work = work.select_by_index(list(inl), invert=True)
    return planes


def box_edges_from_planes(planes, ang_tol=12.0):
    """把平面按法向分组，取每组两个平行面的间距作为边长"""
    if not planes:
        return []
    used = [False] * len(planes)
    edges = []
    for i in range(len(planes)):
        if used[i]:
            continue
        group = [i]
        ni = planes[i][0]
        for j in range(i + 1, len(planes)):
            if used[j]:
                continue
            nj = planes[j][0]
            c = abs(float(ni @ nj))
            if np.degrees(np.arccos(np.clip(c, -1, 1))) < ang_tol:
                group.append(j)
        for g in group:
            used[g] = True
        if len(group) >= 2:
            # 同组内法向统一符号后，取 d 的最大最小差
            n_ref = planes[group[0]][0]
            ds = []
            for g in group:
                n_g, d_g, _ = planes[g]
                ds.append(d_g if float(n_ref @ n_g) > 0 else -d_g)
            edges.append(max(ds) - min(ds))
    return edges


def exp3_size(no_vis=False):
    print("\n" + "=" * 78)
    print("实验3：尺寸测量 —— 从点云量尺寸")
    print("=" * 78)

    # --- 立方体：包围盒法 vs 平面拟合法 ---
    cube = o3d.io.read_point_cloud(os.path.join(DATA, "cube_noisy.ply"))
    pts = np.asarray(cube.points)
    print("  【立方体】真值边长 100 mm")
    print(f"    点数 {len(pts)}，噪声 σ=1.0mm")

    bbox = cube.get_axis_aligned_bounding_box().get_extent()
    print(f"\n    方法A 包围盒：   {bbox[0]:.2f} x {bbox[1]:.2f} x {bbox[2]:.2f} mm")
    print(f"      误差：{bbox[0]-100:+.2f} / {bbox[1]-100:+.2f} / {bbox[2]-100:+.2f} mm")
    print(f"      -> 包围盒取的是极值，噪声让它系统性偏大（+{bbox[0]-100:.2f}mm）")

    # 用分位数代替极值，抗噪声
    lo = np.percentile(pts, 1, axis=0)
    hi = np.percentile(pts, 99, axis=0)
    ext = hi - lo
    print(f"\n    方法B 1%~99% 分位：{ext[0]:.2f} x {ext[1]:.2f} x {ext[2]:.2f} mm")
    print(f"      误差：{ext[0]-100:+.2f} / {ext[1]-100:+.2f} / {ext[2]:.2f}-100 mm")
    print(f"      -> 分位数排除极端值，比包围盒稳，但引入约 2% 的收缩")

    # --- 圆柱：直径与高度 ---
    cyl = o3d.io.read_point_cloud(os.path.join(DATA, "cylinder.ply"))
    cx, cy, r, zmin, zmax = fit_cylinder_z(cyl)
    print(f"\n  【圆柱】真值 直径 80mm，高 120mm")
    print(f"    拟合直径 {2*r:.2f} mm   误差 {2*r-80:+.2f} mm ({abs(2*r-80)/80*100:.2f}%)")
    print(f"    拟合高度 {zmax-zmin:.2f} mm   误差 {(zmax-zmin)-120:+.2f} mm")

    # --- 方法C：拟合六个面，量面间距（工业标准做法）---
    planes = fit_box_faces(cube, thr=1.5)
    edges = sorted(box_edges_from_planes(planes))
    print()
    print(f"    方法C 拟合 {len(planes)} 个平面 -> {len(edges)} 条边长：")
    for k, e in enumerate(edges):
        print(f"      {k+1}. {e:7.2f} mm   误差 {e-100:+.2f} mm")
    if edges:
        err_c = max(abs(e - 100) for e in edges)
        err_bbox = max(abs(float(b) - 100) for b in bbox)
        print(f"      最大误差 {err_c:.2f} mm  （对比包围盒 {err_bbox:.2f} mm）")

    print()
    print("  看点：")
    print("    - 尺寸测量第一条铁律：极值（包围盒）对噪声最敏感")
    print("    - 拟合（圆/平面）比取极值稳得多，因为用了全部点而不是一两个")
    print("    - 圆柱高度仍是取 Z 极值，所以误差比直径大，这是可改进点")
    print("    - 工业标准做法是方法C：拟合每个面，用面间距当尺寸")
    return {"cube_bbox": bbox, "cube_pct": ext, "cyl_r": r, "cyl_h": zmax - zmin}


# ------------------------------------------------------------------ 入口

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-vis", action="store_true")
    ap.add_argument("--exp", type=int, default=0)
    args = ap.parse_args()

    if not os.path.exists(os.path.join(DATA, "plane.ply")):
        print("缺数据，先跑 python synth_data.py")
        return

    if args.exp in (0, 1):
        exp1_plane(args.no_vis)
    if args.exp in (0, 2):
        exp2_cylinder(args.no_vis)
    if args.exp in (0, 3):
        exp3_size(args.no_vis)

    print("\n" + "=" * 78)
    print("Day3 要能回答的四个问题")
    print("=" * 78)
    print("  1. 为什么拟合比包围盒准？")
    print("     -> 拟合用了全部点的统计信息，包围盒只取了一两个极值点")
    print("  2. RANSAC 平面分割的两个关键参数是什么？")
    print("     -> distance_threshold（内点距离）和 ransac_n（最小样本数 3）")
    print("  3. 圆柱拟合什么时候变复杂？")
    print("     -> 轴不沿坐标轴时，要先估轴向（PCA/法向聚类），不能直接 2D 投影")
    print("  4. 尺寸误差怎么分解？")
    print("     -> 噪声引起的随机误差 + 极值/拟合方法引起的系统偏差")


if __name__ == "__main__":
    main()
