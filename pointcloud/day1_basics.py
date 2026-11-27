"""Day1：点云基础操作 —— 每一步都要能说清"为什么需要"

跑法：
    python day1_basics.py                # 全流程 + 可视化
    python day1_basics.py --no-vis       # 不弹窗，只打印数字（服务器/远程用）

内容：
    1. 读取与基本信息
    2. 体素下采样        解决"点太多算不动"
    3. 统计离群点滤波    解决"噪点在捣乱"
    4. 法向估计          解决"不知道表面朝向"
    5. 可视化

每个步骤都会打印：点数变化、耗时、以及量化指标。
面试时你要能报的是这些数，不是"我会用 Open3D"。
"""
import argparse
import os
import time

import numpy as np
import open3d as o3d

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")


def load(path):
    pcd = o3d.io.read_point_cloud(path)
    if len(pcd.points) == 0:
        raise SystemExit(f"读不到点云或为空: {path}")
    return pcd


def report(pcd, name, t0=None):
    pts = np.asarray(pcd.points)
    ext = pcd.get_axis_aligned_bounding_box().get_extent()
    line = (f"{name:<22} 点数 {len(pts):>7d}   "
            f"包围盒 {ext[0]:7.2f} x {ext[1]:7.2f} x {ext[2]:7.2f}")
    if t0 is not None:
        line += f"   耗时 {(time.perf_counter() - t0) * 1000:7.1f} ms"
    print(line)


# ------------------------------------------------------------------ 1. 下采样

def demo_voxel_downsample(pcd):
    """体素下采样：把空间切成小立方体，每格只留一个点（重心）

    为什么要做：
        原始点云动辄几十万点，配准/拟合的复杂度都是 O(n²) 级别。
        下采样把点数降到 1/10，精度损失很小。

    选 voxel_size 的原则：
        略大于点间距。太小没效果，太大会丢掉细节（棱角被磨平）。
    """
    pts = np.asarray(pcd.points)

    # 先估点间距：取每个点到最近邻的距离中位数（抽样算，别全算）
    sample = o3d.geometry.PointCloud()
    idx = np.random.default_rng(0).choice(len(pts), min(2000, len(pts)), replace=False)
    sample.points = o3d.utility.Vector3dVector(pts[idx])
    d = np.asarray(sample.compute_nearest_neighbor_distance())
    spacing = float(np.median(d))
    print(f"\n[1] 体素下采样")
    print(f"    估计点间距（最近邻距离中位数）= {spacing:.3f} mm")

    results = []
    for ratio in (1.0, 2.0, 4.0, 8.0):
        vs = spacing * ratio
        t0 = time.perf_counter()
        down = pcd.voxel_down_sample(voxel_size=vs)
        dt = time.perf_counter() - t0
        n = len(down.points)
        results.append((vs, n, dt))
        print(f"    voxel={vs:6.3f} mm ({ratio:>3.1f}x间距)  "
              f"点数 {n:>7d} ({n/len(pts)*100:5.1f}%)  耗时 {dt*1000:7.1f} ms")

    print(f"    结论：voxel=2x间距时点数降到 "
          f"{results[1][1]/len(pts)*100:.1f}%，是常用折中")
    return pcd.voxel_down_sample(voxel_size=spacing * 2.0)


# ------------------------------------------------------------------ 2. 滤波

def demo_outlier_removal(pcd):
    """统计离群点滤波：每个点到 k 个邻居的平均距离，超过 (均值 + std_ratio*标准差) 就删掉

    为什么要做：
        打架的扫描噪声、反光造成的飞点、配准时的错误对应点，
        都会把拟合/配准结果拉偏——和你之前做圆拟合时 RANSAC 解决的问题同源。

    参数怎么选：
        nb_neighbors：看多少个邻居，20~50
        std_ratio：阈值倍数，越小删得越狠（1.0 激进，2.0 温和）
    """
    print(f"\n[2] 统计离群点滤波")
    print(f"    原始点数 {len(pcd.points)}")

    for nb, sr in ((20, 1.0), (20, 2.0), (50, 2.0)):
        t0 = time.perf_counter()
        clean, ind = pcd.remove_statistical_outlier(nb_neighbors=nb, std_ratio=sr)
        dt = time.perf_counter() - t0
        removed = len(pcd.points) - len(clean.points)
        print(f"    nb={nb:>3d} std={sr:>3.1f}  "
              f"保留 {len(clean.points):>7d}  删除 {removed:>6d} "
              f"({removed/len(pcd.points)*100:5.1f}%)  耗时 {dt*1000:6.1f} ms")

    clean, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    return clean


# ------------------------------------------------------------------ 3. 法向

def demo_normals(pcd):
    """法向估计：对每个点找邻域，用 PCA 拟合局部平面，最小特征值方向就是法向

    为什么要做：
        - 配准：点到平面（point-to-plane）的 ICP 比点到点收敛快得多
        - 拟合：判断"这个点属于哪个面"
        - 特征：FPFH 描述子依赖法向

    为什么必须先下采样：
        法向对噪声极敏感。密集点云里相邻点太近，噪声主导，
        估出来的法向乱翻。通常先下采样到 2~3 倍点间距，再估法向。
    """
    print(f"\n[3] 法向估计")

    for radius_mult, max_nn in ((2.0, 30), (4.0, 30), (4.0, 100)):
        pts = np.asarray(pcd.points)
        # 用中位最近邻距离推半径
        sample = o3d.geometry.PointCloud()
        idx = np.random.default_rng(0).choice(len(pts), min(2000, len(pts)), replace=False)
        sample.points = o3d.utility.Vector3dVector(pts[idx])
        spacing = float(np.median(np.asarray(sample.compute_nearest_neighbor_distance())))
        radius = spacing * radius_mult

        t0 = time.perf_counter()
        q = o3d.geometry.PointCloud(pcd)
        q.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=radius, max_nn=max_nn))
        dt = time.perf_counter() - t0

        nrm = np.asarray(q.normals)
        print(f"    radius={radius:6.2f} mm max_nn={max_nn:>3d}  "
              f"耗时 {dt*1000:7.1f} ms")

    # 法向一致性：相邻点的法向应该接近。用朝向一致性检查质量
    q = o3d.geometry.PointCloud(pcd)
    q.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))
    q.orient_normals_consistent_tangent_plane(k=15)
    nrm = np.asarray(q.normals)

    # 统计法向模长（应该都接近 1）和相邻点法向夹角
    print(f"    法向模长范围 [{np.linalg.norm(nrm, axis=1).min():.4f}, "
          f"{np.linalg.norm(nrm, axis=1).max():.4f}]  (应接近 1.0000)")

    return q


# ------------------------------------------------------------------ 4. 可视化

def visualize(pcds, names):
    """可视化：不同点云用不同颜色，方便对比"""
    palette = [[0.9, 0.6, 0.6], [0.6, 0.6, 0.9], [0.6, 0.9, 0.6],
               [0.9, 0.9, 0.6], [0.9, 0.6, 0.9]]

    geoms = []
    for i, p in enumerate(pcds):
        q = o3d.geometry.PointCloud(p)
        if not q.has_colors():
            q.paint_uniform_color(palette[i % len(palette)])
        geoms.append(q)

    print(f"\n[4] 可视化 {len(geoms)} 个点云")
    for n, g in zip(names, geoms):
        print(f"    - {n} ({len(g.points)} 点)")
    print("    鼠标：左键旋转 / 滚轮缩放 / 中键平移 / Q 退出")

    o3d.visualization.draw_geometries(
        geoms, window_name="Day1 - 点云基础操作",
        width=1200, height=800,
        point_show_normal=False)


# ------------------------------------------------------------------ 主流程

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-vis", action="store_true", help="不弹可视化窗口")
    ap.add_argument("--input", default=os.path.join(DATA, "cube_noisy.ply"))
    args = ap.parse_args()

    print("=" * 78)
    print("Day1 点云基础操作")
    print("=" * 78)

    # 0. 读取
    t0 = time.perf_counter()
    pcd = load(args.input)
    report(pcd, "0. 原始点云", t0)
    print(f"   文件: {os.path.basename(args.input)}")

    # 用带离群点的球更能体现滤波效果，如果用立方体就跳过噪声演示
    sphere_path = os.path.join(DATA, "sphere_outliers.ply")
    sphere = load(sphere_path) if os.path.exists(sphere_path) else None

    # 1. 下采样
    down = demo_voxel_downsample(pcd)

    # 2. 滤波（用带离群点的球演示，效果更明显）
    if sphere is not None:
        print(f"\n[2] 统计离群点滤波（演示：含 200 个离群点的球）")
        report(sphere, "  滤波前", None)
        t0 = time.perf_counter()
        clean, ind = sphere.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        report(clean, "  滤波后", t0)
        removed = len(sphere.points) - len(clean.points)
        print(f"    删除 {removed} 点（其中人为注入的离群点 200 个）")
        print(f"    检出率 {removed/200*100:.1f}%  "
              f"（理想应接近 100%，多删的是球面上的正常点）")
    else:
        clean = demo_outlier_removal(pcd)

    # 3. 法向
    with_normals = demo_normals(down)

    # 4. 可视化
    if not args.no_vis:
        targets = [pcd, down, with_normals]
        names = ["原始", "下采样后", "下采样+法向"]
        if sphere is not None:
            targets = [sphere, clean, down]
            names = ["含离群点", "滤波后", "下采样"]
        visualize(targets, names)
    else:
        print("\n[4] 跳过可视化（--no-vis）")

    print("\n" + "=" * 78)
    print("Day1 完成。要能回答的三个问题：")
    print("=" * 78)
    print("  1. 为什么先下采样再估法向？（噪声敏感性）")
    print("  2. voxel_size 怎么选？（略大于点间距，2x 是常用折中）")
    print("  3. 统计滤波的 nb_neighbors / std_ratio 各控制什么？")


if __name__ == "__main__":
    main()
