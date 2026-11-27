"""合成点云生成器 —— Day1/Day2 都用

为什么先做合成数据：
    真实点云没有真值，你没法判断配准/拟合的结果对不对。
    合成点云的真值是精确已知的（形状、变换都是自己设的），
    可以拿来做"对拍验证"——这是判断算法对不对的唯一硬手段。

    这个思路和你已经做过的 validate_calib.py 完全一致。

用法：
    python synth_data.py            # 生成并保存几种点云到 data/
"""
import os

import numpy as np
import open3d as o3d


# ------------------------------------------------------------------ 基础形状

def make_cube_surface(size=100.0, n_per_edge=60, noise=0.0, seed=0):
    """立方体表面采样（边长 size mm）

    n_per_edge：每条边上采样多少个点
    返回点云（只含点，不含面）
    """
    rng = np.random.default_rng(seed)
    h = size / 2.0
    pts = []

    # 6 个面，每个面用网格采样
    for axis in range(3):
        for sign in (-1, 1):
            u = np.linspace(-h, h, n_per_edge)
            v = np.linspace(-h, h, n_per_edge)
            U, V = np.meshgrid(u, v)
            coords = np.zeros((U.size, 3))
            other = [i for i in range(3) if i != axis]
            coords[:, axis] = sign * h
            coords[:, other[0]] = U.ravel()
            coords[:, other[1]] = V.ravel()
            pts.append(coords)

    pts = np.vstack(pts)
    if noise > 0:
        pts = pts + rng.normal(0, noise, pts.shape)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd


def make_sphere(radius=50.0, n=4000, noise=0.0, seed=0):
    """球面均匀采样"""
    rng = np.random.default_rng(seed)
    # 用正态分布归一化得到均匀球面分布（避免极点聚集）
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    pts = v * radius
    if noise > 0:
        pts = pts + rng.normal(0, noise, pts.shape)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd


def make_plane(width=200.0, height=150.0, n=8000, noise=0.0, seed=0):
    """平面采样（Z=0），常用于测试平面拟合"""
    rng = np.random.default_rng(seed)
    pts = np.column_stack([
        rng.uniform(-width / 2, width / 2, n),
        rng.uniform(-height / 2, height / 2, n),
        np.zeros(n),
    ])
    if noise > 0:
        pts[:, 2] += rng.normal(0, noise, n)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd


def make_cylinder(radius=40.0, height=120.0, n=6000, noise=0.0, seed=0):
    """圆柱侧面采样（轴沿 Z）"""
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, 2 * np.pi, n)
    z = rng.uniform(-height / 2, height / 2, n)
    pts = np.column_stack([
        radius * np.cos(theta),
        radius * np.sin(theta),
        z,
    ])
    if noise > 0:
        pts = pts + rng.normal(0, noise, pts.shape)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd


# ------------------------------------------------------------------ 工具

def random_transform(max_angle_deg=30.0, max_translate=50.0, seed=0):
    """生成随机刚体变换，返回 (4x4 矩阵, 旋转角(度), 平移量(mm))

    这是 Day2 "对拍验证" 的钥匙：
        真值变换已知 -> 把点云变换过去 -> 跑配准 -> 比较结果和真值
    """
    rng = np.random.default_rng(seed)

    # 随机旋转轴 + 随机角度
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    angle = np.radians(rng.uniform(0, max_angle_deg))
    R = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)

    t = rng.uniform(-max_translate, max_translate, 3)

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T, np.degrees(angle), float(np.linalg.norm(t))


def apply_transform(pcd, T):
    """应用 4x4 变换，返回新点云（不修改原对象）"""
    q = o3d.geometry.PointCloud(pcd)
    return q.transform(T)


def info(pcd, name=""):
    """打印点云基本信息"""
    pts = np.asarray(pcd.points)
    if len(pts) == 0:
        print(f"{name}: 空点云")
        return
    bbox = pcd.get_axis_aligned_bounding_box()
    ext = bbox.get_extent()
    print(f"{name}:")
    print(f"  点数      {len(pts)}")
    print(f"  包围盒    {ext[0]:.2f} x {ext[1]:.2f} x {ext[2]:.2f}")
    print(f"  中心      ({pts[:,0].mean():.2f}, {pts[:,1].mean():.2f}, {pts[:,2].mean():.2f})")
    print(f"  XYZ 范围  "
          f"x[{pts[:,0].min():.1f},{pts[:,0].max():.1f}] "
          f"y[{pts[:,1].min():.1f},{pts[:,1].max():.1f}] "
          f"z[{pts[:,2].min():.1f},{pts[:,2].max():.1f}]")


# ------------------------------------------------------------------ 入口

if __name__ == "__main__":
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(out, exist_ok=True)

    print("=" * 60)
    print("生成合成点云")
    print("=" * 60)

    # 干净立方体（配准用的"源"）
    cube = make_cube_surface(size=100.0, n_per_edge=60)
    cube.paint_uniform_color([0.6, 0.6, 0.9])
    info(cube, "立方体 100mm 干净")
    p = os.path.join(out, "cube_clean.ply")
    o3d.io.write_point_cloud(p, cube)
    print(f"  -> {p}\n")

    # 带噪声立方体（配准用的"目标"）
    cube_n = make_cube_surface(size=100.0, n_per_edge=60, noise=1.0, seed=1)
    cube_n.paint_uniform_color([0.9, 0.6, 0.6])
    info(cube_n, "立方体 100mm 噪声 σ=1.0")
    p = os.path.join(out, "cube_noisy.ply")
    o3d.io.write_point_cloud(p, cube_n)
    print(f"  -> {p}\n")

    # 带离群点的球（测滤波）
    sphere = make_sphere(radius=50.0, n=4000, noise=0.5, seed=2)
    rng = np.random.default_rng(99)
    n_out = 200
    outliers = rng.uniform(-150, 150, (n_out, 3))
    pts = np.vstack([np.asarray(sphere.points), outliers])
    sphere_o = o3d.geometry.PointCloud()
    sphere_o.points = o3d.utility.Vector3dVector(pts)
    sphere_o.paint_uniform_color([0.5, 0.9, 0.5])
    info(sphere_o, f"球 r=50 含 {n_out} 个离群点")
    p = os.path.join(out, "sphere_outliers.ply")
    o3d.io.write_point_cloud(p, sphere_o)
    print(f"  -> {p}\n")

    # 平面（测平面拟合）
    plane = make_plane(width=200, height=150, n=8000, noise=0.5, seed=3)
    plane.paint_uniform_color([0.7, 0.7, 0.7])
    info(plane, "平面 200x150 噪声 σ=0.5")
    p = os.path.join(out, "plane.ply")
    o3d.io.write_point_cloud(p, plane)
    print(f"  -> {p}\n")

    # 圆柱（测曲面拟合）
    cyl = make_cylinder(radius=40, height=120, n=6000, noise=0.5, seed=4)
    cyl.paint_uniform_color([0.9, 0.9, 0.5])
    info(cyl, "圆柱 r=40 h=120 噪声 σ=0.5")
    p = os.path.join(out, "cylinder.ply")
    o3d.io.write_point_cloud(p, cyl)
    print(f"  -> {p}\n")

    print("=" * 60)
    print("全部生成完毕，文件在:", out)
    print("=" * 60)
