"""直观感受点云 —— 从"扫出来的原始数据"到"能用的零件"

和原版 look.py 的区别：

    原版                              改进版
    ─────────────────────────────    ─────────────────────────────────────
    一个球，切掉一小块              真实场景：桌面 + 零件 + 扫描噪声 + 飞点
    只有处理后的结果                分步看：原始 → 去地面 → 去离群点
    颜色 = XYZ 归一化混合           颜色 = 高度（jet 色带），一眼看出结构
    没有方向参照                    带坐标轴 + 原点标记，知道尺度
    不知道删了多少                  每步打印点数变化和耗时

核心目标：让每一步"为什么做、做了什么"都能用眼睛看出来。

用法：
    python look.py                # 分步引导，每步按 Q 进入下一步
    python look.py --step 2       # 直接看第 2 步
    python look.py --size 4       # 点大小
"""
import argparse
import os
import sys

import numpy as np
import open3d as o3d

# ---------------------------------------------------------------- 颜色工具

# jet 色带近似（蓝 -> 青 -> 绿 -> 黄 -> 红），比 matplotlib 依赖少
_JET = np.array([
    [0.00, 0.00, 0.50],
    [0.00, 0.00, 1.00],
    [0.00, 0.50, 1.00],
    [0.00, 1.00, 1.00],
    [0.50, 1.00, 0.50],
    [1.00, 1.00, 0.00],
    [1.00, 0.50, 0.00],
    [1.00, 0.00, 0.00],
    [0.50, 0.00, 0.00],
])


def color_by_height(pcd, lo=None, hi=None, robust=True):
    """按 Z 高度着色（jet 色带）。比 XYZ 归一化更能读出结构

    robust=True 时用 1%~99% 分位数作为色标范围，
    避免少数飞点把整幅图的色标拉偏。
    """
    pts = np.asarray(pcd.points)
    if len(pts) == 0:
        return pcd
    z = pts[:, 2]
    if lo is None:
        lo = np.percentile(z, 1) if robust else z.min()
    if hi is None:
        hi = np.percentile(z, 99) if robust else z.max()
    if hi - lo < 1e-9:
        t = np.zeros(len(z))
    else:
        t = np.clip((z - lo) / (hi - lo), 0, 1)

    # 线性插值取色
    idx = t * (len(_JET) - 1)
    i0 = np.floor(idx).astype(int)
    i1 = np.minimum(i0 + 1, len(_JET) - 1)
    frac = (idx - i0)[:, None]
    colors = _JET[i0] * (1 - frac) + _JET[i1] * frac

    q = o3d.geometry.PointCloud(pcd)
    q.colors = o3d.utility.Vector3dVector(colors)
    return q


# ---------------------------------------------------------------- 构造场景

def build_scan_scene(seed=0, n_table=20000, n_part=12000, n_fly=150):
    """构造一个"工业扫描"场景：

        桌面（Z=0 平面）
        + 立方体零件（60mm，坐在桌面上）
        + 圆柱凸台（φ50 x 40，旁边）
        + 扫描噪声
        + 飞点（反光造成的离群点）

    这样"去地面"和"去离群点"才有物理意义。
    """
    rng = np.random.default_rng(seed)
    parts = []

    # ① 桌面：300x300 mm，Z=0，带轻微噪声
    table = np.column_stack([
        rng.uniform(-150, 150, n_table),
        rng.uniform(-150, 150, n_table),
        rng.normal(0, 0.3, n_table),
    ])
    parts.append(table)

    # ② 立方体零件：60mm，底面贴桌面。用内置网格采样，位置明确
    box = o3d.geometry.TriangleMesh.create_box(
        width=60.0, height=60.0, depth=60.0)
    box.translate([30.0, -30.0, 0.0])      # x 30~90, y -30~30, z 0~60
    cube = np.asarray(
        box.sample_points_uniformly(number_of_points=n_part).points)
    cube += rng.normal(0, 0.4, cube.shape)
    parts.append(cube)

    # ③ 圆柱凸台：半径 25mm，高 40mm
    cyl_mesh = o3d.geometry.TriangleMesh.create_cylinder(
        radius=25.0, height=40.0, resolution=40)
    cyl_mesh.translate([-70.0, 0.0, 20.0])  # 底面 z=0
    cyl = np.asarray(
        cyl_mesh.sample_points_uniformly(number_of_points=n_part // 2).points)
    cyl += rng.normal(0, 0.4, cyl.shape)
    parts.append(cyl)

    # ④ 飞点：反光/杂散光造成的离群点（散布在空间里）
    fly = np.column_stack([
        rng.uniform(-150, 150, n_fly),
        rng.uniform(-150, 150, n_fly),
        rng.uniform(0, 140, n_fly),
    ])
    parts.append(fly)

    pts = np.vstack(parts)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd


# ---------------------------------------------------------------- 显示

def frame_refs(size=100.0):
    """坐标轴 + 原点标记，用来感知方向和尺度"""
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=size, origin=[0, 0, 0])
    return axes


def render_preview(geoms, out_path, lookat=None, zoom=0.72,
                   front=(1.0, -1.3, 0.85), up=(0.0, 0.0, 1.0)):
    """离屏渲染一张固定 45 度俯视图（没显示器也能出图）"""
    try:
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False, width=920, height=660)
        for g in geoms:
            vis.add_geometry(g)

        opt = vis.get_render_option()
        opt.background_color = np.array([0.97, 0.97, 0.98])
        opt.point_size = 2

        ctr = vis.get_view_control()
        if lookat is None:
            lookat = geoms[0].get_axis_aligned_bounding_box().get_center()
        ctr.set_lookat(lookat)
        ctr.set_up(up)
        ctr.set_front(front)
        ctr.set_zoom(zoom)

        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(out_path, do_render=True)
        vis.destroy_window()
        return True
    except Exception as exc:
        print(f"  离屏渲染不可用: {type(exc).__name__}: {exc}")
        return False


def show(geoms, names, point_size=3.0, bg=(0.08, 0.08, 0.10, 1.0)):
    print("\n" + "-" * 66)
    for n, g in zip(names, geoms):
        cnt = len(g.points) if isinstance(g, o3d.geometry.PointCloud) else "-"
        print(f"  {n:<34} 点数 {cnt}")
    print("-" * 66)
    print("  鼠标：左键旋转 / 滚轮缩放 / Ctrl+左键平移")
    print("  键盘：Q 退出（进入下一步）")
    print("-" * 66)

    # Open3D 版本差异很大：
    #   0.19+  draw()           参数 title / bg_color 4分量 / point_size int
    #   <=0.18 draw_geometries() 参数 window_name / 无 point_size
    # 这里先试新 API，任何异常都回退旧 API —— 保证哪个版本都能跑。
    title = " -> ".join(names) if len(names) <= 3 else "点云"
    bg4 = np.asarray(bg, dtype=np.float32).ravel()
    if bg4.size == 3:
        bg4 = np.append(bg4, 1.0)

    try:
        o3d.visualization.draw(
            geoms,
            title=title,
            width=1280, height=800,
            point_size=int(round(point_size)),
            show_axes=False,
            bg_color=bg4,
        )
        return
    except Exception as exc:
        print(f"  [新 API 不可用: {type(exc).__name__}，回退旧接口]")

    try:
        o3d.visualization.draw_geometries(
            geoms,
            window_name=title,
            width=1280, height=800,
        )
    except Exception as exc:
        print(f"  [可视化失败: {type(exc).__name__}: {exc}]")
        print("  改用无窗口模式： python look.py --preview")


# ---------------------------------------------------------------- 主流程

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, default=0,
                    help="0=全部分步显示, 1=原始, 2=去地面, 3=去离群点, 4=全部对比")
    ap.add_argument("--size", type=int, default=3, help="点大小（整数）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--preview", action="store_true",
                    help="不开窗口，只生成预览图到 previews/")
    args = ap.parse_args()

    print("=" * 66)
    print("点云直观感受：从原始扫描到可用零件")
    print("=" * 66)

    pcd = build_scan_scene(args.seed)
    pts = np.asarray(pcd.points)
    print(f"\n扫描场景点数 {len(pts)}")
    print(f"  X [{pts[:,0].min():7.1f}, {pts[:,0].max():7.1f}] mm")
    print(f"  Y [{pts[:,1].min():7.1f}, {pts[:,1].max():7.1f}] mm")
    print(f"  Z [{pts[:,2].min():7.1f}, {pts[:,2].max():7.1f}] mm")
    print("\n场景构成：桌面(300x300) + 立方体零件(60mm) + 圆柱(φ50x40) + 飞点")

    pv_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "previews")
    if args.preview:
        os.makedirs(pv_dir, exist_ok=True)

    def hook(geoms, names, tag):
        """预览模式存图，否则开窗口"""
        if args.preview:
            p = os.path.join(pv_dir, f"{tag}.png")
            ok = render_preview(geoms, p)
            print(f"  -> {'已保存 ' + p if ok else '渲染失败'}")
        else:
            show(geoms, names, args.size)

    # ---------- 步骤 1：原始 ----------
    if args.step in (0, 1):
        print("\n" + "=" * 66)
        print("步骤 1：原始扫描 —— 颜色 = 高度（蓝低 红高）")
        print("=" * 66)
        print("看出什么：")
        print("  · 一片平坦的蓝绿色 = 桌面（Z≈0）")
        print("  · 突出的暖色方块/圆柱 = 零件")
        print("  · 零散的孤点 = 飞点（反光噪声）")
        print("问题：无法直接对零件做尺寸测量——桌面和飞点混在里面")
        raw = color_by_height(pcd)
        hook([raw, frame_refs()], ["原始扫描"], "1_raw")

    # ---------- 步骤 2：去地面 ----------
    ground_z = 2.0
    keep = pts[:, 2] > ground_z
    pcd_nog = o3d.geometry.PointCloud()
    pcd_nog.points = o3d.utility.Vector3dVector(pts[keep])
    print("\n" + "=" * 66)
    print(f"步骤 2：去地面 —— 高度过滤 Z > {ground_z} mm")
    print("=" * 66)
    print(f"  删除 {len(pts) - keep.sum()} 点（桌面），保留 {keep.sum()} 点")
    print("工业里更稳的做法是 RANSAC 拟合平面再删除，而不是直接切 Z")
    print("（因为工件可能倾斜，平面不一定是水平的）")
    if args.step in (0, 2):
        hook([color_by_height(pcd_nog), frame_refs()], ["去地面后"], "2_noground")

    # ---------- 步骤 3：去离群点 ----------
    t0 = __import__("time").perf_counter()
    pcd_clean, ind = pcd_nog.remove_statistical_outlier(
        nb_neighbors=20, std_ratio=2.0)
    dt = (__import__("time").perf_counter() - t0) * 1000
    print("\n" + "=" * 66)
    print("步骤 3：统计离群点滤波")
    print("=" * 66)
    print(f"  {len(pcd_nog.points)} → {len(pcd_clean.points)} 点"
          f"（删除 {len(pcd_nog.points) - len(pcd_clean.points)}）")
    print(f"  耗时 {dt:.1f} ms")
    print("判据：每个点到 20 个邻居的平均距离，超过均值+2σ 就删")
    if args.step in (0, 3):
        hook([color_by_height(pcd_clean), frame_refs()], ["去地面 + 去离群点"], "3_clean")

    # ---------- 步骤 4：并排对比 ----------
    if args.step in (0, 4):
        print("\n" + "=" * 66)
        print("步骤 4：并排对比 —— 左原始，右处理完")
        print("=" * 66)
        # 用统一色标范围（取主体的 1%~99% 分位），才能公平对比
        lo = float(np.percentile(pts[:, 2], 1))
        hi = float(np.percentile(pts[:, 2], 99))
        a = color_by_height(pcd, lo, hi, robust=False)
        b = color_by_height(pcd_clean, lo, hi, robust=False)

        # 右侧整体平移，避免重叠（间距适中，两边都看得清）
        offset = np.array([320.0, 0, 0])
        b_shift = o3d.geometry.PointCloud(b)
        b_shift.translate(offset.ravel())

        axes_a = o3d.geometry.TriangleMesh.create_coordinate_frame(60, [0, 0, 0])
        axes_b = o3d.geometry.TriangleMesh.create_coordinate_frame(60, [320, 0, 0])

        if args.preview:
            p = os.path.join(pv_dir, "4_compare.png")
            ok = render_preview([a, b_shift, axes_a, axes_b], p,
                                lookat=np.array([160.0, 0.0, 20.0]), zoom=0.62)
            print(f"  -> {'已保存 ' + p if ok else '渲染失败'}")
        else:
            show([a, b_shift, axes_a, axes_b],
                 ["左:原始", "右:处理后"], args.size)

    print("\n" + "=" * 66)
    print("三个能讲清的判断")
    print("=" * 66)
    print("  1. 为什么不能直接对原始点云做测量？")
    print("     → 桌面和飞点会污染拟合/配准，比如平面拟合会被飞点拉偏")
    print("  2. 为什么去地面不能简单切 Z？")
    print("     → 工件可能倾斜或放置不正，正确做法是 RANSAC 拟合主平面")
    print("  3. 统计滤波删不干净怎么办？")
    print("     → 它只能删'明显远离主体'的点；靠近主体但错误的点要靠")
    print("       RANSAC / 一致性检查 / 多视角融合")


if __name__ == "__main__":
    main()
