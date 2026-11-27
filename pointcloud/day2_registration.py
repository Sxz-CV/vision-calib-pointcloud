"""Day2：点云配准 + 对拍验证

核心问题：怎么知道配准结果是对的？
    真实数据没有真值，你只能"看着像"。所以先用合成数据：
        已知真值变换 T_true  ->  把源点云变过去得到目标
        跑配准得到 T_est
        比较 T_est 与 T_true：旋转误差(度) + 平移误差(mm)
    这就是对拍。和你做标定时用 validate_calib.py 的思路完全一样。

三个实验：
    实验1  小初始误差（5°, 10mm）-> ICP 直接能收敛
    实验2  大初始误差（30°, 60mm）-> ICP 失败，需要 FPFH+RANSAC 粗配准
    实验3  扫描初始误差 -> 配准误差曲线（找出 ICP 的收敛域）

跑法：
    python day2_registration.py                # 全部实验 + 可视化
    python day2_registration.py --no-vis       # 只打印数字
    python day2_registration.py --exp 2        # 只跑实验2
"""
import argparse
import os
import time

import numpy as np
import open3d as o3d

import synth_data

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

# 结果判定阈值（按点云尺度来定，别写死绝对值）
ROT_OK_DEG = 2.0      # 旋转误差 < 2° 算配准成功
TRANS_OK_MM = 3.0     # 平移误差 < 3mm 算成功


# ------------------------------------------------------------------ 误差度量

def pose_error(T_est, T_true):
    """比较两个刚体变换，返回 (旋转误差度, 平移误差mm)

    旋转误差：R_err = R_est^T @ R_true，其旋转角 = arccos((tr-1)/2)
    平移误差：||t_est - t_true||
    """
    R_est, t_est = T_est[:3, :3], T_est[:3, 3]
    R_true, t_true = T_true[:3, :3], T_true[:3, 3]

    # 归一化，消掉可能的尺度/数值误差
    U, _, Vt = np.linalg.svd(R_est @ R_true.T)
    R_rel = U @ Vt
    cos = np.clip((np.trace(R_rel) - 1.0) / 2.0, -1.0, 1.0)
    rot_err = float(np.degrees(np.arccos(cos)))
    trans_err = float(np.linalg.norm(t_est - t_true))
    return rot_err, trans_err


def fitness_of(src, tgt, T, thr):
    """配准后，source 有多少比例的点能找到 target 中 thr 以内的对应点"""
    s = o3d.geometry.PointCloud(src).transform(T)
    d = np.asarray(s.compute_point_cloud_distance(tgt))
    return float((d < thr).mean()), float(np.sqrt(np.mean(d[d < thr] ** 2))) if (d < thr).any() else 999.0


# ------------------------------------------------------------------ 造数据对

def make_pair(pcd_src, angle_deg, trans_mm, seed=0, noise=0.0, keep_ratio=1.0):
    """用已知变换造目标点云

    keep_ratio < 1 时模拟"单视角扫描"：只保留目标的一部分（按高度裁掉下半）。
    真实场景里相机只能看到物体一面，重叠率通常 40%~70%，
    这正是 ICP 失败的主要原因——不是角度大，而是"看到的不全"。

    返回 (source, target, T_true, 角度, 平移)
    """
    rng = np.random.default_rng(seed)

    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    R = o3d.geometry.get_rotation_matrix_from_axis_angle(
        axis * np.radians(angle_deg))
    direction = rng.normal(size=3)
    direction /= np.linalg.norm(direction)
    t = direction * trans_mm

    T_true = np.eye(4)
    T_true[:3, :3] = R
    T_true[:3, 3] = t

    target = o3d.geometry.PointCloud(pcd_src).transform(T_true)
    if noise > 0:
        pts = np.asarray(target.points) + rng.normal(0, noise, (len(target.points), 3))
        target.points = o3d.utility.Vector3dVector(pts)

    if keep_ratio < 1.0:
        pts = np.asarray(target.points)
        thr = np.percentile(pts[:, 2], 100 * (1 - keep_ratio))
        keep = pts[:, 2] > thr
        t2 = o3d.geometry.PointCloud()
        t2.points = o3d.utility.Vector3dVector(pts[keep])
        target = t2

    return pcd_src, target, T_true, angle_deg, float(np.linalg.norm(t))


# ------------------------------------------------------------------ 两种配准

def icp_only(src, tgt, init=np.eye(4), max_dist=8.0, iters=80):
    """直接 ICP（点到点），从 init 位姿开始"""
    t0 = time.perf_counter()
    reg = o3d.pipelines.registration.registration_icp(
        src, tgt, max_dist, init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=iters))
    dt = (time.perf_counter() - t0) * 1000
    return reg, dt


def fpfh_ransac_then_icp(src, tgt, voxel=5.0, max_dist=2.0, iters=80):
    """粗配准（FPFH + RANSAC）-> 精配准（ICP）"""
    t_start = time.perf_counter()

    # ① 下采样 + 法向（FPFH 需要法向）
    def prep(pcd):
        d = pcd.voxel_down_sample(voxel)
        d.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel * 2.5, max_nn=30))
        f = o3d.pipelines.registration.compute_fpfh_feature(
            d, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 5.0, max_nn=100))
        return d, f

    src_d, src_f = prep(src)
    tgt_d, tgt_f = prep(tgt)

    # ② RANSAC 粗配准
    dist_thr = voxel * 1.5
    coarse = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src_d, tgt_d, src_f, tgt_f,
        mutual_filter=True,
        max_correspondence_distance=dist_thr,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=3,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(dist_thr),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    t_coarse = time.perf_counter()

    # ③ 用粗配准结果做初值，跑 ICP 精配准
    reg = o3d.pipelines.registration.registration_icp(
        src, tgt, max_dist, coarse.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=iters))
    t_end = time.perf_counter()

    return {
        "T_coarse": coarse.transformation,
        "coarse_fitness": coarse.fitness,
        "T_est": reg.transformation,
        "fitness": reg.fitness,
        "inlier_rmse": reg.inlier_rmse,
        "t_coarse_ms": (t_coarse - t_start) * 1000,
        "t_icp_ms": (t_end - t_coarse) * 1000,
        "t_total_ms": (t_end - t_start) * 1000,
    }


# ------------------------------------------------------------------ 实验

def exp1_small_init(pcd, no_vis=False):
    print("\n" + "=" * 78)
    print("实验1：小初始误差（0~15deg / 0~30mm，完整重叠）—— ICP 应直接收敛")
    print("=" * 78)

    results = []
    for ang, tr in ((0, 0), (5, 10), (10, 20), (15, 30)):
        src, tgt, T_true, a, t = make_pair(pcd, ang, tr, seed=42, noise=0.3)
        reg, dt = icp_only(src, tgt, np.eye(4), max_dist=15.0)
        rot, tra = pose_error(reg.transformation, T_true)
        ok = rot < ROT_OK_DEG and tra < TRANS_OK_MM
        results.append((ang, tr, rot, tra, reg.fitness, dt, ok))
        print(f"  真值 {ang:>2.0f}° / {tr:>3.0f}mm   "
              f"估计误差 {rot:6.2f}° / {tra:6.2f}mm   "
              f"fitness {reg.fitness:.3f}   {dt:6.1f} ms   "
              f"{'✓' if ok else '✗'}")

    print()
    print("  看点：目标点云 = 源点云施加真值变换，估计误差应为 0")
    print("        所以'估计误差'= T_est 与 T_true 的差，理想应为 0。")
    return results


def exp2_large_init(pcd, no_vis=False):
    print("\n" + "=" * 78)
    print("实验2：大初始误差（30°, 60mm）—— ICP 会陷入局部最优，需要粗配准")
    print("=" * 78)

    src, tgt, T_true, ang, tr = make_pair(
        pcd, 30, 60, seed=7, noise=0.3, keep_ratio=0.5)
    print(f"  真值: {ang:.0f} deg / {tr:.0f} mm,  overlap ~50% (single-view scan)")
    print(f"  src {len(src.points)} pts, tgt {len(tgt.points)} pts")

    # ① 纯 ICP
    reg_icp, dt_icp = icp_only(src, tgt, np.eye(4), max_dist=20.0)
    rot, tra = pose_error(reg_icp.transformation, T_true)
    print(f"  [纯 ICP]")
    print(f"    误差 {rot:6.2f}° / {tra:7.2f}mm   fitness {reg_icp.fitness:.3f}   "
          f"{dt_icp:.0f} ms   {'✓' if rot < ROT_OK_DEG and tra < TRANS_OK_MM else '✗ 失败'}")

    # ② FPFH + RANSAC + ICP
    t0 = time.perf_counter()
    r = fpfh_ransac_then_icp(src, tgt)
    rot_c, tra_c = pose_error(r["T_coarse"], T_true)
    rot_f, tra_f = pose_error(r["T_est"], T_true)
    print(f"\n  [FPFH + RANSAC + ICP]")
    print(f"    粗配准后   {rot_c:6.2f}° / {tra_c:7.2f}mm   "
          f"fitness {r['coarse_fitness']:.3f}   {r['t_coarse_ms']:.0f} ms")
    print(f"    精配准后   {rot_f:6.2f}° / {tra_f:7.2f}mm   "
          f"fitness {r['fitness']:.3f}   {r['t_icp_ms']:.0f} ms")
    print(f"    总耗时 {r['t_total_ms']:.0f} ms   "
          f"{'✓' if rot_f < ROT_OK_DEG and tra_f < TRANS_OK_MM else '✗ 仍失败'}")

    if not no_vis:
        # 三栏布局：左=源 / 中=目标 / 右=配准结果(绿叠红)
        GAP = 250.0
        src_pc = o3d.geometry.PointCloud(src)
        src_pc.paint_uniform_color([0.20, 0.45, 0.90])      # 蓝：源
        tgt_pc = o3d.geometry.PointCloud(tgt)
        tgt_pc.paint_uniform_color([0.85, 0.25, 0.25])      # 红：目标

        src_aligned = o3d.geometry.PointCloud(src).transform(r["T_est"])
        src_aligned.paint_uniform_color([0.15, 0.80, 0.25])  # 绿：配准后的源

        tgt_mid = o3d.geometry.PointCloud(tgt).translate([GAP, 0, 0])

        tgt_right = o3d.geometry.PointCloud(tgt).translate([GAP * 2, 0, 0])
        tgt_right.paint_uniform_color([0.85, 0.25, 0.25])
        src_aligned.translate([GAP * 2, 0, 0])

        axes = [o3d.geometry.TriangleMesh.create_coordinate_frame(50, [x, 0, 0])
                for x in (0, GAP, GAP * 2)]

        print()
        print("  可视化三栏：")
        print("    左  蓝 = 源点云（完整立方体 CAD）")
        print("    中  红 = 目标点云（真值变换 + 只保留上半 50%，模拟单视角扫描）")
        print("    右  绿 = 配准后的源，红 = 目标")
        print("    -> 右侧绿红贴合 = 配准成功；错开 = 失败")

        try:
            o3d.visualization.draw_geometries(
                [src_pc, tgt_mid, tgt_right, src_aligned] + axes,
                window_name="Day2 配准: 左源 | 中目标 | 右配准结果",
                width=1600, height=800)
        except Exception as exc:
            print(f"  可视化失败: {type(exc).__name__}: {exc}")



def exp3_sweep(pcd, no_vis=False):
    print()
    print("=" * 78)
    print("实验3：初始误差扫描 —— 找出 ICP 的收敛域（完整重叠）")
    print("=" * 78)
    print(f"  成功判据：旋转误差 < {ROT_OK_DEG} 且 平移误差 < {TRANS_OK_MM} mm")
    print("  每档跑 3 次，记录成功率与最好一次的误差")
    print()
    print(f"  {'初始误差':>13} | {'纯 ICP':>21} | {'FPFH+RANSAC+ICP':>21} | {'耗时':>8}")
    print("  " + "-" * 74)

    rows = []
    for ang in (0, 10, 20, 30, 45, 60, 90, 120, 180):
        tr = ang * 2.0
        ok_icp = ok_fpfh = 0
        rot_icp = rot_fpfh = 999.0
        tra_icp = tra_fpfh = 999.0
        t_fpfh = 0.0
        for trial in range(3):
            src, tgt, T_true, _, _ = make_pair(
                pcd, ang, tr, seed=100 + trial, noise=0.3, keep_ratio=1.0)

            reg, _ = icp_only(src, tgt, np.eye(4), max_dist=25.0)
            r1, t1 = pose_error(reg.transformation, T_true)
            if r1 < ROT_OK_DEG and t1 < TRANS_OK_MM:
                ok_icp += 1
            if r1 + t1 < rot_icp + tra_icp:
                rot_icp, tra_icp = r1, t1

            r2 = fpfh_ransac_then_icp(src, tgt)
            rr, tt = pose_error(r2["T_est"], T_true)
            if rr < ROT_OK_DEG and tt < TRANS_OK_MM:
                ok_fpfh += 1
            if rr + tt < rot_fpfh + tra_fpfh:
                rot_fpfh, tra_fpfh = rr, tt
            t_fpfh = r2["t_total_ms"]

        rows.append((ang, ok_icp / 3, ok_fpfh / 3, rot_icp, tra_icp,
                     rot_fpfh, tra_fpfh, t_fpfh))
        a = "OK" if ok_icp == 3 else ("部分" if ok_icp else "失败")
        b = "OK" if ok_fpfh == 3 else ("部分" if ok_fpfh else "失败")
        print(f"  {ang:>5.0f}deg/{tr:>4.0f}mm | {ok_icp}/3 {a:<2} "
              f"{rot_icp:6.1f}deg/{tra_icp:6.1f}mm | {ok_fpfh}/3 {b:<2} "
              f"{rot_fpfh:6.1f}deg/{tra_fpfh:6.1f}mm | {t_fpfh:6.0f}ms")

    print()
    print("  注：立方体是高度对称形状（转 90/180 度后外形一致），")
    print("      所以 >90 度时 FPFH 也分辨不出真实转角——这是数据本身的歧义，")
    print("      不是算法缺陷。换不对称物体（如 L 形）可改善。")
    return rows


# ------------------------------------------------------------------ 入口

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-vis", action="store_true")
    ap.add_argument("--exp", type=int, default=0, help="0=全部, 1/2/3=单个实验")
    args = ap.parse_args()

    cube_path = os.path.join(DATA, "cube_noisy.ply")
    if not os.path.exists(cube_path):
        print("没找到点云，先跑 python synth_data.py")
        return
    pcd = o3d.io.read_point_cloud(cube_path)
    print(f"源点云：{len(pcd.points)} 点，包围盒 "
          f"{pcd.get_axis_aligned_bounding_box().get_extent().round(1)}")

    if args.exp in (0, 1):
        exp1_small_init(pcd, args.no_vis)
    if args.exp in (0, 2):
        exp2_large_init(pcd, args.no_vis)
    if args.exp in (0, 3):
        exp3_sweep(pcd, args.no_vis)

    print("\n" + "=" * 78)
    print("Day2 要能回答的四个问题")
    print("=" * 78)
    print("  1. 怎么知道配准结果是对的？")
    print("     -> 合成数据：已知真值 T_true，比较 T_est 的旋转角和平移量")
    print("  2. ICP 为什么会失败？")
    print("     -> 它只找最近点，初始位姿差就收敛到错误的局部极小")
    print("  3. FPFH + RANSAC 解决的是什么？")
    print("     -> 从任意初始位姿找到粗对齐，把误差降到 ICP 的收敛域内")
    print("  4. 怎么判断配准够不够好？")
    print("     -> fitness（重叠比例）+ inlier_rmse（对应点残差）+ 与真值的偏差")


if __name__ == "__main__":
    main()
