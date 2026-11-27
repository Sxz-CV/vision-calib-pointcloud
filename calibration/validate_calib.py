"""标定的金标准验证：用合成数据检验标定算法本身

思路：
    1. 自己设定"真值"内参 K_true 和畸变 dist_true
    2. 生成随机位姿，把棋盘 3D 点投影成 2D 像素点（这时的真值是精确已知的）
    3. 加已知强度的噪声，模拟真实检测误差
    4. 跑 cv2.calibrateCamera，看能不能把 K_true 恢复出来
    5. 扫噪声水平，得到"噪声 -> 参数误差"曲线

这个实验不需要任何实物、卡尺或照片，却能回答两个关键问题：
    - 我的标定流程（含亚像素窗口选择）是否正确？
    - 这张图的噪声水平下，内参能准到什么程度？

运行：
    python validate_calib.py
"""
import cv2
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ---------------- 真值设定 ----------------

COLS, ROWS = 5, 4          # 内角点
SQUARE = 25.0              # mm
IMG_W, IMG_H = 640, 480

K_TRUE = np.array([[800.0, 0.0, 320.0],
                   [0.0, 800.0, 240.0],
                   [0.0, 0.0, 1.0]])
DIST_TRUE = np.array([-0.15, 0.05, 0.0005, -0.0005, 0.0])   # k1,k2,p1,p2,k3


def board_object_points():
    """棋盘角点的世界坐标（Z=0 平面）"""
    objp = np.zeros((COLS * ROWS, 3), np.float32)
    objp[:, :2] = np.mgrid[0:COLS, 0:ROWS].T.reshape(-1, 2) * SQUARE
    return objp


def random_poses(n, rng, max_tilt=45.0):
    """生成 n 个随机位姿：绕 XYZ 倾斜 + 随机平移，都朝向相机"""
    poses = []
    for _ in range(n):
        ang = np.radians(rng.uniform(-max_tilt, max_tilt, 3))
        Rx = cv2.Rodrigues(np.array([ang[0], 0, 0]))[0]
        Ry = cv2.Rodrigues(np.array([0, ang[1], 0]))[0]
        Rz = cv2.Rodrigues(np.array([0, 0, ang[2]]))[0]
        R = Rz @ Ry @ Rx
        # 平移到棋盘大致落在画面内（距离 250~450 mm）
        z = rng.uniform(250, 450)
        xy = rng.uniform(-60, 60, 2)
        t = np.array([xy[0], xy[1], z], dtype=float)
        rvec = cv2.Rodrigues(R)[0]
        poses.append((rvec, t.reshape(3, 1)))
    return poses


def make_observations(poses, objp, noise_px, rng):
    """按真值投影，加噪声，返回 (objpoints, imgpoints)"""
    obj_list, img_list = [], []
    for rvec, tvec in poses:
        proj, _ = cv2.projectPoints(objp, rvec, tvec, K_TRUE, DIST_TRUE)
        proj = proj.reshape(-1, 2)
        # 只保留全部落在画面内的位姿
        if (proj[:, 0].min() < 0 or proj[:, 0].max() >= IMG_W or
                proj[:, 1].min() < 0 or proj[:, 1].max() >= IMG_H):
            continue
        noise = rng.normal(0, noise_px, proj.shape)
        obj_list.append(objp.copy())
        img_list.append((proj + noise).astype(np.float32).reshape(-1, 1, 2))
    return obj_list, img_list


def calibrate(obj_list, img_list):
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_list, img_list, (IMG_W, IMG_H), None, None, flags=0)
    return ret, K, dist


def report(tag, ret, K, dist, n_views):
    fx_err = (K[0, 0] - K_TRUE[0, 0]) / K_TRUE[0, 0] * 100
    fy_err = (K[1, 1] - K_TRUE[1, 1]) / K_TRUE[1, 1] * 100
    cx_err = K[0, 2] - K_TRUE[0, 2]
    cy_err = K[1, 2] - K_TRUE[1, 2]
    k1_err = dist.ravel()[0] - DIST_TRUE[0]
    print(f"{tag:22s} n={n_views:3d}  RMS={ret:7.4f}px  "
          f"fx误差={fx_err:+6.2f}%  fy误差={fy_err:+6.2f}%  "
          f"cx偏差={cx_err:+6.2f}px  cy偏差={cy_err:+6.2f}px  "
          f"k1误差={k1_err:+.4f}")
    return fx_err, cx_err, k1_err


# ---------------- 实验 1：零噪声，验证算法正确性 ----------------

def exp_noise_free():
    print("=" * 108)
    print("实验 1：零噪声 —— 标定算法能不能精确恢复真值？")
    print("=" * 108)
    print(f"真值: fx=fy={K_TRUE[0,0]:.1f}  cx={K_TRUE[0,2]:.1f}  cy={K_TRUE[1,2]:.1f}  "
          f"k1={DIST_TRUE[0]:.3f}")
    rng = np.random.default_rng(0)
    objp = board_object_points()
    poses = random_poses(40, rng)

    all_poses = []
    for rvec, tvec in poses:
        proj, _ = cv2.projectPoints(objp, rvec, tvec, K_TRUE, DIST_TRUE)
        p = proj.reshape(-1, 2)
        if (p[:, 0].min() >= 0 and p[:, 0].max() < IMG_W and
                p[:, 1].min() >= 0 and p[:, 1].max() < IMG_H):
            all_poses.append((rvec, tvec))
    print(f"可用位姿: {len(all_poses)} 个（投影完整落在画面内）\n")

    obj_list, img_list = make_observations(all_poses, objp, 0.0, rng)
    ret, K, dist = calibrate(obj_list, img_list)
    report("零噪声", ret, K, dist, len(obj_list))
    print()


# ---------------- 实验 2：噪声扫描 ----------------

def exp_noise_sweep():
    print("=" * 108)
    print("实验 2：噪声水平 -> 内参误差（每组重复 10 次取中位数）")
    print("=" * 108)
    objp = board_object_points()
    levels = [0.0, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0]
    rows = []

    for noise in levels:
        fx_errs, cx_errs, k1_errs, rmss = [], [], [], []
        for trial in range(10):
            rng = np.random.default_rng(1000 + trial)
            poses = random_poses(40, rng)
            valid = []
            for rvec, tvec in poses:
                proj, _ = cv2.projectPoints(objp, rvec, tvec, K_TRUE, DIST_TRUE)
                p = proj.reshape(-1, 2)
                if (p[:, 0].min() >= 0 and p[:, 0].max() < IMG_W and
                        p[:, 1].min() >= 0 and p[:, 1].max() < IMG_H):
                    valid.append((rvec, tvec))
            obj_list, img_list = make_observations(valid, objp, noise, rng)
            if len(obj_list) < 6:
                continue
            ret, K, dist = calibrate(obj_list, img_list)
            fx_errs.append(abs(K[0, 0] - K_TRUE[0, 0]) / K_TRUE[0, 0] * 100)
            cx_errs.append(abs(K[0, 2] - K_TRUE[0, 2]))
            k1_errs.append(abs(dist.ravel()[0] - DIST_TRUE[0]))
            rmss.append(ret)

        med = np.median
        rows.append((noise, med(fx_errs), med(cx_errs), med(k1_errs), med(rmss)))
        print(f"  噪声 {noise:4.1f} px  ->  RMS {med(rmss):6.3f} px   "
              f"fx误差 {med(fx_errs):5.2f}%   cx偏差 {med(cx_errs):6.2f} px   "
              f"k1误差 {med(k1_errs):.4f}")
    return rows


# ---------------- 实验 3：图片数量扫描 ----------------

def exp_view_count():
    print("\n" + "=" * 108)
    print("实验 3：图片数量 -> 内参误差（噪声固定 0.3 px，每档重复 10 次取中位数）")
    print("=" * 108)
    objp = board_object_points()
    counts = [4, 6, 8, 10, 15, 20, 30]
    rows = []
    for n in counts:
        fx_errs, cx_errs = [], []
        for trial in range(10):
            rng = np.random.default_rng(2000 + trial)
            poses = random_poses(40, rng)
            valid = []
            for rvec, tvec in poses:
                proj, _ = cv2.projectPoints(objp, rvec, tvec, K_TRUE, DIST_TRUE)
                p = proj.reshape(-1, 2)
                if (p[:, 0].min() >= 0 and p[:, 0].max() < IMG_W and
                        p[:, 1].min() >= 0 and p[:, 1].max() < IMG_H):
                    valid.append((rvec, tvec))
            valid = valid[:n]
            if len(valid) < 4:
                continue
            obj_list, img_list = make_observations(valid, objp, 0.3, rng)
            if len(obj_list) < 4:
                continue
            ret, K, dist = calibrate(obj_list, img_list)
            fx_errs.append(abs(K[0, 0] - K_TRUE[0, 0]) / K_TRUE[0, 0] * 100)
            cx_errs.append(abs(K[0, 2] - K_TRUE[0, 2]))
        med = np.median
        rows.append((n, med(fx_errs), med(cx_errs)))
        print(f"  {n:2d} 张  ->  fx误差 {med(fx_errs):5.2f}%   cx偏差 {med(cx_errs):6.2f} px")
    return rows


if __name__ == "__main__":
    exp_noise_free()
    noise_rows = exp_noise_sweep()
    view_rows = exp_view_count()

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    n_arr = [r[0] for r in noise_rows]
    axes[0].plot(n_arr, [r[1] for r in noise_rows], 'o-')
    axes[0].set_xlabel("合成噪声 (px)")
    axes[0].set_ylabel("fx 相对误差 (%)")
    axes[0].set_title("噪声 vs 焦距误差")
    axes[0].grid(alpha=.3)

    axes[1].plot(n_arr, [r[2] for r in noise_rows], 's-', color='C1')
    axes[1].set_xlabel("合成噪声 (px)")
    axes[1].set_ylabel("主点偏差 (px)")
    axes[1].set_title("噪声 vs 主点误差")
    axes[1].grid(alpha=.3)

    v_arr = [r[0] for r in view_rows]
    axes[2].plot(v_arr, [r[1] for r in view_rows], '^-', color='C2')
    axes[2].set_xlabel("图片数量")
    axes[2].set_ylabel("fx 相对误差 (%)")
    axes[2].set_title("图片数量 vs 焦距误差（噪声 0.3px）")
    axes[2].grid(alpha=.3)

    plt.tight_layout()
    plt.savefig("validate_result.png", dpi=110)
    print("\n-> validate_result.png")
