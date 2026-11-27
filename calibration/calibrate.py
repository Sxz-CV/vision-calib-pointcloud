"""相机内参 + 畸变标定（OpenCV 棋盘格）

用法：
    python calibrate.py --dir calib --pattern 9x6 --square 25

参数：
    --pattern  棋盘内角点数，格式 列x行（9x6 表示横向 9 个内角点、纵向 6 个）
               注意：内角点数 = 格子数 - 1
    --square   方格边长（mm），只影响外参尺度，不影响内参

输出：
    重投影误差、内参 K、畸变系数、每张图的残差、残差场可视化
"""
import argparse
import glob
import os

import cv2
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="calib")
    p.add_argument("--pattern", default="9x6", help="内角点数 列x行")
    p.add_argument("--square", type=float, default=25.0, help="方格边长 mm")
    p.add_argument("--subpix", type=int, default=0,
                   help="角点亚像素精化窗口（奇数）。0 = 自动按方格像素尺寸取 1/5")
    p.add_argument("--out", default="", help="输出目录，默认同 --dir 的上一级")
    p.add_argument("--exclude", default="", help="排除的图片，逗号分隔，如 left02.jpg,left13.jpg")
    return p.parse_args()


def main():
    args = parse_args()
    cols, rows = (int(v) for v in args.pattern.lower().split("x"))
    pattern = (cols, rows)

    out_dir = args.out or os.path.join(os.path.abspath(args.dir), "output")
    os.makedirs(out_dir, exist_ok=True)

    # 世界坐标系下的棋盘角点（Z=0 平面）
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * args.square

    # 只收集输入图片；排除自己生成的输出（避免把残差图当成标定图读回来）
    files = sorted(glob.glob(os.path.join(args.dir, "*.jpg")) +
                   glob.glob(os.path.join(args.dir, "*.png")))
    files = [f for f in files
             if os.path.dirname(os.path.abspath(f)) != os.path.abspath(out_dir)
             and os.path.basename(f) not in
             ("residual_field.png", "undistort_compare.png")]
    # 手动排除指定图片（例如残差偏大的坏帧）
    exclude = {s.strip() for s in args.exclude.split(",") if s.strip()}
    if exclude:
        before = len(files)
        files = [f for f in files if os.path.basename(f) not in exclude]
        print(f"已排除 {before - len(files)} 张：{sorted(exclude)}")

    if not files:
        raise SystemExit(f"目录里没有图片: {args.dir}")

    print(f"棋盘内角点 {cols}x{rows}，方格 {args.square} mm，找到 {len(files)} 张图\n")
    print(f"输出目录: {out_dir}\n")

    objpoints, imgpoints = [], []
    used, failed = [], []
    img_size = None
    subpix_reported = False

    for f in files:
        img = cv2.imread(f)
        if img is None:
            failed.append((os.path.basename(f), "读取失败"))
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 第一张图锁定尺寸，后续尺寸不一致的直接跳过
        if img_size is None:
            img_size = gray.shape[::-1]
        elif gray.shape[::-1] != img_size:
            failed.append((os.path.basename(f),
                           f"尺寸不符 {gray.shape[1]}x{gray.shape[0]}"))
            continue
        ok, corners = cv2.findChessboardCorners(
            gray, pattern,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
        if not ok:
            failed.append((os.path.basename(f), "未找到角点"))
            continue
        # 亚像素精化：窗口大小必须匹配方格尺寸。
        # 实测：方格约 25px 时，窗口 5 给 RMS 0.055；窗口 11 给 RMS 1.93（坍塌）。
        # 经验规则：winSize ≈ 方格边长像素数 / 5。
        step = np.linalg.norm(corners[0] - corners[1])
        win = args.subpix if args.subpix > 0 else max(3, int(round(step / 5.0)))
        win = win if win % 2 == 1 else win + 1
        if not subpix_reported:
            print(f"方格约 {step:.1f} px → 亚像素窗口取 ({win},{win})")
            subpix_reported = True
        corners = cv2.cornerSubPix(
            gray, corners, (win, win), (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3))
        objpoints.append(objp)
        imgpoints.append(corners)
        used.append(os.path.basename(f))

    print(f"成功检角点 {len(used)} 张：{used}")
    if failed:
        print(f"失败 {len(failed)} 张：{failed}")
    if len(used) < 3:
        raise SystemExit("可用图片不足 3 张，无法标定")

    print("\n开始标定…")
    # 默认只用 5 参数模型。不要因为图多就自动上有理模型：
    # 实测 31 张小棋盘图（5x4）用 RATIONAL_MODEL 会崩成 k1=-30、k5=1175。
    flags = 0
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_size, None, None, flags=flags)

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    print("\n" + "=" * 62)
    print(f"重投影误差 RMS = {ret:.4f} px")
    print("=" * 62)
    print(f"内参 K =\n{np.array2string(K, precision=3)}")
    print(f"\n  fx = {fx:.2f}   fy = {fy:.2f}   (应接近)")
    print(f"  cx = {cx:.2f}   cy = {cy:.2f}   图像中心 = ({img_size[0]/2:.1f}, {img_size[1]/2:.1f})")
    print(f"  fx/fy = {fx/fy:.4f}  (偏离 1 越多越可疑)")
    print(f"  主点偏移 = ({cx - img_size[0]/2:+.1f}, {cy - img_size[1]/2:+.1f}) px")
    print(f"\n畸变系数 = {np.array2string(dist.ravel()[:5], precision=4)}")
    k1 = float(dist.ravel()[0])
    if k1 < 0:
        print("  k1 < 0 → 桶形畸变（常见）")
    else:
        print("  k1 > 0 → 枕形畸变")
    if abs(k1) > 1.0:
        print(f"  ⚠ k1 = {k1:.2f} 过大（正常 |k1| < 1），标定可能崩溃："
              "先查是否模型参数过多、图太少或角点误检")

    # 姿态多样性：姿态太单一时低 RMS 不可信
    Rs = [cv2.Rodrigues(r)[0] for r in rvecs]
    tx = [np.degrees(np.arctan2(R[2, 1], R[2, 2])) for R in Rs]
    ty = [np.degrees(np.arctan2(-R[2, 0], np.hypot(R[2, 1], R[2, 2]))) for R in Rs]
    tz = [np.degrees(np.arctan2(R[1, 0], R[0, 0])) for R in Rs]
    span = (max(tx) - min(tx), max(ty) - min(ty), max(tz) - min(tz))
    print(f"\n姿态跨度 X/Y/Z = {span[0]:.0f}°/{span[1]:.0f}°/{span[2]:.0f}°")
    if max(span) < 30:
        print("  ⚠ 姿态变化太小，标定参数外推不可信")
    elif min(span) > 40:
        print("  姿态多样性良好")

    # 每张图的重投影误差
    print("\n" + "=" * 62)
    print("每张图的单独重投影误差")
    print("=" * 62)
    per_img = []
    for i, (op, ip, rv, tv) in enumerate(zip(objpoints, imgpoints, rvecs, tvecs)):
        proj, _ = cv2.projectPoints(op, rv, tv, K, dist)
        err = np.linalg.norm(ip.reshape(-1, 2) - proj.reshape(-1, 2), axis=1)
        per_img.append((used[i], err.mean(), err.max()))
        flag = "  <-- 偏高" if err.mean() > ret * 2 else ""
        print(f"  {used[i]:15s} 均值 {err.mean():.4f} 最大 {err.max():.4f} px{flag}")

    # 残差场可视化
    print("\n绘制残差场…")
    n = min(len(used), 4)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, i in zip(axes, range(n)):
        img = cv2.imread(os.path.join(args.dir, used[i]))
        proj, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)
        p0 = imgpoints[i].reshape(-1, 2)
        p1 = proj.reshape(-1, 2)
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        scale = 40.0   # 箭头放大倍数，残差太小看不见
        for a, b in zip(p0, p1):
            ax.annotate("", xy=(a[0] + (a[0] - b[0]) * scale, a[1] + (a[1] - b[1]) * scale),
                        xytext=(a[0], a[1]),
                        arrowprops=dict(arrowstyle="->", color="red", lw=0.8))
        ax.set_title(f"{used[i]}\nmean {per_img[i][1]:.3f} px", fontsize=9)
        ax.axis("off")
    fig.suptitle(f"residual field (arrows x{int(scale)})   RMS = {ret:.4f} px", fontsize=12)
    plt.tight_layout()
    p_res = os.path.join(out_dir, "residual_field.png")
    plt.savefig(p_res, dpi=110)
    print("  ->", p_res)

    # 去畸变对比
    img0 = cv2.imread(os.path.join(args.dir, used[0]))
    h, w = img0.shape[:2]
    newK, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))
    undist = cv2.undistort(img0, K, dist, None, newK)
    both = np.hstack([img0, undist])
    for x in range(0, w, 60):
        cv2.line(img0, (x, 0), (x, h), (0, 255, 0), 1)
    both = np.hstack([img0, cv2.resize(undist, (w, h))])
    cv2.putText(both, "original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(both, "undistorted", (w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    p_und = os.path.join(out_dir, "undistort_compare.png")
    cv2.imwrite(p_und, both)
    print("  ->", p_und)

    # 保存参数，后续可直接复用
    np.savez(os.path.join(out_dir, "camera_params.npz"),
             K=K, dist=dist, image_size=img_size, rms=ret)
    print(f"  -> {os.path.join(out_dir, 'camera_params.npz')}")

    # 判据提示
    print("\n" + "=" * 62)
    print("判据")
    print("=" * 62)
    if ret < 0.1:
        print("  RMS < 0.1 px：可疑，可能图片太少/姿态太单一（过拟合）")
    elif ret < 0.5:
        print(f"  RMS = {ret:.3f} px：正常范围")
    else:
        print(f"  RMS = {ret:.3f} px：偏高，检查标定板是否平整、是否模糊")
    print(f"  图片数量：{len(used)} 张" + ("（偏少，建议 15+）" if len(used) < 15 else ""))
    print(f"  fx/fy = {fx/fy:.4f}（偏离 1 超过 2% 需检查）")


if __name__ == "__main__":
    main()
