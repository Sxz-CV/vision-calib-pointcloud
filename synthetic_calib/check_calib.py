"""读取合成数据集 JSON，跑标定，和真值对比

它回答的是：给定这批图（及其角点质量），标定算法能把真值恢复得多准。

用法：
    python check_calib.py -d output/demo
    python check_calib.py -d output/demo --use true       # 用无噪声真值角点
    python check_calib.py -d output/demo --detect         # 从图像里实际检测角点

三种角点来源的区别：
    --use detected （默认）  JSON 里带噪声的模拟检测结果 -> 看算法在噪声下的表现
    --use true               无噪声真值 -> 零噪声时应当精确恢复真值
    --detect                 真跑 findChessboardCorners -> 看你的检测流程行不行
"""
import argparse
import json
import os

import cv2
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("-d", "--dir", required=True)
    p.add_argument("--use", choices=["detected", "true"], default="detected")
    p.add_argument("--detect", action="store_true",
                   help="从图像实际检测角点（覆盖 --use）")
    p.add_argument("--flags", default="0",
                   help="cv2.calibrateCamera flags 的整数，例如 0 或 2(FIX_PRINCIPAL_POINT)")
    p.add_argument("--json", default=None, help="手动指定 json 路径")
    return p.parse_args()


def err_pct(est, true):
    return (est - true) / true * 100.0


def main():
    args = parse_args()
    jp = args.json or os.path.join(args.dir, "dataset.json")
    with open(jp, encoding="utf-8") as f:
        meta = json.load(f)

    cam = meta["camera_true"]
    K_true = np.array(cam["K"], float)
    d_true = np.array(cam["dist"], float)
    img_w, img_h = cam["image_size"]
    board = meta["board"]
    sq = board["square_mm"]
    cols_sq, rows_sq = board["cols_squares"], board["rows_squares"]
    nc, nr = cols_sq - 1, rows_sq - 1

    # 世界坐标内角点：位于 (i*sq, j*sq, 0)
    grid = np.mgrid[0:nc, 0:nr].T.reshape(-1, 2).astype(np.float32)
    objp = np.zeros((grid.shape[0], 3), np.float32)
    objp[:, :2] = grid * sq

    obj_list, img_list = [], []
    skipped = 0

    for v in meta["views"]:
        path = os.path.join(args.dir, v["file"])
        if args.detect:
            img = cv2.imread(path)
            g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ok, c = cv2.findChessboardCorners(
                g, (nc, nr),
                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
            if not ok:
                skipped += 1
                continue
            step = np.linalg.norm(c[0] - c[1])
            w = max(3, int(round(step / 5.0)))
            w = w if w % 2 else w + 1
            c = cv2.cornerSubPix(g, c, (w, w), (-1, -1),
                                 (cv2.TERM_CRITERIA_EPS +
                                  cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3))
            img_list.append(c)
        else:
            key = ("corners_true_px" if args.use == "true"
                   else "corners_detected_px")
            c = np.array(v[key], np.float32).reshape(-1, 1, 2)
            img_list.append(c)
        obj_list.append(objp)

    if not obj_list:
        raise SystemExit("没有可用视图")

    src = "图像检测" if args.detect else (
        "真值角点" if args.use == "true" else "带噪检测角点")
    print(f"角点来源: {src}    可用视图: {len(obj_list)}/{len(meta['views'])}"
          + (f"（跳过 {skipped}）" if skipped else ""))
    print(f"内角点 {nc}x{nr}  方格 {sq} mm\n")

    flags = int(args.flags)
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_list, img_list, (img_w, img_h), None, None, flags=flags)

    print("=" * 74)
    print(f"重投影误差 RMS = {ret:.5f} px")
    print("=" * 74)

    print(f"\n{'':10s}{'估计值':>14s}{'真值':>14s}{'误差':>16s}")
    print("-" * 74)
    rows = [("fx", K[0, 0], K_true[0, 0]),
            ("fy", K[1, 1], K_true[1, 1]),
            ("cx", K[0, 2], K_true[0, 2]),
            ("cy", K[1, 2], K_true[1, 2])]
    for name, est, tru in rows:
        unit = "%" if name in ("fx", "fy") else " px"
        val = err_pct(est, tru) if name in ("fx", "fy") else (est - tru)
        print(f"{name:10s}{est:14.4f}{tru:14.4f}{val:+15.4f}{unit}")

    print("-" * 74)
    dnames = ["k1", "k2", "p1", "p2", "k3"]
    for i, nm in enumerate(dnames):
        est = dist.ravel()[i] if i < dist.size else 0.0
        tru = d_true[i]
        print(f"{nm:10s}{est:14.6f}{tru:14.6f}{est - tru:+15.6f}")

    print("-" * 74)
    fx_err = abs(err_pct(K[0, 0], K_true[0, 0]))
    cx_err = abs(K[0, 2] - K_true[0, 2])
    cy_err = abs(K[1, 2] - K_true[1, 2])

    print("\n判据：")
    print(f"  焦距误差 {fx_err:.3f}%"
          + ("  ✓ 好" if fx_err < 1 else "  ✗ 偏大"))
    print(f"  主点偏差 ({cx_err:.2f}, {cy_err:.2f}) px"
          + ("  ✓ 好" if max(cx_err, cy_err) < 3 else "  ✗ 偏大"))

    # 每张图重投影误差
    print("\n每张图重投影误差（均值 / 最大）：")
    worst = []
    for i, (op, ip, rv, tv) in enumerate(zip(obj_list, img_list, rvecs, tvecs)):
        proj, _ = cv2.projectPoints(op, rv, tv, K, dist)
        e = np.linalg.norm(ip.reshape(-1, 2) - proj.reshape(-1, 2), axis=1)
        worst.append((meta["views"][i]["file"], e.mean(), e.max()))
    for f, m, mx in worst:
        flag = "  <-- 偏高" if m > ret * 2.5 else ""
        print(f"  {f:16s} 均值 {m:7.4f}  最大 {mx:8.4f}{flag}")

    print("\n说明：")
    if args.use == "true" and args.flags == "0":
        print("  用真值角点时应精确恢复真值（误差≈0）。不为 0 说明算法/参数有问题。")
    print(f"  真值角点来自几何投影，可视为无噪声；检测噪声 = "
          f"{meta.get('corner_noise_px', 0)} px")


if __name__ == "__main__":
    main()
