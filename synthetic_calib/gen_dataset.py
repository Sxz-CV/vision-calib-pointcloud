"""合成标定数据集生成器

核心思路：
    设定相机真值 -> 随机构造位姿 -> 精确投影棋盘 -> 渲染成真实图像 -> 落盘 JSON

做的不是"调焦距生成棋盘格"，而是"调相机参数，看同一块板被拍成什么样"。

用法示例：
    # 默认：640x480, fx=800, 轻微桶形畸变, 20 张
    python gen_dataset.py -o output/demo

    # 长焦 + 强畸变
    python gen_dataset.py -o output/tele --fx 1600 --fy 1600 --k1 -0.3

    # 加噪声和模糊，模拟真实相机
    python gen_dataset.py -o output/noisy --noise 0.8 --blur 1.2 --jpeg 85

    # 高清
    python gen_dataset.py -o output/hd --img-w 1920 --img-h 1080 --fx 1500

输出：
    output/demo/
    ├─ img_000.png ... img_019.png     合成图像
    ├─ corners_000.png ...             带角点的调试图（可选 --preview）
    ├─ dataset.json                    真值 + 带噪声的检测角点
    └─ board.png                       使用的标定板贴图
"""
import argparse
import json
import os

import cv2
import numpy as np


# ---------------------------------------------------------------- 标定板贴图

def make_board_texture(cols, rows, square_mm, px_per_mm, margin_mm=8.0):
    """生成高分辨率棋盘格贴图。返回 (贴图, 板宽mm, 板高mm, 原点偏移mm)

    原点偏移：贴图左上角对应的板坐标。棋盘第一格左上角在 (margin, margin)。
    """
    bw = cols * square_mm
    bh = rows * square_mm
    W = int(round((bw + 2 * margin_mm) * px_per_mm))
    H = int(round((bh + 2 * margin_mm) * px_per_mm))

    tex = np.full((H, W), 235, np.uint8)          # 白底
    s = square_mm * px_per_mm
    for r in range(rows):
        for c in range(cols):
            if (r + c) % 2 == 0:
                x0 = int(round((margin_mm + c * square_mm) * px_per_mm))
                y0 = int(round((margin_mm + r * square_mm) * px_per_mm))
                x1 = int(round(x0 + s))
                y1 = int(round(y0 + s))
                tex[y0:y1, x0:x1] = 20

    return tex, bw, bh, margin_mm


# ---------------------------------------------------------------- 位姿

def sample_pose(rng, bw, bh, K, max_tilt_deg, dist_mm_range, img_w, img_h,
                margin_px=6):
    """采样一个让整块板落在画面内的位姿。返回 (rvec, tvec) 或 None"""
    for _ in range(200):
        ang = np.radians(rng.uniform(-max_tilt_deg, max_tilt_deg, 3))
        Rx = cv2.Rodrigues(np.array([ang[0], 0.0, 0.0]))[0]
        Ry = cv2.Rodrigues(np.array([0.0, ang[1], 0.0]))[0]
        Rz = cv2.Rodrigues(np.array([0.0, 0.0, ang[2]]))[0]
        R = Rz @ Ry @ Rx

        z = rng.uniform(*dist_mm_range)
        # 板中心要大致对准光轴，再随机偏移
        cx_off = rng.uniform(-0.35, 0.35) * z * (img_w / K[0, 0])
        cy_off = rng.uniform(-0.35, 0.35) * z * (img_h / K[1, 1])
        t = np.array([cx_off, cy_off, z], float).reshape(3, 1)
        rvec = cv2.Rodrigues(R)[0]

        corners, _ = cv2.projectPoints(
            np.array([[0, 0, 0], [bw, 0, 0], [bw, bh, 0], [0, bh, 0]], float),
            rvec, t, K, np.zeros(5))
        p = corners.reshape(-1, 2)
        if (p[:, 0].min() > margin_px and p[:, 0].max() < img_w - margin_px and
                p[:, 1].min() > margin_px and p[:, 1].max() < img_h - margin_px):
            return rvec, t
    return None


# ---------------------------------------------------------------- 渲染

def render_view(K, dist, rvec, tvec, bw, bh, tex, px_per_mm, margin_mm,
                img_w, img_h, bg=185, board_px_shift=None):
    """把平面棋盘按给定位姿+畸变渲染成图像

    关键：对输出图像每个像素，先反解畸变得到理想坐标，再用单应逆映射到板坐标，
    最后从贴图采样。这样畸变与透视同时精确。
    """
    # 板 mm 坐标 -> 理想像素坐标 的单应
    obj4 = np.array([[0, 0, 0], [bw, 0, 0], [bw, bh, 0], [0, bh, 0]], float)
    proj4, _ = cv2.projectPoints(obj4, rvec, tvec, K, np.zeros(5))
    H = cv2.getPerspectiveTransform(
        obj4[:, :2].astype(np.float32),
        proj4.reshape(-1, 2).astype(np.float32))
    Hinv = np.linalg.inv(H)

    # 输出像素网格
    xs, ys = np.meshgrid(np.arange(img_w, dtype=np.float32),
                         np.arange(img_h, dtype=np.float32))
    pts = np.stack([xs, ys], axis=-1).reshape(-1, 1, 2)

    # 畸变像素 -> 理想像素（undistortPoints 做的正是这件事）
    ideal = cv2.undistortPoints(pts, K, dist, P=K).reshape(img_h, img_w, 2)

    # 理想像素 -> 板 mm 坐标
    board = cv2.perspectiveTransform(
        ideal.reshape(-1, 1, 2), Hinv).reshape(img_h, img_w, 2)

    # 板 mm -> 贴图像素
    mapx = ((board[:, :, 0] + margin_mm) * px_per_mm).astype(np.float32)
    mapy = ((board[:, :, 1] + margin_mm) * px_per_mm).astype(np.float32)

    img = cv2.remap(tex, mapx, mapy, cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=(bg, bg, bg))
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


# ---------------------------------------------------------------- 退化

def degrade(img, rng, noise_px, blur_sigma, jpeg_q, vignette):
    """噪声 / 模糊 / JPEG / 暗角，模拟真实成像"""
    out = img.astype(np.float32)

    if blur_sigma > 0:
        k = int(max(3, round(blur_sigma * 4)) | 1)
        out = cv2.GaussianBlur(out, (k, k), blur_sigma)

    if vignette > 0:
        h, w = img.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        r = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 +
                    ((yy - h / 2) / (h / 2)) ** 2) / np.sqrt(2)
        mask = 1.0 - vignette * (r ** 2)
        out *= mask[..., None]

    if noise_px > 0:
        out += rng.normal(0, noise_px, out.shape)

    out = np.clip(out, 0, 255).astype(np.uint8)

    if jpeg_q is not None:
        ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, jpeg_q])
        if ok:
            out = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return out


# ---------------------------------------------------------------- 主流程

def parse_args():
    p = argparse.ArgumentParser(description="合成棋盘格标定数据集")
    p.add_argument("-o", "--out", default="output/dataset")

    # 板
    p.add_argument("--cols", type=int, default=9, help="横向格子数")
    p.add_argument("--rows", type=int, default=6, help="纵向格子数")
    p.add_argument("--square", type=float, default=20.0, help="格子边长 mm")
    p.add_argument("--px-per-mm", type=float, default=14.0, help="板贴图分辨率")

    # 相机
    p.add_argument("--img-w", type=int, default=640)
    p.add_argument("--img-h", type=int, default=480)
    p.add_argument("--fx", type=float, default=800.0)
    p.add_argument("--fy", type=float, default=None, help="默认等于 fx")
    p.add_argument("--cx", type=float, default=None, help="默认在图像中心")
    p.add_argument("--cy", type=float, default=None)
    p.add_argument("--k1", type=float, default=-0.15)
    p.add_argument("--k2", type=float, default=0.05)
    p.add_argument("--p1", type=float, default=0.0005)
    p.add_argument("--p2", type=float, default=-0.0005)
    p.add_argument("--k3", type=float, default=0.0)

    # 采样
    p.add_argument("-n", "--n", type=int, default=20, help="图片数量")
    p.add_argument("--max-tilt", type=float, default=45.0, help="最大倾斜角(度)")
    p.add_argument("--dist-mm", type=float, nargs=2, default=None,
                   help="板到相机距离范围，默认按视野自动推")
    p.add_argument("--device-noise", type=float, default=0.0,
                   help="角点检测噪声 px（写进 JSON 的 detected_corners）")

    # 退化
    p.add_argument("--noise", type=float, default=0.0, help="图像高斯噪声 std")
    p.add_argument("--blur", type=float, default=0.0, help="高斯模糊 sigma")
    p.add_argument("--jpeg", type=int, default=None, help="JPEG 质量 1-100")
    p.add_argument("--vignette", type=float, default=0.0, help="暗角强度 0-1")

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--preview", action="store_true", help="输出带角点的调试图")
    return p.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out, exist_ok=True)

    fy = args.fy if args.fy is not None else args.fx
    cx = args.cx if args.cx is not None else args.img_w / 2.0
    cy = args.cy if args.cy is not None else args.img_h / 2.0
    K = np.array([[args.fx, 0, cx], [0, fy, cy], [0, 0, 1]], float)
    dist = np.array([args.k1, args.k2, args.p1, args.p2, args.k3], float)

    tex, bw, bh, margin = make_board_texture(
        args.cols, args.rows, args.square, args.px_per_mm)
    cv2.imwrite(os.path.join(args.out, "board.png"), tex)

    # 距离范围：让板的横向尺寸约占画面 40%~85%
    if args.dist_mm:
        dmin, dmax = args.dist_mm
    else:
        d_fit = bw * args.fx / args.img_w
        dmin, dmax = d_fit * 1.25, d_fit * 2.6

    print(f"相机 : {args.img_w}x{args.img_h}  fx={args.fx:.1f} fy={fy:.1f} "
          f"cx={cx:.1f} cy={cy:.1f}")
    print(f"畸变 : k1={args.k1} k2={args.k2} p1={args.p1} p2={args.p2} k3={args.k3}")
    print(f"标定板: {args.cols}x{args.rows} 格 @ {args.square} mm = {bw}x{bh} mm")
    print(f"        OpenCV patternSize 应填 ({args.cols - 1}, {args.rows - 1})")
    print(f"距离 : {dmin:.0f} ~ {dmax:.0f} mm   倾斜 <= {args.max_tilt:.0f}°")
    print(f"计划生成 {args.n} 张\n")

    # 世界坐标下的真实角点（内角点，Z=0）
    grid = np.mgrid[0:args.cols - 1, 0:args.rows - 1].T.reshape(-1, 2)
    obj_corners = np.zeros((len(grid), 3), float)
    obj_corners[:, :2] = (grid + 1) * args.square - args.square  # 内角点位置
    # 内角点位于第 (1..cols-1) 格边界，坐标 = 索引 * square
    obj_corners[:, :2] = grid * args.square

    views = []
    made = 0
    attempts = 0
    while made < args.n and attempts < args.n * 30:
        attempts += 1
        pose = sample_pose(rng, bw, bh, K, args.max_tilt, (dmin, dmax),
                           args.img_w, args.img_h)
        if pose is None:
            continue
        rvec, tvec = pose

        img = render_view(K, dist, rvec, tvec, bw, bh, tex,
                          args.px_per_mm, margin, args.img_w, args.img_h)
        img = degrade(img, rng, args.noise, args.blur, args.jpeg, args.vignette)

        name = f"img_{made:03d}.png"
        cv2.imwrite(os.path.join(args.out, name), img)

        # 真值角点（带畸变投影，无噪声）
        proj, _ = cv2.projectPoints(obj_corners, rvec, tvec, K, dist)
        proj = proj.reshape(-1, 2)

        # 模拟"检测到的"角点 = 真值 + 高斯噪声
        detected = proj + rng.normal(0, args.device_noise, proj.shape)

        views.append({
            "file": name,
            "rvec": rvec.ravel().tolist(),
            "tvec": tvec.ravel().tolist(),
            "corners_true_px": np.round(proj, 4).tolist(),
            "corners_detected_px": np.round(detected, 4).tolist(),
        })

        if args.preview:
            vis = img.copy()
            for p in detected:
                cv2.circle(vis, (int(round(p[0])), int(round(p[1]))), 3,
                           (0, 0, 255), 1)
            cv2.imwrite(os.path.join(args.out, f"preview_{made:03d}.png"), vis)

        made += 1
        if made % 5 == 0 or made == args.n:
            print(f"  已生成 {made}/{args.n}")

    if made < args.n:
        print(f"警告：只生成了 {made} 张（位姿采样失败 {attempts} 次）")

    meta = {
        "generator": "gen_dataset.py",
        "camera_true": {
            "K": K.tolist(),
            "dist": dist.tolist(),
            "fx": args.fx, "fy": fy, "cx": cx, "cy": cy,
            "image_size": [args.img_w, args.img_h],
        },
        "board": {
            "cols_squares": args.cols,
            "rows_squares": args.rows,
            "square_mm": args.square,
            "board_size_mm": [bw, bh],
            "pattern_size": [args.cols - 1, args.rows - 1],
        },
        "sampling": {
            "n_views": made,
            "max_tilt_deg": args.max_tilt,
            "dist_mm_range": [dmin, dmax],
            "seed": args.seed,
        },
        "degradation": {
            "gaussian_noise_std": args.noise,
            "blur_sigma": args.blur,
            "jpeg_quality": args.jpeg,
            "vignette": args.vignette,
        },
        "corner_noise_px": args.device_noise,
        "views": views,
    }
    jp = os.path.join(args.out, "dataset.json")
    with open(jp, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"\n-> {made} 张图 + dataset.json")
    print(f"-> {os.path.abspath(jp)}")
    print(f"\n下一步验证：python check_calib.py -d {args.out}")


if __name__ == "__main__":
    main()
