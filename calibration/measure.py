"""单目毫米测量：从一张"棋盘格 + 待测物"的照片，量出毫米尺寸并给出不确定度

原理（三步，缺一不可）：
    1. 去畸变   —— 用标定参数，把弯的直线拉直（可选，强烈建议）
    2. 单应变换 —— 用棋盘格四角建立 像素 -> 毫米 的映射（DLT，本质是线性最小二乘）
    3. 正视图   —— 把斜着拍的平面还原成 1px = 1mm 的正视图，之后量的都是真长度

为什么不能跳过第 1 步：
    畸变会让直线变弯，边缘定位点偏移几到十几像素。640x480 下 k1=-0.26 时，
    四角能偏 10px 以上，直接进测量结果。

用法：
    # 交互式：鼠标点两点量距离
    python measure.py -i shot.jpg --pattern 9x6 --square 20

    # 带标定参数（推荐）
    python measure.py -i shot.jpg --pattern 9x6 --square 20 --calib calib/output/camera_params.npz

    # 命令行给坐标（无 GUI 也能用）
    python measure.py -i shot.jpg --pattern 9x6 --square 20 --points 300,400 600,410

    # 光出正视图，不测量
    python measure.py -i shot.jpg --pattern 9x6 --square 20 --warp-only

输出：
    正视图 PNG（1 像素 = 1 毫米）+ 测量结果 + 误差分解
"""
import argparse
import os

import cv2
import numpy as np

STATE = {"pts": [], "img": None, "disp": None, "scale": 1.0, "title": ""}


# ------------------------------------------------------------------ 标定参数

def load_calib(path):
    """读 camera_params.npz，返回 (K, dist) 或 (None, None)"""
    if not path:
        return None, None
    if not os.path.exists(path):
        print(f"⚠ 标定文件不存在: {path}，将跳过去畸变")
        return None, None
    d = np.load(path, allow_pickle=True)
    K = d["K"] if "K" in d else d["K_est"]
    dist = d["dist"] if "dist" in d else d["dist_est"]
    print(f"已加载标定: {path}")
    print(f"  fx={K[0,0]:.2f} fy={K[1,1]:.2f} cx={K[0,2]:.2f} cy={K[1,2]:.2f}")
    return K, dist


# ------------------------------------------------------------------ 建坐标系

def find_board(gray, pattern):
    """检测棋盘内角点，返回 (N*2 角点, 每格平均像素)"""
    ok, corners = cv2.findChessboardCorners(
        gray, pattern,
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not ok:
        # 换 SB 检测器兜底
        ok, corners = cv2.findChessboardCornersSB(gray, pattern)
    if not ok:
        return None, None

    # 亚像素精化，窗口按方格尺寸自适应（之前踩过的坑）
    step = np.linalg.norm(corners[0] - corners[1])
    w = max(3, int(round(step / 5.0)))
    w = w if w % 2 else w + 1
    corners = cv2.cornerSubPix(
        gray, corners, (w, w), (-1, -1),
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3))
    return corners.reshape(-1, 2), float(step)


def make_homography(corners, pattern, square_mm):
    """角点 -> 板毫米坐标，用最小二乘解单应

    findHomography 对 4 点就是 DLT；对 >4 点是最小二乘 —— 用全部角点更稳。

    注意：cv2.findChessboardCorners 返回的角点顺序不固定，
    可能从右下开始向左向上（x 递减、y 递减）。必须先探测实际方向，
    否则正视图会转 90°/180°。
    """
    cols, rows = pattern
    n = len(corners)
    idx = np.arange(n)
    gx = idx % cols
    gy = idx // cols

    # 探测行方向（沿 x）与列方向（沿 y）
    v1 = corners[1] - corners[0]                 # 横向相邻
    sx = 1 if v1[0] >= 0 else -1
    sy = 1
    if n >= cols + 1:
        v2 = corners[cols] - corners[0]          # 纵向相邻（跨一整行）
        sy = 1 if v2[1] >= 0 else -1

    if sx < 0:
        gx = cols - 1 - gx
    if sy < 0:
        gy = rows - 1 - gy

    board_mm = np.zeros((n, 2), np.float32)
    board_mm[:, 0] = gx * square_mm
    board_mm[:, 1] = gy * square_mm

    H, mask = cv2.findHomography(corners.astype(np.float32), board_mm,
                                 method=0)  # 0 = 普通最小二乘，不用 RANSAC
    # 单应拟合残差（在毫米域算，物理意义清楚）
    proj = cv2.perspectiveTransform(
        corners.reshape(-1, 1, 2).astype(np.float32), H).reshape(-1, 2)
    resid_mm = np.linalg.norm(proj - board_mm, axis=1)
    return H, resid_mm


# ------------------------------------------------------------------ 正视图

def warp_to_mm(img, H, square_mm, margin_mm=10.0):
    """把整幅图按 1px=1mm 还原成正视图

    画布大小 = 原图四角映射到毫米域后的包围盒，向外留 margin。
    """
    h, w = img.shape[:2]
    corners_px = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    corners_mm = cv2.perspectiveTransform(corners_px, H).reshape(-1, 2)

    x0, y0 = corners_mm.min(axis=0) - margin_mm
    x1, y1 = corners_mm.max(axis=0) + margin_mm
    out_w = int(np.ceil(x1 - x0))
    out_h = int(np.ceil(y1 - y0))

    # 平移，使左上角为 0
    T = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], float)
    H_total = T @ H

    # 输出画布 1px = 1mm
    warped = cv2.warpPerspective(
        img, H_total, (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    return warped, H_total, (x0, y0)


# ------------------------------------------------------------------ 测量

def measure_two_points(warped_mm, p1, p2):
    """在正视图上量两点距离（单位：mm，因为正视图 1px=1mm）"""
    d_px = float(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))
    return d_px


def error_budget(square_mm, step_px, H_resid_mm, corner_sigma_px=0.3,
                 measure_span_mm=None):
    """误差预算：把各项拆开，这才是"能报精度"的前提"""
    px_per_mm = step_px / square_mm          # 每毫米多少像素
    mm_per_px = 1.0 / px_per_mm              # 像素当量

    # ① 两端点定位误差（两个端点各含 sigma，RSS 合成）
    edge_mm = corner_sigma_px * mm_per_px * np.sqrt(2)

    # ② 单应拟合残差（在毫米域，已包含角点误差和模型误差）
    homography_mm = float(np.sqrt(np.mean(H_resid_mm ** 2)))

    # ③ 合成
    total = float(np.sqrt(edge_mm ** 2 + homography_mm ** 2))

    return {
        "px_per_mm": px_per_mm,
        "mm_per_px": mm_per_px,
        "edge_mm": edge_mm,
        "homography_mm": homography_mm,
        "total_mm": total,
        "corner_sigma_px": corner_sigma_px,
    }


# ------------------------------------------------------------------ 交互

def interactive_measure(warped, title, max_w=1200):
    """在正视图上点两点量距离。返回 [(p1,p2), ...]"""
    h, w = warped.shape[:2]
    scale = min(1.0, max_w / max(w, h))
    disp0 = cv2.resize(warped, None, fx=scale, fy=scale,
                       interpolation=cv2.INTER_AREA)

    STATE["pts"] = []
    results = []

    def on_mouse(event, x, y, flags, _):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        # 换算回正视图坐标（即毫米坐标）
        px = int(round(x / scale))
        py = int(round(y / scale))
        STATE["pts"].append((px, py))
        if len(STATE["pts"]) == 2:
            p1, p2 = STATE["pts"]
            d = measure_two_points(warped, p1, p2)
            results.append((d, p1, p2))
            print(f"  测量 #{len(results)}: ({p1[0]},{p1[1]}) -> "
                  f"({p2[0]},{p2[1]}) = {d:.2f} mm")
            STATE["pts"] = []

    win = f"{title}  (left-click 2 points | r=reset | q=quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, disp0.shape[1], disp0.shape[0])
    cv2.setMouseCallback(win, on_mouse)

    while True:
        disp = cv2.resize(warped, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_AREA).copy()

        # 已完成的测量
        for d, p1, p2 in results[-6:]:
            a = (int(p1[0] * scale), int(p1[1] * scale))
            b = (int(p2[0] * scale), int(p2[1] * scale))
            cv2.line(disp, a, b, (0, 255, 0), 2)
            mid = ((a[0] + b[0]) // 2, (a[1] + b[1]) // 2)
            cv2.putText(disp, f"{d:.1f}mm", (mid[0] + 5, mid[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 当前正在点的第一个点
        if len(STATE["pts"]) == 1:
            p = STATE["pts"][0]
            cv2.circle(disp, (int(p[0] * scale), int(p[1] * scale)),
                       5, (0, 0, 255), -1)

        cv2.putText(disp, f"1px = 1mm   canvas {w}x{h}",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow(win, disp)

        k = cv2.waitKey(20) & 0xFF
        if k in (ord("q"), 27):
            break
        if k == ord("r"):
            results.clear()
            STATE["pts"] = []

    cv2.destroyAllWindows()
    return results


# ------------------------------------------------------------------ 主流程

def parse_args():
    p = argparse.ArgumentParser(description="单目毫米测量")
    p.add_argument("-i", "--image", required=True)
    p.add_argument("--pattern", default="9x6", help="棋盘内角点 列x行")
    p.add_argument("--square", type=float, required=True, help="方格实际边长 mm")
    p.add_argument("--calib", default="", help="camera_params.npz 路径")
    p.add_argument("--points", nargs=4, type=float, default=None,
                   metavar=("X1", "Y1", "X2", "Y2"),
                   help="原图像素坐标，非交互模式")
    p.add_argument("--warp-only", action="store_true", help="只出正视图")
    p.add_argument("--verify", action="store_true",
                   help="自动用棋盘格已知距离自检（不需要手动点坐标）")
    p.add_argument("--out", default="measure_out")
    p.add_argument("--corner-sigma", type=float, default=0.3,
                   help="角点定位标准差 px（经验 0.2~0.5）")
    return p.parse_args()


def main():
    args = parse_args()
    cols, rows = (int(v) for v in args.pattern.lower().split("x"))
    pattern = (cols, rows)
    os.makedirs(args.out, exist_ok=True)

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"读不到图: {args.image}")
    print(f"图像 {img.shape[1]}x{img.shape[0]}  "
          f"棋盘 {cols}x{rows} 内角点  方格 {args.square} mm\n")

    # ① 去畸变
    K, dist = load_calib(args.calib)
    if K is not None:
        img_u = cv2.undistort(img, K, dist)
        print("已去畸变\n")
    else:
        img_u = img
        print("⚠ 未去畸变，边缘区域会有系统误差\n")

    gray = cv2.cvtColor(img_u, cv2.COLOR_BGR2GRAY)

    # ② 找板
    corners, step_px = find_board(gray, pattern)
    if corners is None:
        raise SystemExit(f"未检测到 {cols}x{rows} 棋盘角点，检查 pattern 是否正确")
    print(f"检测到 {len(corners)} 个角点，方格约 {step_px:.1f} px")

    # ③ 单应
    H, resid_mm = make_homography(corners, pattern, args.square)
    print(f"单应拟合残差：均值 {resid_mm.mean():.4f} mm  "
          f"最大 {resid_mm.max():.4f} mm")

    # ④ 误差预算
    eb = error_budget(args.square, step_px, resid_mm,
                      corner_sigma_px=args.corner_sigma)
    print(f"\n{'='*60}")
    print("误差预算")
    print(f"{'='*60}")
    print(f"  像素当量      {eb['mm_per_px']:.4f} mm/px  "
          f"({eb['px_per_mm']:.2f} px/mm)")
    print(f"  端点定位      ±{eb['edge_mm']:.4f} mm  "
          f"(假设角点 {args.corner_sigma} px, 两端 RSS)")
    print(f"  单应拟合      ±{eb['homography_mm']:.4f} mm")
    print(f"  {'-'*56}")
    print(f"  合成(1σ)     ±{eb['total_mm']:.4f} mm")
    print(f"  95% 置信     ±{eb['total_mm']*1.96:.4f} mm")

    # ⑤ 正视图
    warped, H_total, origin_mm = warp_to_mm(img_u, H, args.square)
    pw = os.path.join(args.out, "warped_1mm_per_px.png")
    cv2.imwrite(pw, warped)
    print(f"\n正视图（1px = 1mm）: {pw}   尺寸 {warped.shape[1]}x{warped.shape[0]} mm")

    # 标尺叠加，方便肉眼核对
    ruler = warped.copy()
    for x in range(0, warped.shape[1], 10):
        c = (0, 0, 255) if x % 50 == 0 else (120, 120, 120)
        cv2.line(ruler, (x, 0), (x, 12 if x % 50 else 20), c, 1)
        cv2.line(ruler, (x, warped.shape[0] - 1),
                 (x, warped.shape[0] - (13 if x % 50 else 21)), c, 1)
    for y in range(0, warped.shape[0], 10):
        c = (0, 0, 255) if y % 50 == 0 else (120, 120, 120)
        cv2.line(ruler, (0, y), (12 if y % 50 else 20, y), c, 1)
        cv2.line(ruler, (warped.shape[1] - 1, y),
                 (warped.shape[1] - (13 if y % 50 else 21), y), c, 1)
    pr = os.path.join(args.out, "warped_with_ruler.png")
    cv2.imwrite(pr, ruler)
    print(f"带标尺版: {pr}")

    if args.warp_only:
        return

    # 自检模式：用棋盘格本身的已知间距当标尺
    if args.verify:
        n = len(corners)
        idx = np.arange(n)
        gx = idx % cols
        gy = idx // cols
        # 找出四个极值角点的索引
        i_00 = int(np.argmin(gx + gy * 1000))          # (0,0)
        i_x1 = int(np.argmax(gx * 1000 - gy))          # (cols-1, 0)
        i_y1 = int(np.argmax(gy * 1000 - gx))          # (0, rows-1)
        i_xy = int(np.argmax(gx + gy))                 # (cols-1, rows-1)

        tests = [
            (f"横向 {cols-1} 个间距", corners[i_00], corners[i_x1], (cols - 1) * args.square),
            (f"纵向 {rows-1} 个间距", corners[i_00], corners[i_y1], (rows - 1) * args.square),
            ("相邻 1 格", corners[i_00], corners[int(np.argmin(np.abs(gx-1) + gy*1000))], args.square),
            ("对角线", corners[i_00], corners[i_xy],
             float(np.hypot((cols - 1) * args.square, (rows - 1) * args.square))),
        ]

        print(f"\n{'='*62}")
        print("自检：用棋盘格已知间距当标尺")
        print(f"{'='*62}")
        print(f"{'测量对象':<18}{'真值(mm)':>10}{'实测(mm)':>11}{'误差(mm)':>11}")
        print("-" * 62)
        errs = []
        for name, p1, p2, truth in tests:
            a = cv2.perspectiveTransform(
                np.array([[p1, p2]], np.float32), H_total).reshape(-1, 2)
            v = float(np.linalg.norm(a[1] - a[0]))
            e = v - truth
            errs.append(e)
            print(f"{name:<18}{truth:>10.2f}{v:>11.2f}{e:>+11.3f}")
        errs = np.array(errs)
        print("-" * 62)
        print(f"RMS 误差 {np.sqrt(np.mean(errs**2)):.4f} mm   "
              f"最大 {np.abs(errs).max():.4f} mm   "
              f"均值 {errs.mean():+.4f} mm")
        print(f"误差预算预测 1σ = ±{eb['total_mm']:.4f} mm")
        if abs(errs.mean()) > eb["total_mm"]:
            print(f"⚠ 平均误差 {errs.mean():+.4f} mm 超过预测的随机误差，"
                  f"说明存在系统偏差（方格边长不准/打印缩放/角点系统偏移）")
        else:
            print("✓ 误差在预测范围内")
        return

    # ⑥ 测量
    results = []
    if args.points:
        x1, y1, x2, y2 = args.points
        # 原图像素 -> 正视图毫米坐标
        src = np.array([[[x1, y1]], [[x2, y2]]], np.float32)
        dst = cv2.perspectiveTransform(src, H_total).reshape(-1, 2)
        d = measure_two_points(warped, dst[0], dst[1])
        results.append((d, tuple(dst[0]), tuple(dst[1])))
        print(f"\n测量: ({x1:.0f},{y1:.0f}) -> ({x2:.0f},{y2:.0f}) = {d:.2f} mm")
    else:
        print("\n进入交互模式：在正视图窗口点两点量距离，r 重置，q 退出")
        results = interactive_measure(warped, "measure")

    if not results:
        print("没有测量结果")
        return

    print(f"\n{'='*60}")
    print("测量汇总")
    print(f"{'='*60}")
    print(f"{'#':>3} {'距离(mm)':>12} {'±1σ':>10} {'±95%':>10}")
    for i, (d, p1, p2) in enumerate(results, 1):
        print(f"{i:>3} {d:>12.2f} {eb['total_mm']:>10.4f} "
              f"{eb['total_mm']*1.96:>10.4f}")

    if len(results) >= 2:
        ds = np.array([r[0] for r in results])
        print(f"\n重复性（同一批测量的标准差）: {ds.std(ddof=1):.4f} mm")
        print("  ↑ 这个数和上面的'合成误差'应该同一量级；"
              "差很多说明还有未建模的误差源")

    # 保存带测量的图
    fin = warped.copy()
    for d, p1, p2 in results:
        a = (int(p1[0]), int(p1[1]))
        b = (int(p2[0]), int(p2[1]))
        cv2.line(fin, a, b, (0, 255, 0), 2)
        cv2.circle(fin, a, 4, (0, 0, 255), -1)
        cv2.circle(fin, b, 4, (0, 0, 255), -1)
        mid = ((a[0] + b[0]) // 2, (a[1] + b[1]) // 2)
        cv2.putText(fin, f"{d:.2f}mm", (mid[0] + 6, mid[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    pm = os.path.join(args.out, "measured.png")
    cv2.imwrite(pm, fin)
    print(f"\n结果图: {pm}")

    # 写报告
    rp = os.path.join(args.out, "report.md")
    with open(rp, "w", encoding="utf-8") as f:
        f.write(f"# 测量报告\n\n")
        f.write(f"- 图像: `{os.path.basename(args.image)}`\n")
        f.write(f"- 标定: `{args.calib or '未使用（未去畸变）'}`\n")
        f.write(f"- 棋盘: {cols}x{rows} 内角点，方格 {args.square} mm\n")
        f.write(f"- 像素当量: {eb['mm_per_px']:.4f} mm/px\n")
        f.write(f"- 单应残差: {resid_mm.mean():.4f} mm (均值), "
                f"{resid_mm.max():.4f} mm (最大)\n\n")
        f.write(f"## 误差预算\n\n")
        f.write(f"| 项 | 值 (mm) |\n|---|---|\n")
        f.write(f"| 端点定位 (σ={args.corner_sigma} px) | {eb['edge_mm']:.4f} |\n")
        f.write(f"| 单应拟合残差 | {eb['homography_mm']:.4f} |\n")
        f.write(f"| **合成 (1σ)** | **{eb['total_mm']:.4f}** |\n")
        f.write(f"| 95% 置信 | {eb['total_mm']*1.96:.4f} |\n\n")
        f.write(f"## 测量结果\n\n")
        f.write(f"| # | 距离 (mm) | ±1σ |\n|---|---|---|\n")
        for i, (d, p1, p2) in enumerate(results, 1):
            f.write(f"| {i} | {d:.2f} | {eb['total_mm']:.4f} |\n")
        if len(results) >= 2:
            ds = np.array([r[0] for r in results])
            f.write(f"\n重复性 (n={len(ds)}, std) = {ds.std(ddof=1):.4f} mm\n")
    print(f"报告: {rp}")


if __name__ == "__main__":
    main()
