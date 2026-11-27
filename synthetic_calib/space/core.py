"""合成标定数据集核心逻辑（不依赖 GUI，可单独测试）

把 gen_dataset.py 和 check_calib.py 的能力收敛成函数，
供 app.py（Gradio）调用，也便于单元测试。
"""
import numpy as np
import cv2


# ------------------------------------------------------------------ 标定板

def make_board_texture(cols, rows, square_mm, px_per_mm, margin_mm=8.0):
    """生成棋盘格贴图。返回 (tex, board_w_mm, board_h_mm)"""
    bw = cols * square_mm
    bh = rows * square_mm
    W = int(round((bw + 2 * margin_mm) * px_per_mm))
    H = int(round((bh + 2 * margin_mm) * px_per_mm))

    tex = np.full((H, W), 235, np.uint8)
    s = square_mm * px_per_mm
    for r in range(rows):
        for c in range(cols):
            if (r + c) % 2 == 0:
                x0 = int(round((margin_mm + c * square_mm) * px_per_mm))
                y0 = int(round((margin_mm + r * square_mm) * px_per_mm))
                tex[y0:int(round(y0 + s)), x0:int(round(x0 + s))] = 20
    return tex, bw, bh


# ------------------------------------------------------------------ 位姿采样

def sample_pose(rng, bw, bh, K, max_tilt_deg, dist_range, img_w, img_h,
                margin_px=6, max_tries=200):
    """采样一个让整块板落在画面内的位姿"""
    for _ in range(max_tries):
        ang = np.radians(rng.uniform(-max_tilt_deg, max_tilt_deg, 3))
        Rx = cv2.Rodrigues(np.array([ang[0], 0.0, 0.0]))[0]
        Ry = cv2.Rodrigues(np.array([0.0, ang[1], 0.0]))[0]
        Rz = cv2.Rodrigues(np.array([0.0, 0.0, ang[2]]))[0]
        R = Rz @ Ry @ Rx

        z = rng.uniform(*dist_range)
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


# ------------------------------------------------------------------ 渲染

def render_view(K, dist, rvec, tvec, bw, bh, tex, px_per_mm, margin_mm,
                img_w, img_h, bg=185):
    """按给定位姿+畸变渲染平面棋盘

    对每个输出像素：反解畸变 -> 理想像素 -> 单应逆变换 -> 板坐标 -> 采样贴图
    """
    obj4 = np.array([[0, 0, 0], [bw, 0, 0], [bw, bh, 0], [0, bh, 0]], float)
    proj4, _ = cv2.projectPoints(obj4, rvec, tvec, K, np.zeros(5))
    H = cv2.getPerspectiveTransform(
        obj4[:, :2].astype(np.float32),
        proj4.reshape(-1, 2).astype(np.float32))
    Hinv = np.linalg.inv(H)

    xs, ys = np.meshgrid(np.arange(img_w, dtype=np.float32),
                         np.arange(img_h, dtype=np.float32))
    pts = np.stack([xs, ys], axis=-1).reshape(-1, 1, 2)

    ideal = cv2.undistortPoints(pts, K, dist, P=K).reshape(img_h, img_w, 2)
    board = cv2.perspectiveTransform(
        ideal.reshape(-1, 1, 2), Hinv).reshape(img_h, img_w, 2)

    mapx = ((board[:, :, 0] + margin_mm) * px_per_mm).astype(np.float32)
    mapy = ((board[:, :, 1] + margin_mm) * px_per_mm).astype(np.float32)

    img = cv2.remap(tex, mapx, mapy, cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=(bg, bg, bg))
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def degrade(img, rng, noise, blur, jpeg_q, vignette):
    """噪声 / 模糊 / JPEG / 暗角"""
    out = img.astype(np.float32)
    if blur > 0:
        k = int(max(3, round(blur * 4)) | 1)
        out = cv2.GaussianBlur(out, (k, k), blur)
    if vignette > 0:
        h, w = img.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        r = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 +
                    ((yy - h / 2) / (h / 2)) ** 2) / np.sqrt(2)
        out *= (1.0 - vignette * (r ** 2))[..., None]
    if noise > 0:
        out += rng.normal(0, noise, out.shape)
    out = np.clip(out, 0, 255).astype(np.uint8)
    if jpeg_q is not None:
        ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_q)])
        if ok:
            out = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return out


# ------------------------------------------------------------------ 主流程

def generate_dataset(img_w=640, img_h=480, fx=800.0, fy=None, k1=-0.15,
                     k2=0.05, p1=0.0005, p2=-0.0005, k3=0.0,
                     cols_sq=9, rows_sq=6, square_mm=20.0,
                     n_views=20, max_tilt=45.0, device_noise=0.5,
                     noise=0.8, blur=1.0, jpeg=90, vignette=0.2,
                     seed=0, px_per_mm=14.0, keep_images=True):
    """生成数据集。返回 dict：
        images:   [ndarray] 生成的图像
        views:    [{corners_true, corners_detected, rvec, tvec}]
        truth:    {K, dist, image_size, pattern_size, square_mm}
    """
    rng = np.random.default_rng(int(seed))
    fy = float(fx) if fy is None else float(fy)
    cx, cy = img_w / 2.0, img_h / 2.0
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], float)
    dist = np.array([k1, k2, p1, p2, k3], float)

    cols_sq, rows_sq = int(cols_sq), int(rows_sq)
    tex, bw, bh = make_board_texture(cols_sq, rows_sq, square_mm, px_per_mm)

    d_fit = bw * fx / img_w
    dmin, dmax = d_fit * 1.25, d_fit * 2.6

    nc, nr = cols_sq - 1, rows_sq - 1
    grid = np.mgrid[0:nc, 0:nr].T.reshape(-1, 2).astype(np.float64)
    obj_corners = np.zeros((grid.shape[0], 3), float)
    obj_corners[:, :2] = grid * square_mm

    images, views = [], []
    made, attempts = 0, 0
    max_attempts = max(60, n_views * 30)

    while made < n_views and attempts < max_attempts:
        attempts += 1
        pose = sample_pose(rng, bw, bh, K, max_tilt, (dmin, dmax),
                           img_w, img_h)
        if pose is None:
            continue
        rvec, tvec = pose

        img = render_view(K, dist, rvec, tvec, bw, bh, tex,
                          px_per_mm, 8.0, img_w, img_h)
        img = degrade(img, rng, noise, blur, jpeg, vignette)

        proj, _ = cv2.projectPoints(obj_corners, rvec, tvec, K, dist)
        proj = proj.reshape(-1, 2)
        detected = proj + rng.normal(0, device_noise, proj.shape)

        images.append(img if keep_images else None)
        views.append({
            "rvec": rvec.ravel().tolist(),
            "tvec": tvec.ravel().tolist(),
            "corners_true": proj,
            "corners_detected": detected,
        })
        made += 1

    return {
        "images": images,
        "views": views,
        "truth": {
            "K": K, "dist": dist,
            "fx": fx, "fy": fy, "cx": cx, "cy": cy,
            "image_size": (img_w, img_h),
            "cols_sq": cols_sq, "rows_sq": rows_sq,
            "pattern_size": (nc, nr),
            "square_mm": square_mm,
            "board_mm": (bw, bh),
            "d_range": (dmin, dmax),
        },
        "requested": n_views,
        "made": made,
    }


def calibrate_from_dataset(ds, source="detected", subpix_on_images=False):
    """跑标定并与真值对比

    source:
        "detected"  用 JSON 式的带噪角点
        "true"      用无噪真值角点
        "detect"    从图像里实际检测（需 keep_images=True）
    """
    truth = ds["truth"]
    nc, nr = truth["pattern_size"]
    sq = truth["square_mm"]
    img_w, img_h = truth["image_size"]

    grid = np.mgrid[0:nc, 0:nr].T.reshape(-1, 2).astype(np.float32)
    objp = np.zeros((grid.shape[0], 3), np.float32)
    objp[:, :2] = grid * sq

    obj_list, img_list = [], []
    skipped = 0
    for i, v in enumerate(ds["views"]):
        if source == "detect":
            img = ds["images"][i]
            if img is None:
                skipped += 1
                continue
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
        elif source == "true":
            img_list.append(np.array(v["corners_true"], np.float32).reshape(-1, 1, 2))
        else:
            img_list.append(np.array(v["corners_detected"], np.float32).reshape(-1, 1, 2))
        obj_list.append(objp)

    if len(obj_list) < 6:
        return None

    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_list, img_list, (img_w, img_h), None, None, flags=0)

    Kt = truth["K"]
    dt = truth["dist"]
    return {
        "rms": float(ret),
        "n_views_used": len(obj_list),
        "skipped": skipped,
        "K_est": K, "dist_est": dist,
        "K_true": Kt, "dist_true": dt,
        "fx_est": float(K[0, 0]), "fx_true": float(Kt[0, 0]),
        "fy_est": float(K[1, 1]), "fy_true": float(Kt[1, 1]),
        "cx_est": float(K[0, 2]), "cx_true": float(Kt[0, 2]),
        "cy_est": float(K[1, 2]), "cy_true": float(Kt[1, 2]),
        "fx_err_pct": float((K[0, 0] - Kt[0, 0]) / Kt[0, 0] * 100),
        "fy_err_pct": float((K[1, 1] - Kt[1, 1]) / Kt[1, 1] * 100),
        "cx_err": float(K[0, 2] - Kt[0, 2]),
        "cy_err": float(K[1, 2] - Kt[1, 2]),
        "k1_est": float(dist.ravel()[0]), "k1_true": float(dt[0]),
    }


def verdict(res):
    """把结果翻译成人话"""
    if res is None:
        return "❌ 可用视图不足（<6），无法标定"
    lines = []
    fx_err = abs(res["fx_err_pct"])
    cx_err = abs(res["cx_err"])
    cy_err = abs(res["cy_err"])

    lines.append(f"重投影误差 RMS = **{res['rms']:.4f} px**  "
                 f"（用 {res['n_views_used']} 张）")

    if fx_err < 0.5:
        lines.append(f"✅ 焦距误差 {fx_err:.3f}%，很好")
    elif fx_err < 1.5:
        lines.append(f"⚠️ 焦距误差 {fx_err:.3f}%，可接受")
    else:
        lines.append(f"❌ 焦距误差 {fx_err:.3f}%，偏大")

    if max(cx_err, cy_err) < 3:
        lines.append(f"✅ 主点偏差 ({cx_err:.2f}, {cy_err:.2f}) px，很好")
    elif max(cx_err, cy_err) < 10:
        lines.append(f"⚠️ 主点偏差 ({cx_err:.2f}, {cy_err:.2f}) px，"
                     f"可接受但偏高")
    else:
        lines.append(f"❌ 主点偏差 ({cx_err:.2f}, {cy_err:.2f}) px，偏大。"
                     f"主点是最难估的量，对噪声极敏感")

    if res["rms"] < 0.05:
        lines.append("ℹ️ RMS 极低（<0.05 px）：若这是实拍数据，要警惕过拟合；"
                     "合成数据下属正常")
    return "\n\n".join(lines)


def sweep_noise(base_kwargs, levels=(0.0, 0.25, 0.5, 1.0, 1.5, 2.0),
                repeats=2, n_views=15, progress=None):
    """扫描角点噪声，返回 [(noise, rms, fx_err, cx_err)]"""
    rows = []
    for i, nz in enumerate(levels):
        acc = []
        for r in range(repeats):
            kw = dict(base_kwargs)
            kw.update(dict(n_views=n_views, device_noise=nz,
                           seed=1000 + r, keep_images=False))
            ds = generate_dataset(**kw)
            res = calibrate_from_dataset(ds, source="detected")
            if res:
                acc.append(res)
        if acc:
            med = lambda k: float(np.median([a[k] for a in acc]))
            rows.append((nz, med("rms"), med("fx_err_pct"), abs(med("cx_err"))))
        if progress:
            progress((i + 1) / len(levels),
                     desc=f"噪声 {nz} px ({i + 1}/{len(levels)})")
    return rows
