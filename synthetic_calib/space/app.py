"""合成标定数据集生成器 —— Gradio 网页界面

任何人打开链接就能调参数、生成标定图、看"真值 vs 估计"的对比，
不需要打印标定板、不需要相机、不需要装环境。
"""
import io

import cv2
import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import core

# 图表用英文标签，避免 Linux 上缺中文字体出现方块
plt.rcParams.update({
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

INTRO = """
# 合成标定数据集生成器

不用打印机、不用相机，直接生成**相机参数真值已知**的棋盘格标定图，
然后立刻跑一遍标定，看算法能不能把真值估出来。

> **能证明什么**：你的标定算法/流程是否正确。
> **不能证明什么**：你真实相机的内参（那是下面这些参数，是你自己填的）。

### 怎么玩

1. 左边调参数，点 **生成并标定**
2. 看右下角的 **真值 vs 估计** 表格——差距就是标定误差
3. 把 **角点噪声** 从 0 拖到 2，再点一次，观察主点怎么崩
4. 展开底部的 **参数扫描**，一次跑多组，直接出曲线
"""


def build_preview(images, corners_list, max_show=6):
    """把前几张图拼成一张预览，并叠加检测到的角点"""
    if not images:
        return None
    sel = list(range(min(max_show, len(images))))
    tiles = []
    for i in sel:
        img = images[i]
        vis = img.copy()
        if corners_list and i < len(corners_list):
            for p in corners_list[i]:
                cv2.circle(vis, (int(round(p[0])), int(round(p[1]))),
                           2, (0, 0, 255), -1)
        cv2.putText(vis, f"#{i}", (6, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)
        tiles.append(vis)

    while len(tiles) % 3 != 0:
        tiles.append(np.full_like(tiles[0], 255))
    rows = [np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)]
    return cv2.cvtColor(np.vstack(rows), cv2.COLOR_BGR2RGB)


def result_table(res):
    """真值 vs 估计 的数据表"""
    if res is None:
        return [["—", "—", "—", "—"]]
    return [
        ["fx", f"{res['fx_true']:.3f}", f"{res['fx_est']:.3f}",
         f"{res['fx_err_pct']:+.4f} %"],
        ["fy", f"{res['fy_true']:.3f}", f"{res['fy_est']:.3f}",
         f"{res['fy_err_pct']:+.4f} %"],
        ["cx", f"{res['cx_true']:.3f}", f"{res['cx_est']:.3f}",
         f"{res['cx_err']:+.4f} px"],
        ["cy", f"{res['cy_true']:.3f}", f"{res['cy_est']:.3f}",
         f"{res['cy_err']:+.4f} px"],
        ["k1", f"{res['k1_true']:.5f}", f"{res['k1_est']:.5f}",
         f"{res['k1_est'] - res['k1_true']:+.6f}"],
    ]


def do_generate(img_w, img_h, fx, fy, k1, k2, cols_sq, rows_sq, square_mm,
                n_views, max_tilt, device_noise, noise, blur, jpeg, vignette,
                mode, seed, progress=gr.Progress()):
    progress(0.05, desc="生成数据集…")
    try:
        kw = dict(img_w=int(img_w), img_h=int(img_h),
                  fx=float(fx), fy=float(fy) if fy else None,
                  k1=float(k1), k2=float(k2),
                  cols_sq=int(cols_sq), rows_sq=int(rows_sq),
                  square_mm=float(square_mm),
                  n_views=int(n_views), max_tilt=float(max_tilt),
                  device_noise=float(device_noise),
                  noise=float(noise), blur=float(blur),
                  jpeg=int(jpeg) if jpeg else None,
                  vignette=float(vignette),
                  seed=int(seed), keep_images=True)
        ds = core.generate_dataset(**kw)
    except Exception as exc:
        return None, f"❌ 生成失败：{exc}", [["—"] * 4]

    progress(0.45, desc=f"已生成 {ds['made']} 张，开始标定…")

    src = {"真值角点（零噪声）": "true",
           "带噪角点（模拟检测误差）": "detected",
           "从图像实际检测": "detect"}.get(mode, "detected")

    try:
        res = core.calibrate_from_dataset(ds, source=src)
    except Exception as exc:
        return None, f"❌ 标定失败：{exc}", [["—"] * 4]

    progress(0.85, desc="整理结果…")

    corners = [v["corners_detected"] for v in ds["views"]]
    preview = build_preview(ds["images"], corners)

    if res is None:
        return preview, "❌ 可用视图不足（<6 张），无法标定", [["—"] * 4]

    truth = ds["truth"]
    head = [
        f"**角点来源**：{mode}　|　**实际生成**：{ds['made']}/{ds['requested']} 张　|　"
        f"**patternSize**：{truth['pattern_size'][0]}×{truth['pattern_size'][1]}　|　"
        f"**板尺寸**：{truth['board_mm'][0]:.0f}×{truth['board_mm'][1]:.0f} mm",
        "",
        core.verdict(res),
    ]
    if ds["made"] < ds["requested"]:
        head.append(f"\n⚠️ 只生成了 {ds['made']} 张：倾斜角或张数设置过激，"
                    f"位姿采样失败较多。把倾斜角调小一点试试。")

    progress(1.0, desc="完成")
    return preview, "\n\n".join(head), result_table(res)


def do_sweep(img_w, img_h, fx, cols_sq, rows_sq, square_mm,
             device_noise, noise, blur, jpeg, n_views, repeats,
             progress=gr.Progress()):
    base = dict(img_w=int(img_w), img_h=int(img_h), fx=float(fx),
                fy=float(fx), cols_sq=int(cols_sq), rows_sq=int(rows_sq),
                square_mm=float(square_mm), max_tilt=45.0,
                noise=float(noise), blur=float(blur),
                jpeg=int(jpeg) if jpeg else None, vignette=0.15,
                keep_images=False)
    try:
        rows = core.sweep_noise(base, repeats=max(1, int(repeats)),
                                n_views=max(8, int(n_views)),
                                progress=lambda p, desc="": progress(p, desc=desc))
    except Exception as exc:
        return None, [["失败", str(exc), "", "", ""]]

    if not rows:
        return None, [["失败", "没有可用结果", "", "", ""]]

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    x = [r[0] for r in rows]
    ax.plot(x, [r[2] for r in rows], "o-", color="C0",
            label="focal length error (%)")
    ax.set_xlabel("corner noise injected (px)")
    ax.set_ylabel("focal error (%)", color="C0")
    ax.tick_params(axis="y", labelcolor="C0")
    ax2 = ax.twinx()
    ax2.plot(x, [r[3] for r in rows], "s--", color="C1",
             label="principal point error (px)")
    ax2.set_ylabel("principal point error (px)", color="C1")
    ax2.tick_params(axis="y", labelcolor="C1")
    ax2.grid(False)
    ax.set_title("Corner noise vs calibration accuracy")
    fig.tight_layout()

    table = [[f"{r[0]:.2f}", f"{r[1]:.4f}", f"{r[2]:+.3f} %", f"{r[3]:.3f}"]
             for r in rows]
    return fig, table


def build_ui():
    with gr.Blocks(title="合成标定数据集生成器") as demo:
        gr.Markdown(INTRO)

        with gr.Row():
            # ---------------- 左：参数 ----------------
            with gr.Column(scale=1):
                with gr.Group():
                    gr.Markdown("### 相机")
                    with gr.Row():
                        img_w = gr.Slider(320, 1920, value=640, step=32,
                                          label="图像宽 px")
                        img_h = gr.Slider(240, 1440, value=480, step=32,
                                          label="图像高 px")
                    with gr.Row():
                        fx = gr.Number(value=800, label="fx (px)",
                                       precision=1)
                        fy = gr.Number(value=800, label="fy (px)",
                                       precision=1)
                    with gr.Row():
                        k1 = gr.Slider(-0.8, 0.3, value=-0.15, step=0.01,
                                       label="k1 径向畸变")
                        k2 = gr.Slider(-0.5, 0.5, value=0.05, step=0.01,
                                       label="k2")
                    gr.Markdown("<sub>k1 越负 = 桶形越强（手机广角常 −0.3）</sub>")

                with gr.Group():
                    gr.Markdown("### 标定板")
                    with gr.Row():
                        cols_sq = gr.Slider(4, 12, value=9, step=1,
                                            label="横向格子数")
                        rows_sq = gr.Slider(3, 10, value=6, step=1,
                                            label="纵向格子数")
                    square_mm = gr.Number(value=20.0, label="格子边长 mm")
                    gr.Markdown("<sub>内角点 = 格子数−1（OpenCV patternSize）</sub>")

                with gr.Group():
                    gr.Markdown("### 采样")
                    n_views = gr.Slider(6, 40, value=20, step=1, label="图片张数")
                    max_tilt = gr.Slider(5, 60, value=45, step=1,
                                         label="最大倾斜角 °")
                    device_noise = gr.Slider(0.0, 2.5, value=0.5, step=0.05,
                                             label="角点噪声 px")
                    gr.Markdown("<sub>模拟亚像素定位误差，**影响精度最大的因素**，拖到 0 看结果</sub>")

                with gr.Group():
                    mode = gr.Radio(
                        choices=["带噪角点（模拟检测误差）",
                                 "真值角点（零噪声）",
                                 "从图像实际检测"],
                        value="带噪角点（模拟检测误差）",
                        label="标定用哪种角点")
                    run_btn = gr.Button("生成并标定", variant="primary", size="lg")

                with gr.Accordion("高级：图像退化与随机种子", open=False):
                    gr.Markdown(
                        "拖动这些会改变图像质量，模拟不同相机/光照条件。"
                        "一般不需要改。")
                    noise = gr.Slider(0.0, 5.0, value=0.8, step=0.1,
                                      label="高斯噪声 std")
                    blur = gr.Slider(0.0, 3.0, value=1.0, step=0.1,
                                     label="模糊 sigma")
                    jpeg = gr.Slider(0, 100, value=90, step=1, label="JPEG 质量")
                    vignette = gr.Slider(0.0, 1.0, value=0.2, step=0.05,
                                         label="暗角强度")
                    seed = gr.Number(value=0, label="随机种子", precision=0)

            # ---------------- 右：结果 ----------------
            with gr.Column(scale=1):
                preview = gr.Image(label="生成的图（前 6 张，红点=喂给标定的角点）",
                                   type="numpy", height=280)
                verdict = gr.Markdown("点左边的 **生成并标定** 开始。")
                table = gr.Dataframe(
                    headers=["参数", "真值", "估计", "误差"],
                    datatype=["str"] * 4,
                    value=[["—"] * 4],
                    label="真值 vs 估计", interactive=False, wrap=True)

        with gr.Accordion("参数扫描：角点噪声 vs 标定精度（约 20~60 秒）",
                          open=False):
            gr.Markdown(
                "一次跑 6 组不同角点噪声，输出曲线。"
                "**这张图是理解标定精度最有用的东西**："
                "看主点（橙线）怎么随噪声爆炸，而焦距（蓝线）慢得多。")
            with gr.Row():
                sw_views = gr.Slider(8, 25, value=15, step=1, label="每组张数")
                sw_repeats = gr.Slider(1, 4, value=2, step=1, label="每组重复次数")
                sw_btn = gr.Button("开始扫描", variant="secondary")
            with gr.Row():
                sw_plot = gr.Plot(label="扫描曲线")
                sw_table = gr.Dataframe(
                    headers=["角点噪声 px", "RMS px", "fx 误差 %", "cx 偏差 px"],
                    datatype=["str"] * 4, value=[["—"] * 4],
                    label="数据", interactive=False)

        run_btn.click(
            do_generate,
            inputs=[img_w, img_h, fx, fy, k1, k2, cols_sq, rows_sq, square_mm,
                    n_views, max_tilt, device_noise, noise, blur, jpeg,
                    vignette, mode, seed],
            outputs=[preview, verdict, table],
        )
        sw_btn.click(
            do_sweep,
            inputs=[img_w, img_h, fx, cols_sq, rows_sq, square_mm,
                    device_noise, noise, blur, jpeg, sw_views, sw_repeats],
            outputs=[sw_plot, sw_table],
        )

        gr.Markdown("""
---
### 三个可以自己验证的结论

1. **把角点噪声拖到 0**：焦距和主点误差都接近 0 → 说明标定算法本身是对的
2. **把角点噪声拖到 0.5 → 2.0**：焦距误差涨到约 3%，但主点会偏移十几像素 → **主点比焦距敏感约 10 倍**
3. **把 k1 从 −0.15 拖到 −0.5**：RMS 几乎不变 → **重投影误差看不见"参数设错"，它只证明模型和数据自洽**

第 3 条是标定这件事最容易踩的坑：真实工程里你把格子边长写错 10%，RMS 依然漂亮，
但你量任何东西都会偏 10%。
""")

    return demo


if __name__ == "__main__":
    build_ui().queue().launch(theme=gr.themes.Soft())
