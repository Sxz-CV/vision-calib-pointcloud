"""生成精确 A4 尺寸的棋盘格标定板 PDF（无外部依赖，只需 matplotlib）

特点：
    - 页面严格等于 A4（210 x 297 mm），PDF 矢量输出，任意倍率不失真
    - 棋盘尺寸按毫米精确绘制
    - 页面上自带 100 mm 校验尺：打印后用普通尺子量一下，就知道有没有被缩放
    - 自动标注内角点数，避免 rows/columns 混淆

用法：
    python make_board.py                                  # 默认 9x6 格, 20mm
    python make_board.py --cols 6 --rows 5 --square 25    # 自定义
    python make_board.py --out my_board.pdf

打印设置（关键）：
    PDF 阅读器里选"实际大小 / 100%"或"不缩放"，绝不能选"适应页面"。
    打印后量一下页面底部的校验尺：从 0 到 100 刻度必须是 100 mm。
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

MM = 1.0 / 25.4          # mm -> inch
A4_W, A4_H = 210.0, 297.0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cols", type=int, default=9, help="横向格子数")
    p.add_argument("--rows", type=int, default=6, help="纵向格子数")
    p.add_argument("--square", type=float, default=20.0, help="格子边长 mm")
    p.add_argument("--out", default="chessboard_A4.pdf")
    p.add_argument("--png", action="store_true", help="同时输出 PNG 预览")
    return p.parse_args()


def main():
    args = parse_args()
    bw = args.cols * args.square      # 板宽 mm
    bh = args.rows * args.square      # 板高 mm

    print(f"棋盘: {args.cols} x {args.rows} 格, 每格 {args.square} mm")
    print(f"板尺寸: {bw:.1f} x {bh:.1f} mm")
    print(f"OpenCV patternSize 应填: ({args.cols - 1}, {args.rows - 1})")
    print(f"A4 剩余边距: 左右各 {(A4_W - bw) / 2:.1f} mm, 上方 {(A4_H - bh - 40) / 2:.1f} mm")

    if bw > A4_W - 10 or bh > A4_H - 60:
        raise SystemExit(f"板 {bw:.0f}x{bh:.0f} mm 放不进 A4（需留边距）")

    # 页面严格 A4，坐标系直接用毫米
    fig = plt.figure(figsize=(A4_W * MM, A4_H * MM))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, A4_W)
    ax.set_ylim(0, A4_H)
    ax.set_aspect("equal")
    ax.axis("off")

    # 棋盘：左上角定位，第一格为黑
    x0 = (A4_W - bw) / 2
    y0 = A4_H - (A4_H - bh - 40) / 2 - bh + 10
    for r in range(args.rows):
        for c in range(args.cols):
            if (r + c) % 2 == 0:
                ax.add_patch(Rectangle(
                    (x0 + c * args.square, y0 + r * args.square),
                    args.square, args.square,
                    facecolor="black", edgecolor="none"))

    # 板外框细线，方便对位
    ax.add_patch(Rectangle((x0, y0), bw, bh, fill=False,
                           edgecolor="0.6", lw=0.4))

    # 100 mm 校验尺
    ry = y0 - 18
    ax.plot([x0, x0 + 100], [ry, ry], color="black", lw=0.8)
    for i in range(0, 101, 10):
        h = 4 if i % 50 == 0 else 2.2
        ax.plot([x0 + i, x0 + i], [ry, ry + h], color="black", lw=0.7)
        if i % 50 == 0:
            ax.text(x0 + i, ry - 4.5, str(i), ha="center", va="top", fontsize=5)
    ax.text(x0 + 103, ry - 1, "mm (verify this is exactly 100 mm)",
            fontsize=5, va="bottom")

    # 说明文字（同时竖放一遍，防止方向误解）
    info = (f"{args.cols}x{args.rows} squares @ {args.square:g} mm  |  "
            f"patternSize ({args.cols - 1},{args.rows - 1})  |  "
            f"board {bw:g}x{bh:g} mm  |  print at 100%")
    ax.text(A4_W / 2, ry - 12, info, ha="center", va="top", fontsize=6)

    fig.savefig(args.out, format="pdf")
    print(f"\n-> {args.out}")

    if args.png:
        png = args.out.rsplit(".", 1)[0] + ".png"
        fig.savefig(png, dpi=300)
        print(f"-> {png}")

    # 校验：PDF 页面尺寸
    import re
    with open(args.out, "rb") as f:
        raw = f.read()
    m = re.search(rb"/MediaBox\s*\[\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", raw)
    if m:
        w = (float(m.group(3)) - float(m.group(1))) / 72 * 25.4
        h = (float(m.group(4)) - float(m.group(2))) / 72 * 25.4
        print(f"PDF 页面: {w:.2f} x {h:.2f} mm  "
              f"({'A4 ✓' if abs(w - 210) < 0.5 and abs(h - 297) < 0.5 else '不是 A4 ✗'})")


if __name__ == "__main__":
    main()
