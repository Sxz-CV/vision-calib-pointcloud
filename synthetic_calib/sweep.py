"""参数扫描：一次跑多组配置，输出对比表和曲线

回答的问题：
    焦距变化、畸变强度变化、角点噪声变化、图片数量变化，
    分别对标定精度影响多大？

用法：
    python sweep.py --what fx          # 扫焦距
    python sweep.py --what noise       # 扫角点噪声
    python sweep.py --what views       # 扫图片数量
    python sweep.py --what k1          # 扫畸变强度
    python sweep.py --what all         # 全部（较慢，约 5-10 分钟）
    python sweep.py --what fx --repeats 5 --n 20
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

HERE = os.path.dirname(os.path.abspath(__file__))
GEN = os.path.join(HERE, "gen_dataset.py")
CHK = os.path.join(HERE, "check_calib.py")
PY = sys.executable


def run_gen(outdir, extra):
    cmd = [PY, GEN, "-o", outdir] + [str(x) for x in extra]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                       errors="ignore")
    return r.returncode == 0, r.stdout + r.stderr


def run_check(outdir, use="detected"):
    cmd = [PY, CHK, "-d", outdir, "--use", use]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                       errors="ignore")
    if r.returncode != 0:
        return None
    return parse_check(r.stdout)


def parse_check(txt):
    """从 check_calib 输出里抓数字"""
    out = {"rms": None, "fx": None, "fy": None, "cx": None, "cy": None,
           "fx_err_pct": None, "cx_err": None}
    for line in txt.splitlines():
        s = line.strip()
        if s.startswith("重投影误差 RMS"):
            try:
                out["rms"] = float(s.split("=")[1].split("px")[0])
            except Exception:
                pass
        elif s.startswith("fx "):
            p = s.split()
            out["fx"] = float(p[1]); out["fx_err_pct"] = float(p[3].rstrip("%"))
        elif s.startswith("fy "):
            p = s.split()
            out["fy"] = float(p[1])
        elif s.startswith("cx "):
            p = s.split()
            out["cx"] = float(p[1]); out["cx_err"] = float(p[3])
        elif s.startswith("cy "):
            out["cy"] = float(p[1])
    return out


def one_config(tmpdir, gen_args, repeats, use, quiet=True):
    """一组配置跑 repeats 次，取中位数"""
    acc = []
    for k in range(repeats):
        d = os.path.join(tmpdir, f"r{k}")
        if os.path.isdir(d):
            shutil.rmtree(d)
        ok, log = run_gen(d, list(gen_args) + ["--seed", str(100 + k)])
        if not ok:
            continue
        r = run_check(d, use=use)
        if r and r["fx_err_pct"] is not None:
            acc.append(r)
        shutil.rmtree(d, ignore_errors=True)
    if not acc:
        return None
    med = lambda key: float(np.median([a[key] for a in acc]))
    return {"rms": med("rms"), "fx_err_pct": med("fx_err_pct"),
            "cx_err": abs(med("cx_err"))}


def sweep(name, values, gen_fn, repeats, use, base_n):
    print(f"\n{'=' * 78}")
    print(f"扫描 {name}")
    print(f"{'=' * 78}")
    print(f"{'取值':>10} {'RMS(px)':>10} {'fx误差(%)':>12} {'cx偏差(px)':>12}")
    print("-" * 78)
    tmp = os.path.join(HERE, "output", "_tmp")
    os.makedirs(tmp, exist_ok=True)
    rows = []
    for v in values:
        gen_args = gen_fn(v, base_n)
        r = one_config(tmp, gen_args, repeats, use)
        if r is None:
            print(f"{str(v):>10} 失败")
            continue
        rows.append((v, r["rms"], r["fx_err_pct"], r["cx_err"]))
        print(f"{str(v):>10} {r['rms']:>10.4f} {r['fx_err_pct']:>12.3f} "
              f"{r['cx_err']:>12.3f}")
    shutil.rmtree(tmp, ignore_errors=True)
    return rows


def plot_all(results, outdir):
    n = len(results)
    if n == 0:
        return
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, (name, rows, xlabel) in zip(axes, results):
        x = [r[0] for r in rows]
        ax.plot(x, [r[2] for r in rows], 'o-', label='fx 相对误差 (%)')
        ax.set_xlabel(xlabel)
        ax.set_ylabel('fx 相对误差 (%)')
        ax.set_title(name)
        ax.grid(alpha=.3)
        ax2 = ax.twinx()
        ax2.plot(x, [r[3] for r in rows], 's--', color='C1', alpha=.75)
        ax2.set_ylabel('cx 偏差 (px)', color='C1')
        ax2.tick_params(axis='y', labelcolor='C1')
    plt.tight_layout()
    p = os.path.join(outdir, "sweep_result.png")
    plt.savefig(p, dpi=110)
    print(f"\n-> {p}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--what", default="fx",
                   choices=["fx", "noise", "views", "k1", "all"])
    p.add_argument("--repeats", type=int, default=3, help="每组重复次数")
    p.add_argument("-n", "--n", type=int, default=15, help="每组图片数")
    p.add_argument("--use", choices=["detected", "true"], default="detected")
    p.add_argument("--out", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    outdir = args.out or os.path.join(HERE, "output")
    os.makedirs(outdir, exist_ok=True)
    R = args.repeats

    results = []

    if args.what in ("fx", "all"):
        rows = sweep("焦距 fx", [400, 600, 800, 1200, 1600, 2400],
                     lambda v, n: ["-n", n, "--fx", v, "--device-noise", 0.5],
                     R, args.use, args.n)
        results.append(("焦距 fx", rows, "fx (px)"))

    if args.what in ("noise", "all"):
        rows = sweep("角点噪声", [0.0, 0.2, 0.5, 1.0, 1.5, 2.0],
                     lambda v, n: ["-n", n, "--fx", 800,
                                   "--device-noise", v],
                     R, args.use, args.n)
        results.append(("角点噪声", rows, "角点噪声 (px)"))

    if args.what in ("views", "all"):
        rows = sweep("图片数量", [4, 6, 8, 12, 16, 20, 30],
                     lambda v, n: ["-n", v, "--fx", 800,
                                   "--device-noise", 0.5],
                     R, args.use, args.n)
        results.append(("图片数量", rows, "图片数"))

    if args.what in ("k1", "all"):
        rows = sweep("畸变强度 k1", [0.0, -0.1, -0.2, -0.35, -0.5, -0.7],
                     lambda v, n: ["-n", n, "--fx", 800, "--k1", v,
                                   "--device-noise", 0.5],
                     R, args.use, args.n)
        results.append(("畸变 k1", rows, "k1"))

    plot_all(results, outdir)

    # 存 CSV
    import csv
    cp = os.path.join(outdir, "sweep_result.csv")
    with open(cp, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["scan", "value", "rms_px", "fx_err_pct", "cx_err_px"])
        for name, rows, _ in results:
            for r in rows:
                w.writerow([name, r[0], f"{r[1]:.4f}", f"{r[2]:.3f}", f"{r[3]:.3f}"])
    print(f"-> {cp}")


if __name__ == "__main__":
    main()
