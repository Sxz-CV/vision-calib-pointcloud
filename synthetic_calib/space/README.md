---
title: 合成标定数据集生成器
emoji: 🎯
colorFrom: blue
colorTo: purple
sdk: gradio
app_file: app.py
pinned: false
license: mit
short_description: 生成真值已知的棋盘格标定图，验证相机标定算法
---

# 合成标定数据集生成器

不用打印机、不用相机，直接生成**相机参数真值已知**的棋盘格标定图，
然后立刻跑一遍标定，看算法能不能把真值估出来。

## 能证明什么 / 不能证明什么

| 能 | 不能 |
|---|---|
| 验证你的标定算法和代码是否正确 | 得到你真实相机的内参 |
| 量化"角点噪声如何影响精度" | 替代实拍标定 |
| 让人直观理解主点为什么难估 | 反映真实世界的纸褶皱、打印缩放 |

## 三个可以自己动手验证的结论

**① 零噪声时，标定能精确恢复真值**

把「角点噪声」拖到 0，点生成。焦距和主点误差都接近 0（1e-4 量级）。
这证明标定算法本身没有系统偏差。

**② 主点比焦距敏感约 10 倍**

| 角点噪声 | 焦距误差 | 主点偏差 |
|---|---|---|
| 0.0 px | 0.00 % | 0.00 px |
| 0.5 px | 0.83 % | 11.1 px |
| 1.0 px | 1.80 % | 26.6 px |
| 2.0 px | 4.14 % | 75.3 px |

这解释了实拍标定里主点为什么常常偏离图像中心几十像素——不是算法错，是约束不足。

**③ 重投影误差看不见"参数设错"**

把 k1 从 −0.15 拖到 −0.5，RMS 几乎不变。

真实工程里，把标定板格子边长写错 10%，RMS 依然漂亮，
但你量任何东西都会偏 10%。

## 关于本工具

| 项 | 说明 |
|---|---|
| 渲染方式 | 对每个输出像素反解畸变 → 单应逆变换 → 采样棋盘贴图 |
| 几何精度 | 零噪声时能恢复真值到 1e-4，说明投影链路无系统误差 |
| 依赖 | gradio + opencv-python-headless + numpy + matplotlib |

## 本地运行

```bash
pip install -r requirements.txt
python app.py
```

## 文件结构

```
app.py            Gradio 界面
core.py           生成 + 标定的核心逻辑（可脱离 GUI 单独调用）
requirements.txt
```

`core.py` 也可以直接当库用：

```python
import core
ds = core.generate_dataset(n_views=20, device_noise=0.5)
res = core.calibrate_from_dataset(ds, source="detected")
print(res["fx_err_pct"], res["cx_err"])
```
