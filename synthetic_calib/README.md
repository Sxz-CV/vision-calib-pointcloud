# 合成标定数据集生成器

用代码生成**相机参数真值已知**的棋盘格标定图像，用来验证标定流程、理解参数敏感性。

不需要打印机、不需要相机、不需要标定板。

> **能证明什么**：你的标定算法/代码是否正确。
> **不能证明什么**：你的真实相机的内参是多少（那是你自己填进去的）。

---

## 安装

```bash
pip install opencv-python numpy matplotlib
```

Python 3.8+，无其他依赖。

---

## 快速开始

```bash
# 1. 生成 20 张图 + 真值 JSON
python gen_dataset.py -o output/demo -n 20

# 2. 用生成的真值角点跑标定，和真值对比
python check_calib.py -d output/demo --use true
```

第二次命令的输出应该长这样（误差都在 1e-4 量级）：

```
重投影误差 RMS = 0.00004 px
                     估计值            真值              误差
fx              800.0002      800.0000        +0.0000%
fy              800.0003      800.0000        +0.0000%
cx              320.0002      320.0000        +0.0002 px
cy              240.0003      240.0000        +0.0003 px
```

**这就是对拍（ground truth verification）**：算法正确时，应该能精确恢复真值。

---

## 三个脚本

| 脚本 | 作用 |
|---|---|
| `gen_dataset.py` | 生成合成标定图 + `dataset.json`（含相机真值、位姿、角点） |
| `check_calib.py` | 读 JSON，跑 `cv2.calibrateCamera`，和真值对比 |
| `sweep.py` | 批量扫参数，输出对比表 + 曲线 |

---

## 常用参数

### 生成

```bash
# 高清 + 长焦 + 强畸变
python gen_dataset.py -o output/hd --img-w 1920 --img-h 1080 --fx 1600 --k1 -0.35

# 模拟真实退化：噪声 + 模糊 + JPEG + 暗角
python gen_dataset.py -o output/real --noise 0.8 --blur 1.0 --jpeg 90 --vignette 0.2

# 顺便输出角点叠加图，方便肉眼确认
python gen_dataset.py -o output/demo --preview

# 同时给"检测角点"加噪声（模拟亚像素定位误差）
python gen_dataset.py -o output/noisy --device-noise 0.5
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `--cols / --rows` | 9 / 6 | **格子数**，不是内角点数 |
| `--square` | 20.0 | 格子边长 mm |
| `--img-w / --img-h` | 640 / 480 | 输出分辨率 |
| `--fx / --fy` | 800 / =fx | 焦距像素 |
| `--cx / --cy` | 中心 | 主点 |
| `--k1 --k2 --p1 --p2 --k3` | −0.15, 0.05, 5e-4, −5e-4, 0 | 畸变 |
| `-n / --n` | 20 | 图片数量 |
| `--max-tilt` | 45 | 最大倾斜角（度） |
| `--dist-mm` | 自动 | 板到相机距离范围 |
| `--device-noise` | 0 | 写进 JSON 的角点噪声（px） |
| `--noise --blur --jpeg --vignette` | 0 | 图像退化 |
| `--seed` | 0 | 随机种子，同种子完全可复现 |

### 验证

```bash
python check_calib.py -d output/demo                    # 用 JSON 里的带噪角点
python check_calib.py -d output/demo --use true         # 用无噪真值
python check_calib.py -d output/demo --detect           # 真跑 findChessboardCorners
python check_calib.py -d output/demo --flags 2          # 固定主点
```

`--flags` 是 `cv2.calibrateCamera` 的整数标志位，常用值：

| 值 | 含义 |
|---|---|
| 0 | 全估（5 参数） |
| 2 | `CALIB_FIX_PRINCIPAL_POINT` 固定主点 |
| 4 | `CALIB_FIX_ASPECT_RATIO` 固定 fx/fy 比 |
| 6 | 2 + 4 |

### 扫描

```bash
python sweep.py --what noise --repeats 5
python sweep.py --what fx --repeats 5 -n 20
python sweep.py --what all --repeats 3
```

---

## 输出结构

```
output/demo/
├─ img_000.png ~ img_019.png    合成图像
├─ preview_000.png ...          角点叠加（--preview 时才生成）
├─ board.png                    使用的棋盘贴图
└─ dataset.json                 真值 + 位姿 + 角点
```

### dataset.json 结构

```json
{
  "camera_true": {
    "K": [[800, 0, 320], [0, 800, 240], [0, 0, 1]],
    "dist": [-0.15, 0.05, 0.0005, -0.0005, 0.0],
    "fx": 800.0, "fy": 800.0, "cx": 320.0, "cy": 240.0,
    "image_size": [640, 480]
  },
  "board": {
    "cols_squares": 9, "rows_squares": 6, "square_mm": 20.0,
    "board_size_mm": [180.0, 120.0],
    "pattern_size": [8, 5]
  },
  "sampling": {"n_views": 20, "max_tilt_deg": 45, "dist_mm_range": [..], "seed": 0},
  "degradation": {"gaussian_noise_std": 0.5, "blur_sigma": 0.8, ...},
  "corner_noise_px": 0.5,
  "views": [
    {
      "file": "img_000.png",
      "rvec": [...], "tvec": [...],
      "corners_true_px": [[x, y], ...],
      "corners_detected_px": [[x, y], ...]
    }
  ]
}
```

**`camera_true` 是整个数据集的价值所在**，任何标定实现都能拿它做对拍。

---

## 实测结论

### 零噪声能精确恢复真值

```
fx 800.0002 vs 800.0000   误差 0.0000%
cx 320.0002 vs 320.0000   偏差 0.0002 px
RMS = 0.00004 px
```

### 角点噪声是主导因素（repeats=2, n=15）

| 角点噪声 | RMS (px) | fx 误差 | cx 偏差 |
|---|---|---|---|
| 0.0 | 0.000 | 0.00% | 0.00 px |
| 0.2 | 0.276 | 0.29% | 2.61 px |
| 0.5 | 0.689 | 0.73% | 7.70 px |
| 1.0 | 1.378 | 1.49% | 12.70 px |
| 2.0 | 2.756 | 2.81% | 16.44 px |

**主点比焦距敏感约 10 倍。** 这解释了为什么实拍标定里 `cx` 常常偏离图像中心几十像素——不是算法错，是约束不足。

### 从图像实际检测 vs 人造噪声

同样 20 张带模糊/暗角的图：

| 角点来源 | RMS | fx 误差 | cx 偏差 |
|---|---|---|---|
| 人造高斯噪声 0.5 px | 0.668 | +0.42% | **+15.97 px** |
| `findChessboardCorners + cornerSubPix` | 0.159 | +0.23% | **−2.62 px** |

**真实检测（利用梯度）比人造均匀噪声准得多。** 人造噪声更"毒"，不能用来模拟真实检测误差。

### 其他参数在当前设置下测不出影响

焦距、图片数量、畸变强度的扫描结果被位姿随机波动（约 ±0.7%）淹没。要测出这些影响，需要 `--repeats 8` 以上，且效应本身要大于 0.7%。

---

## 常见问题

**Q: patternSize 到底填几？**

`patternSize = (cols-1, rows-1)`。9×6 个格子 → `(8, 5)`。这是内角点数，不是格子数。

**Q: 为什么我的 reals 图跑出来 cx 偏了几十像素？**

主点是最难估的量。本工具扫描显示：0.5 px 的角点噪声就能让 cx 偏 7.7 px，2 px 噪声能偏 16 px。实拍里 cx 偏移 20~30 px 是常见现象。

**Q: 生成的和真实拍的一样吗？**

几何上精确（透视 + 畸变），但没有真实世界的纸褶皱、打印缩放、光照不均。**它验证算法，不验证硬件。**

**Q: 能不能用它替代打印标定板？**

不能。它给不了你真实相机的参数。它能告诉你"如果相机是 fx=800、k1=-0.15，算法应该能估出这些数"。

---

## 原理简述

```
设定真值 K, dist
     ↓
随机采样位姿 (rvec, tvec)，保证整块板在画面内
     ↓
渲染：对输出每个像素
      1. undistortPoints 反解畸变 → 理想像素坐标
      2. 单应逆变换 → 板平面 mm 坐标
      3. 从棋盘贴图采样
     ↓
退化：模糊 / 暗角 / 噪声 / JPEG
     ↓
精确投影内角点 → corners_true_px
     ↓
加高斯噪声 → corners_detected_px
     ↓
落盘 PNG + JSON
```

**渲染的关键在"反解畸变"这一步**：如果直接正向投影再画，会有插值空洞和重叠；反向映射保证每个输出像素都精确对应一个板坐标。
