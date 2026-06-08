# py-imaging-quality · 成像质量测试

Python 版成像质量测试工具，原项目为 MATLAB 程序。提供纯 matplotlib GUI，用于检测相机/镜头的各项成像指标：SFR/MTF 解像力、镜头畸变测量与校正、ColorChecker 色彩准确度、阶调响应（Step Chart）、白平衡偏差、暗场噪声检测、棋盘格标定。

---

## 功能总览

| 功能 | 按钮 | 所需 ROI 数 | 说明 |
|------|------|-------------|------|
| **SFR / MTF** | SFR (1 ROI) | 1 | 边缘扩散函数 ESF → MTF 曲线，输出 MTF50/MTF30 |
| **畸变测量** | Dist. Meas. | 1 | 径向畸变系数 k1 与 SMIA TV 畸变率 |
| **畸变校正** | Dist. Correct | — | 输入 k1，对当前图像做畸变校正并对比显示 |
| **棋盘格标定** | Chessboard Cal. | — | 选择棋盘格图片文件夹，输出焦距/畸变/重投影误差 |
| **色彩准确度** | ColorChk (24) | 24 | 24 色卡 patch 的 ΔE、ΔC、饱和度 vs X-Rite 理想值 |
| **阶调响应** | StepChrt (20) | 20 | 20 级灰阶的对比度与可分辨级数 |
| **白平衡** | WB (6 patches) | 6 | 6 级灰阶的 RGB 最大-最小差值与偏色通道 |
| **暗场噪声** | Dark Field | — | 全图的均值/标准差/最值，用于暗场均匀性评估 |

---

## 环境与安装

### 依赖

- Python ≥ 3.9
- numpy
- scipy
- matplotlib（TkAgg 后端）
- Pillow
- scikit-image
- opencv-python
- sympy
- tkinter（Python 标准库，通常已内置）

### 使用 conda 环境

```bash
conda create -n bo_env python=3.10
conda activate bo_env
pip install numpy scipy matplotlib pillow scikit-image opencv-python sympy
```

或者使用项目现有的 `bo_env`：

```bash
conda run -n bo_env python -c "from py_imaging_quality.main import main; main()"
```

---

## 使用方式

### 1. GUI 交互模式（推荐）

```bash
conda run -n bo_env python -c "from py_imaging_quality.main import main; main()"
```

操作流程：

1. 点击 **Open Image** 选择测试图片
2. 点击左侧功能按钮进入对应测试模式
3. 在图片上用鼠标拖拽绘制 ROI 矩形（绿色框）
4. 如需撤销上一个 ROI，点击 **Undo ROI**
5. 完成所需 ROI 数量后，点击 **Compute**
6. 结果以弹窗图表显示，同时状态栏给出关键数值

**不同模式的 ROI 绘制方式：**

- **SFR (1 ROI)**：框选一条清晰的边缘（斜边），边缘应大致从框的一边进入、另一边穿出
- **Dist. Meas. (1 ROI)**：框选包含完整网格图案（如 DIS 测试图）的区域
- **ColorChk (24)**：按行顺序依次框选 24 个色卡 patch，从左上到右下
- **StepChrt (20)**：从最暗到最亮依次框选 20 个灰阶 patch
- **WB (6 patches)**：从最暗到最亮依次框选 6 个灰色 patch

### 2. 程序化调用（无 GUI）

```python
import numpy as np
from PIL import Image
from py_imaging_quality.sfr.sfr_cal import sfr_cal

img = np.array(Image.open('data/v_edge.tif'))
freq, mtf, esf = sfr_cal(img)
```

```python
from py_imaging_quality.distortion.distortion import distortion
k1, TV = distortion(img)
```

```python
from py_imaging_quality.color_check import color_check
delta_L, delta_a, delta_b, delta_E, delta_C, whiteB = color_check(img)
```

```python
from py_imaging_quality.step_chart import step_chart
n, LOG1 = step_chart(img)
```

```python
from py_imaging_quality.chessboard.calibration import calibrate_camera
params = calibrate_camera('path/to/chessboard/images/')
```

---

## 项目结构

```
py_imaging_quality/
├── __init__.py                  # 包入口
├── main.py                      # GUI 主程序（纯 matplotlib 界面）
├── imageread.py                 # 图片读取（PIL）
├── getmyroi.py                  # 交互式 ROI 选择（matplotlib）
├── color_check.py               # ColorChecker 24 色卡色彩准确度
├── step_chart.py                # 阶调响应（Step Chart）分析
├── white_balance.py             # 白平衡分析
├── data/                        # 测试图片
│   ├── v_edge.tif / h_edge.tif  # 垂直/水平边缘图（SFR 测试）
│   ├── ISO_DSC_300a.jpg         # ISO 测试图（畸变/综合测试）
│   ├── lut1.dat / lut3.dat      # LUT 数据文件
│   └── ...
├── sfr/                         # SFR/MTF 解像力计算
│   ├── sfr_cal.py               # 主入口
│   ├── rotatev2.py              # 边缘旋转至垂直
│   ├── centroid.py              # 向量质心计算
│   ├── ahamming.py              # 非对称 Hamming 窗
│   ├── deriv1.py                # FIR 一阶导数
│   ├── findedge.py              # 边缘位置线性拟合
│   ├── project.py               # 超采样边缘投影
│   ├── fir2fix.py               # 导数滤波器 MTF 校正
│   ├── mtf_eff.py               # MTF50/MTF30 计算
│   └── ...                      # 其他辅助模块
├── distortion/                  # 镜头畸变测量与校正
│   ├── distortion.py            # 畸变测量（k1, TV）
│   ├── distortion_correct.py    # 畸变校正
│   ├── findcenter.py            # 网格线中心检测
│   ├── invert_distortion_poly.py # 多项式求逆
│   └── ...                      # 其他辅助模块
└── chessboard/                  # 棋盘格相机标定
    └── calibration.py           # OpenCV 标定
```

---

## 各模块算法说明

### SFR / MTF

遵循 ISO 12233 标准的斜边 SFR 算法：

1. 对 ROI 做 gamma 校正与亮度加权
2. `rotatev2` 将边缘旋转至垂直方向
3. `centroid` 找每行边缘质心 → `findedge` 线性拟合
4. `project` 超采样投影（4×）获取 ESF
5. 差分 → `ahamming` 窗 → FFT → `fir2fix` 校正
6. `mtf_eff` 插值得到 MTF50 与 MTF30

### 畸变测量

径向畸变模型：`ru = rd + k1 · rd³`

通过检测网格图案的横纵线中心位置，使用 scipy.optimize.least_squares（LM 方法）拟合 k1，同时计算 SMIA TV 畸变率。

### 畸变校正

通过多项式求逆得到从校正坐标到畸变坐标的映射，再以 RegularGridInterpolator 双线性插值重建图像。

### ColorChecker

将 24 个色卡 patch 的 RGB 均值转换到 CIELAB 空间，与 X-Rite ColorChecker（D65）理想值对比，计算每色卡的 ΔL\*、Δa\*、Δb\*、ΔE\*、ΔC\* 以及饱和度比值。

### 棋盘格标定

使用 OpenCV 的 `findChessboardCorners` + `calibrateCamera`，对标定图像序列进行角点检测和相机内参估计，输出焦距、径向畸变系数、重投影误差。

---

## 数据输出格式

| 指标 | 输出内容 |
|------|---------|
| SFR | 每通道 MTF50（cy/px）、MTF30（cy/px），ESF 与 MTF 曲线图 |
| 畸变 | `k1` 系数、SMIA TV 畸变率（%） |
| 畸变校正 | 校正前后对比图 |
| 色彩 | 每色卡的 ΔE / ΔC、平均与最大值、饱和度比值（%） |
| 阶调 | 可分辨级数、对比度比值 |
| 白平衡 | 每灰阶的 RGB 最大-最小差（WB error）、偏色通道 |
| 暗场 | 均值、标准差、最小值、最大值 |
| 棋盘格标定 | 图像尺寸、焦距 `(fx, fy)`、径向畸变 `(k1, k2, k3)`、重投影误差 |

---

## MATLAB → Python 转换参考

| MATLAB | Python |
|--------|--------|
| `imread()` | `PIL.Image.open()` / `plt.imread()` |
| `imshow()` | `matplotlib.imshow()` |
| `polyfit` / `polyval` | `np.polyfit` / `np.polyval` |
| `conv()` | `np.convolve()` |
| `fft(x, n)` | `np.fft.fft(x, n)` |
| `meshgrid()` | `np.meshgrid()` |
| `interp2()` | `scipy.interpolate.RegularGridInterpolator` |
| `rgb2lab()` | `skimage.color.rgb2lab()` |
| `lsqnonlin('lm')` | `scipy.optimize.least_squares(method='lm')` |
| `detectCheckerboardPoints` | `cv2.findChessboardCorners()` |
| `estimateCameraParameters` | `cv2.calibrateCamera()` |
| `imdilate` / `imerode` | `scipy.ndimage.binary_dilation` / `binary_erosion` |
| `bwareaopen()` | `skimage.morphology.remove_small_objects()` |

---

## 与原版 MATLAB 程序的差异

1. **GUI**：MATLAB 使用 GUIDE `.fig` 文件 + `appdata`，Python 使用纯 matplotlib 控件（Button、RectangleSelector）+ 临时 tkinter 文件对话框，无额外 GUI 框架依赖
2. **非交互接口**：每个测试模块均提供 `_from_rois()` 非交互版本，供 GUI 传递已绘制好的 ROI 数据，避免重复弹窗选图
3. **畸变拼写**：MATLAB 目录名为 `distrotion/`，Python 中修正为 `distortion/`
4. **LM 算法**：MATLAB 有手写 LevenbergMarquardt.m，Python 直接使用 `scipy.optimize.least_squares(method='lm')`
5. **文件读取**：MATLAB 支持 DICOM 和 RAW 格式，Python 版本暂未包含
6. **标定 GUI**：MATLAB 有 figure_CaliProcess.fig 交互式标定 GUI，Python 通过函数参数传入文件夹路径

---

## License

本项目为成像质量测试算法的 Python 移植，仅供学习和工程使用。
