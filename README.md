# 任务三：语言驱动像质评价 Agent

这是一个用于“像质评价实验”的本地 Agent 程序。用户只需要上传测试图并输入自然语言，例如“帮我测画面边缘 MTF”或“分析色彩还原度和灰阶”，程序会：

1. 调用 DeepSeek 理解自然语言并生成 workflow。
2. 用 OpenCV 自动定位 ROI。
3. 调用本地 `py_imaging_quality` 算法计算 MTF/SFR、色彩还原度、灰阶等指标。
4. 输出 OpenCV 框选图、曲线、指标表、JSON 和 Agent 总结。
5. 自动识别失败时，提供人工 ROI 修正和按任务类型的框选说明。

本发布包已经包含程序代码、网页界面、本地算法库和测试图片。不要把自己的 API Key 写入代码或提交到 GitHub。

## 目录结构

```text
task3_image_quality_agent_release/
  task3_agent.py              # Agent 核心、OpenCV ROI、光学测量调用
  task3_web_app.py            # Flask 图形页面
  task3_web/                  # HTML/CSS/JS 前端
  py_imaging_quality/         # 本地像质评价算法库
  sample_images/              # 测试图片
  requirements.txt            # Python 依赖
  start_web.bat               # Windows 一键启动图形页面
  run_smoke_test.bat          # 离线自测
  smoke_test.py               # 不需要 API Key 的基础自测
  push_to_github.ps1          # 推送到 GitHub 的脚本
```

## 环境要求

- Windows 10/11。
- Python 3.10+ 或 Anaconda。
- 能访问 DeepSeek API 的网络环境。
- 一个可用的 DeepSeek API Key。

## 一键启动

双击：

```text
start_web.bat
```

脚本会自动：

1. 创建 `.venv` 虚拟环境。
2. 安装 `requirements.txt` 中的依赖。
3. 启动本地图形页面。

启动后浏览器打开：

```text
http://127.0.0.1:7860
```

如果浏览器没有自动弹出，就手动复制上面的地址。

## 第一次使用

1. 打开页面后，先输入 DeepSeek API Key。
2. 点击“验证 API Key 并进入”。
3. 上传 `sample_images` 中的测试图片。
4. 输入自然语言要求。
5. 点击“让 Agent 开始评价”。
6. 查看 OpenCV 框选图、曲线、指标和 Agent 回答。

API Key 只保存在当前浏览器页面内，不会写入代码、JSON 或结果文件。

## 推荐测试图片和提示词

| 功能 | 图片 | 推荐输入 |
|---|---|---|
| 中心 MTF | `sample_images/2014_3_composite.JPG` | `请测量画面中心的水平和竖直MTF，并给出曲线和总结` |
| 边缘 MTF | `sample_images/2014_3_composite.JPG` | `请帮我测量画面边缘的MTF` |
| 灰阶 | `sample_images/2014_3_composite.JPG` | `请自动定位环形20级灰阶，分析可分辨过渡和阶调响应` |
| 色彩还原度 | `sample_images/colorD65.bmp` | `请识别ColorChecker的24个色块，评价色彩还原度` |
| 多任务 | `sample_images/2014_3_composite.JPG` | `请同时测量边缘MTF和20级灰阶，并分别总结` |
| 独立 SFR 图 | `sample_images/sfr180T.bmp` | `请测一下这张图的MTF` |
| 线性灰阶 | `sample_images/stepchartT.bmp` | `请分析这张图的灰阶响应` |
| 畸变 | `sample_images/distortion_DIS180R1.jpg` | `请分析这张图的畸变` |

## MTF 方向说明

程序内部区分“MTF 方向”和“刃边方向”：

- 水平 MTF：需要框竖直刃边。
- 竖直 MTF：需要框水平刃边。
- 不指定方向时，默认同时测水平和竖直方向。
- 不指定视场位置时，默认测画面中心斜正方形。
- 输入“边缘、四周、边角、周边、视场边缘、画面边缘”时，会优先测画面边缘的多个斜正方形。

## 人工修正 ROI

如果自动识别失败或框选不满意，页面会保留对应任务的原图，并开放人工修正入口。

支持人工修正：

- MTF/SFR：可画 1 个或多个跨斜边的窄框。
- ColorChecker：必须画 24 个色块中心 ROI。
- 20 级灰阶：必须画 20 个灰阶方块中心 ROI。
- 畸变：画 1 个包含完整网格主体的 ROI，尽量避开外侧斜纹、箭头和背景。

页面中有“人工框选说明”按钮。进入人工修正时会自动弹出说明，告诉你每类任务应该框哪里，并给出示意图和失败原因。

## 离线自测

如果只是想确认程序文件和测试图片是否完整，不需要 API Key，可以运行：

```text
run_smoke_test.bat
```

或者：

```powershell
python smoke_test.py
```

自测会检查：

- 本地 `py_imaging_quality` 是否能被找到。
- `2014_3_composite.JPG` 是否能识别 20 级灰阶。
- `2014_3_composite.JPG` 是否能识别边缘 MTF ROI。
- `colorD65.bmp` 是否能识别 24 色块。
- `distortion_DIS180R1.jpg` 是否能识别畸变网格线。

## 命令行用法

图形页面更适合展示和验收。命令行用于调试：

```powershell
python task3_agent.py "请测量画面边缘的MTF" "sample_images\2014_3_composite.JPG"
```

如果没有设置环境变量，程序会提示输入 DeepSeek API Key。

也可以临时设置环境变量：

```powershell
$env:DEEPSEEK_API_KEY="你的Key"
python task3_agent.py "请分析色彩还原度" "sample_images\colorD65.bmp"
```

## 推送到 GitHub

先在 GitHub 上新建一个空仓库，然后在本目录打开 PowerShell：

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\push_to_github.ps1 -RepoUrl "https://github.com/你的用户名/你的仓库名.git"
```

脚本会自动：

1. 初始化 git 仓库。
2. 添加全部文件。
3. 提交 commit。
4. 设置远程仓库。
5. 推送到 `main` 分支。

如果 GitHub 要求登录，请按 Git Credential Manager 的提示完成浏览器登录，或使用 GitHub Personal Access Token。

## 常见问题

### 1. 页面打不开

确认终端里显示了：

```text
Running on http://127.0.0.1:7860
```

然后手动打开 `http://127.0.0.1:7860`。

### 2. 安装依赖失败

通常是网络或 pip 源问题。可以换国内源：

```powershell
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 3. DeepSeek Key 验证失败

检查 Key 是否有效，网络是否能访问 DeepSeek API。不要把 Key 发给别人或提交到 GitHub。

### 4. 自动框选失败

这不一定是程序错误，可能是：

- 测试卡太小或被裁切。
- 图像模糊、过曝、反光。
- 色卡/灰阶/斜方块不是程序支持的标准版式。
- 背景中存在大量相似方块造成干扰。

此时可以进入人工修正 ROI，并查看页面弹出的框选说明。

### 5. MTF 数值与 MATLAB 手动画框不同

MTF 对 ROI 很敏感。要严格对比，需要保证：

- Python 和 MATLAB 使用完全相同的 ROI。
- 方向一致。
- Gamma、单位、采样区域一致。
- 图像预处理一致。

本程序会生成 ESF/MTF 曲线，方便与 MATLAB 曲线一起核对。
