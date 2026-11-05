# YOLOv5 杂草识别 · 电脑训练 / Jetson 部署

使用 **Ultralytics 原版 YOLOv5**，默认 `yolov5s`、640 输入。先在电脑训练；需要时可切换到 Jetson 训练。支持图片、视频、USB 相机和 RTSP 输入，后续可接入 C10pro 的实际视频流。

源码固定在子模块提交 `9777cd00e86382dfa5a12e6244ec4f609bc4774c`。`ultralytics` 仅提供上游所需工具函数，模型构建与训练使用 `third_party/yolov5`。

## 当前状态

- 已提供数据检查、分组划分、训练、验证、推理、ONNX / TensorRT 导出、模型打包和部署脚本。
- 本机 RTX 5060 Laptop GPU 已通过 CUDA 和 torchvision NMS 检查；工程验证详情见 [验证记录](docs/validation.md)。
- 用户设备输出为 **L4T 35.5.0，对应 JetPack 5.1.3**。支持其 Python 3.8 环境，另保留 JetPack 6 的配置说明。
- 杂草数据集、真实 C10pro 码流和目标 Jetson 的实机验证尚未提供／完成。仓库没有训练好的杂草权重。

## 快速开始

克隆时必须包含子模块。仓库默认私有，在 Jetson 上先使用有仓库权限的 GitHub 账户登录 GitHub CLI，或配置 SSH 密钥。

```bash
gh auth login
gh auth setup-git
git clone --recurse-submodules https://github.com/JLKin/jetson-yolov5-weed.git
cd jetson-yolov5-weed
```

环境安装：

- **当前电脑**：[Windows 安装和训练](docs/windows.md)。
- **当前 Jetson（JetPack 5.1.3）**：[Jetson 安装和部署](docs/jetson.md)。
- **相机**：[C10pro 接入与断流排查](docs/camera.md)。
- **反光、模糊、镜像与加速验收**：[复杂画面处理、训练增强、TensorRT 精度与延迟测试](docs/robustness.md)。

安装后，在项目根目录、已激活的项目虚拟环境中运行：

```bash
python -m weed_detector doctor --require-cuda
python -m weed_detector check-data --data datasets/weeds/data.yaml
python -m weed_detector train --data datasets/weeds/data.yaml --model yolov5s --epochs 100 --batch 8 --device 0
python -m weed_detector evaluate --data datasets/weeds/data.yaml --weights runs/train/weed/weights/best.pt --split test
python -m weed_detector infer --weights runs/train/weed/weights/best.pt --source datasets/weeds/images/test
```

`datasets/weeds/data.yaml` 是未来真实数据集的位置，需先放入数据或按下文准备；首次训练会下载官方预训练权重。首次运行目录为 `runs/train/weed`，后续运行会增加编号，以终端实际输出路径为准。

## 数据集

使用 YOLO detection 格式：

```text
datasets/weeds/
├── data.yaml
├── images/train/*.jpg
├── images/val/*.jpg
├── images/test/*.jpg
├── labels/train/*.txt
├── labels/val/*.txt
└── labels/test/*.txt
```

每张图片对应同名 TXT，每行 `class_id center_x center_y width height`，坐标归一化至 0～1。纯背景图片使用空 TXT；缺标注会被明确报错。类别编号从 0 连续递增。

```yaml
path: .
train: images/train
val: images/val
test: images/test
names:
  0: weed
```

`path` 相对于 YAML 文件所在目录解析。类别名称应与真实标注一致；若区分不同杂草，增加相应类别。仅有图片或分类标签不能直接用于矩形框目标检测训练。COCO、VOC 或分割标注需要按实际数据结构转换。

已有 train/val/test 划分时直接使用。未划分的图片和标签可复制生成默认约 80/10/10 的划分：

```bash
python -m weed_detector prepare-data --images /data/raw/images --labels /data/raw/labels --output datasets/weeds --classes weed
```

对于同一段视频或同一地块连续拍摄的数据，应按场景分组，例如文件名 `field01_frame001.jpg`：

```bash
python -m weed_detector prepare-data --images /data/raw/images --labels /data/raw/labels --output datasets/weeds --classes weed --group-regex '(field[0-9]+)_'
```

每个组只进入一个划分，至少需要 3 个组。验证器检测跨划分的完全重复文件；近似图片仍需按场景分组，不能依赖文件哈希识别。已有测试集不要用于调整训练参数。

## 训练、结果与迁移

训练产出：`weights/best.pt`、`weights/last.pt`、`results.csv`、训练曲线、混淆矩阵及本项目保存的数据检查报告。独立评估输出 `metrics.json`，包含 precision、recall、mAP50、mAP50-95；CLI 默认评估 test，没有测试集时明确指定 `--split val`。

```bash
# 中断后从 last.pt 恢复，沿用该次训练的参数
python -m weed_detector train --resume runs/train/weed/weights/last.pt

# 打包到 Jetson；压缩包只包含模型、类别和 SHA256 清单
python -m weed_detector package-model --weights runs/train/weed/weights/best.pt --output work/weed-model.zip

# Jetson 上训练的低内存起点，额外参数可覆盖默认值
bash scripts/train_jetson.sh /absolute/path/data.yaml --batch 2
```

发生显存不足时先降低 `--batch`（8→4→2），再考虑 `--model yolov5n` 或调整输入尺寸。降低尺寸可能损失小杂草的检出率，需要重新评估。单卡训练占用 GPU 时不要同时运行实时推理。

## 推理输入和结果

```bash
python -m weed_detector infer --weights weights/best.pt --source /data/weed.jpg
python -m weed_detector infer --weights weights/best.pt --source /data/video.mp4 --max-frames 300
python -m weed_detector infer --weights weights/best.pt --source 0
```

RTSP 地址可通过环境变量 `CAMERA_SOURCE` 提供，详见相机文档。推理默认无窗口，适合 SSH / 无显示器运行。

每次运行创建独立的 `runs/infer/weed*`：

| 文件 | 内容 |
| --- | --- |
| `detections.jsonl` | 每处理一帧写一条 JSON，零检测帧也保留 |
| `detections.csv` | 每个检测目标一行，便于分析与外部程序读取 |
| `frame_*.jpg` / `annotated_*.mp4` | 带框图片／分段视频，可用 `--no-visual` 关闭 |
| `summary.json` | 运行状态、累计逐帧检测数、平均耗时、处理帧率 |
| `run.json` | 类别表、阈值、模型路径及脱敏输入信息 |

新增 `augment-data`、`stress-test`、`compare-models` 和 `benchmark` 命令。推理默认记录模糊／高光质量风险；可选 `--mirror horizontal`、`--enhance clahe` 或 `--enhance gamma --gamma 0.8`，具体用途和边界见复杂画面处理文档。JSONL schema=2 保留原字段，并增加质量标记与预处理耗时。

`bbox_xyxy` 是原始图像像素坐标；`captured_at` 是 Jetson/电脑接收帧的 UTC 时间，**不是相机曝光时间**。本地视频另有按帧率换算的 `media_time_ms`，实时流为 null。实时输入只保留最新帧，帧序号可以跳跃。录像为固定输出帧率，丢帧时播放速度可能与现场不同；JSONL 时间戳是事件对齐依据。

类别表保留中文，图中绘制 `#类别编号 置信度` 以避免字体缺字。累计检测数不是去重后的植株总数；当前未实现跟踪、面积估算或云台运动控制。

## 模型加速

```bash
python -m pip install onnx
python -m weed_detector export --weights weights/best.pt --format onnx --imgsz 640

# 在目标 Jetson 上执行，使用其已安装的 TensorRT 8/10
python -m weed_detector export --weights weights/best.pt --format engine --device 0 --half --imgsz 640 --workspace 2
python -m weed_detector infer --weights runs/export/weed_001/model.engine --data runs/export/weed_001/classes.yaml --source /data/video.mp4 --imgsz 640
```

导出目录会自动编号，应替换为终端实际路径。导出的 engine 是 batch=1、固定方形尺寸，推理尺寸须一致。TensorRT 引擎应在目标 Jetson 上构建，不能直接把电脑生成的 engine 当作通用模型。当前适配 TensorRT 8～10，目标板的导出、性能及与 `.pt` 的精度一致性需实测。

## 开发与更新

```bash
python -m unittest discover -s tests -v
python scripts/make_smoke_data.py --output work/smoke-data
python -m weed_detector train --data work/smoke-data/data.yaml --model yolov5n --scratch --epochs 1 --batch 4 --imgsz 320 --name smoke

git pull --ff-only
git submodule update --init --recursive
```

合成数据只是工程测试，不能作为杂草识别数据。数据集、模型权重、运行输出及相机凭据均由 `.gitignore` 排除。GitHub 克隆只获取代码；正式权重需要另行传输。

上游 YOLOv5 采用 AGPL-3.0，许可证保留在子模块中；本项目代码同样按 [AGPL-3.0](LICENSE) 提供。依赖各自保留其许可证。
