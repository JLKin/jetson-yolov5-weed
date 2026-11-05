# Jetson Orin Nano 部署

## 当前设备识别

用户提供的 `/etc/nv_tegra_release` 为 **R35 / REVISION 5.0**，即 L4T 35.5.0，对应 JetPack 5.1.3。设备树确认 `NVIDIA Orin Nano Developer Kit`，系统 Python **3.8.10**；`nvidia-l4t-core` 和 `nvidia-l4t-cuda` 均为 35.5.0，APT 尚未安装 `libnvinfer8`。该版本官方计算栈为 Ubuntu 20.04、CUDA 11.4、TensorRT 8.5.2；这些是配套版本，不表示用户设备已装齐。终端中 `(base)` 表示激活了 Conda，本项目 Jetson 安装使用 `/usr/bin/python3`。

```bash
cat /etc/nv_tegra_release
cat /proc/device-tree/model
/usr/bin/python3 --version
dpkg-query -W nvidia-l4t-core nvidia-l4t-cuda libnvinfer8
```

当前按现有 JetPack 5.1.3 接入，不执行系统刷机或升级。它不提供新版本 Orin Nano Super 模式；若后续需要该模式，应先确认实际模块／载板，再按 NVIDIA 或载板厂家文档规划系统与固件升级。不要把 JetPack 5→6 当成普通 Python 包升级。

## 1. 拉取私有 GitHub 仓库

```bash
sudo apt-get update
sudo apt-get install -y git gh
gh auth login
gh auth setup-git
git clone --recurse-submodules https://github.com/JLKin/jetson-yolov5-weed.git
cd jetson-yolov5-weed
```

登录具有仓库权限的账户。若发行版源不提供 `gh`，按 [GitHub CLI 官方安装说明](https://github.com/cli/cli/blob/trunk/docs/install_linux.md) 安装，或使用已配置 SSH 密钥的 Git 克隆地址。

## 2. 系统基础包

```bash
sudo apt-get install -y python3-venv python3-dev python3-pip python3-opencv \
  libopenblas-dev libopenmpi-dev libjpeg-dev zlib1g-dev build-essential ffmpeg unzip
```

CUDA、cuDNN、TensorRT 属于 JetPack。`nvidia-l4t-cuda` 存在不等于完整 CUDA 开发工具已安装。按 [JetPack 5.1.3 安装说明](https://docs.nvidia.com/jetson/jetpack/5.1.3/install-jetpack/index.html) 补齐与当前 L4T 匹配的包：先确认 NVIDIA APT 源仍对应 r35.5，不要混装 r36/r39 包。当前仓库不会自动修改 NVIDIA 系统源。

```bash
cat /etc/apt/sources.list.d/nvidia-l4t-apt-source.list
apt-cache policy nvidia-jetpack libnvinfer8
```

确认源和候选包对应 JetPack 5.1.3 后，可按官方方法安装完整计算栈（会占用较多磁盘空间）：

```bash
sudo apt-get install nvidia-jetpack
```

它是在匹配的 r35.5 源上补齐 JetPack 5.1.3 软件包，不是系统跨版本升级。安装后检查 `/usr/local/cuda/bin/nvcc --version` 以及 `dpkg-query -W libnvinfer8 python3-libnvinfer`。

## 3. GPU PyTorch / torchvision

先尝试复用可用的系统环境：

```bash
bash scripts/setup_jetson.sh
```

脚本先验证 GPU torch 和 GPU torchvision NMS。若失败且设备确认是 L4T 35.5.0、系统 Python 3.8，可运行本仓库的专用安装配方：

```bash
bash scripts/install_torch_jetpack5.sh
bash scripts/setup_jetson.sh
source .venv/bin/activate
python -m weed_detector doctor --require-cuda
```

该配方使用 NVIDIA 官方 `torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64.whl`，配套从源码编译 torchvision v0.15.1。JetPack 5 固定 NumPy 1.23.5，避免 TensorRT 8.5 的旧 dtype API 与 NumPy 1.24+ 不兼容。已可用的 GPU 框架会保留；安装发生在项目 `.venv`。编译需要 CUDA 开发工具，默认 `MAX_JOBS=2`，可按可用内存调整。这个目标板配方尚未在用户设备执行，最终以 `doctor` 的 CUDA 计算和 NMS 检查结果为准。

JetPack 6/7 使用不同的 CUDA/Python/TensorRT 组合，**不能使用上述 cp38 wheel**。应按 [NVIDIA PyTorch 安装说明及兼容表](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html) 配好 GPU torch/torchvision，再运行 `setup_jetson.sh`。主安装器将应用依赖与 GPU 框架分离，并保留系统 OpenCV；NumPy 被约束在 1.x 以兼容 JetPack 5/6 的系统 OpenCV。

## 4. 接收正式模型并验证

收到电脑端生成的 `weed-model.zip` 后：

```bash
mkdir -p weights
unzip -n ~/weed-model.zip -d weights
python -c 'import hashlib,json,pathlib; p=pathlib.Path("weights"); m=json.loads((p/"manifest.json").read_text()); assert hashlib.sha256((p/m["file"]).read_bytes()).hexdigest()==m["sha256"]; print("SHA256 OK")'
python -m weed_detector infer --weights weights/best.pt --source /absolute/path/weed-test.jpg --device 0
```

先用已知测试图片检查框、类别、置信度。仓库没有预装正式杂草权重；`git clone` 本身不能代替训练和模型传输。

## 5. C10pro 和结果

确认 Jetson 与相机／图传接收端的网络连通，并从厂家资料或地面站获得真实 RTSP 地址：

```bash
export CAMERA_SOURCE='rtsp://CAMERA_IP:554/STREAM_PATH'
bash scripts/run_camera.sh --max-frames 300
```

命令中的 IP 和路径是占位符，不能原样使用。去掉 `--max-frames` 可持续运行，Ctrl+C 会结束并关闭结果文件。详细说明见 [相机接入](camera.md)。长时间运行需要监控存储空间；分段录像不会自动删除历史文件。

## 6. TensorRT 加速与板端训练

先安装该 JetPack 自带的 TensorRT Python 包，确认 `python -c 'import tensorrt; print(tensorrt.__version__)'` 成功，再执行：

```bash
python -m pip install onnx
python -m weed_detector export --weights weights/best.pt --format engine --device 0 --half --imgsz 640 --workspace 2
python -m weed_detector infer --weights runs/export/weed/model.engine --data runs/export/weed/classes.yaml --source /absolute/path/weed-test.jpg --imgsz 640
```

以终端输出的导出目录为准。若内存不足，将 workspace 降到 1 GiB；必须在目标设备上构建引擎。先与 `.pt` 在同一批测试图片上核对精度，再测处理帧率；没有实机测试时不承诺 FPS。

必要时在 Jetson 上训练：

```bash
bash scripts/train_jetson.sh /absolute/path/data.yaml --batch 2 --epochs 100
```

监测资源可使用 JetPack 提供的 `tegrastats`。性能模式编号依设备配置变化，先查看 `sudo nvpmodel -q --verbose`，不要直接套用其他设备的模式编号。

## 官方依据

- [JetPack 5.1.3 版本与计算组件](https://developer.nvidia.com/embedded/jetpack-sdk-513)
- [NVIDIA JetPack 5.1.1 系列 PyTorch 下载目录](https://developer.download.nvidia.com/compute/redist/jp/v511/pytorch/)
- [torch / torchvision 版本对应](https://github.com/pytorch/vision#installation)
- [Orin Nano Super 模式与 JetPack 6.2](https://developer.nvidia.com/embedded/jetpack-sdk-62)
