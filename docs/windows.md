# Windows 本机训练

已检测到 RTX 5060 Laptop GPU 8GB，以及 `D:\Anaconda3\envs\yolo26\python.exe` 中可用的 GPU PyTorch。本项目在自己的 `.venv` 内安装依赖，通过 `--system-site-packages` 复用 GPU 框架，原环境不会被卸载或覆盖。

在项目根目录的 PowerShell 中：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_windows.ps1 -PythonExe 'D:\Anaconda3\envs\yolo26\python.exe'
.\.venv\Scripts\python.exe -m weed_detector doctor --require-cuda
.\.venv\Scripts\python.exe -m weed_detector check-data --data datasets/weeds/data.yaml
.\.venv\Scripts\python.exe -m weed_detector train --data datasets/weeds/data.yaml --model yolov5s --epochs 100 --batch 8 --workers 0 --device 0
```

`yolo26` 是本机已有 Conda 环境的名字；本项目实际训练的网络仍是原版 YOLOv5。换电脑时，用该电脑已装好 GPU PyTorch 的解释器替换 `-PythonExe`。新环境按 [PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/) 安装匹配显卡的 torch 和 torchvision，然后运行安装脚本。

所有命令也可先激活 `.venv` 再用 `python`。CLI 在 Windows 上自动启用 UTF-8，支持项目和数据集路径包含中文。直接调用上游脚本时请自行使用 `python -X utf8`。

正式训练前需要真实图片和检测框标注。建议从 100 epochs、batch 8 开始，依据验证集曲线决定是否调整；没有数据集之前不报告杂草精度。Windows 默认 workers=0，避免多进程数据加载问题。

传输训练权重示例：

```powershell
.\.venv\Scripts\python.exe -m weed_detector package-model --weights runs/train/weed/weights/best.pt --output work/weed-model.zip
scp work/weed-model.zip jlking@JETSON_IP:~/weed-model.zip
```

把 `JETSON_IP` 替换成设备实际 IP；如果 SSH 未配置，也可通过 U 盘传输该压缩包。
