# 工程验证记录

日期：2026-11-05。以下测试使用合成图像和合成视频，**不代表杂草识别精度**。

## 本机环境

- Windows 11，RTX 5060 Laptop GPU 8GB。
- 项目虚拟环境 Python 3.12.15，torch 2.13.0+cu132，torchvision 0.28.0+cu132。
- OpenCV 5.0.0，NumPy 2.5.2，ultralytics 工具包 8.4.192。
- YOLOv5 源码固定提交：`9777cd00e86382dfa5a12e6244ec4f609bc4774c`。

## 已执行

| 检查 | 结果 |
| --- | --- |
| `doctor --require-cuda` | CUDA 矩阵计算及 torchvision GPU NMS 通过 |
| 14 个单元测试 | 全部通过；覆盖无标注、非法框、数据泄漏、分组划分、零检测帧、坐标保存、URL 脱敏和有限重连 |
| YOLOv5n 从零训练 | 12 张训练 / 4 张验证图、1 epoch、320 输入、batch 4；生成 best.pt 和 last.pt |
| YOLOv5s 预训练微调 | 成功获取官方预训练权重；同一合成数据、1 epoch、320 输入、batch 2；生成 best.pt 和 last.pt |
| 独立 test 评估 | 4 张独立合成测试图；生成 metrics.json 与评估图表 |
| 图片推理 | 4 张图成功写出可视化、JSONL、CSV 和完成摘要 |
| 视频推理 | 20 帧全部处理；2 段 MP4 均可解码，每段 10 帧 |
| 输出一致性 | CSV 行数与 JSONL 检测目标总数一致；全部检测框位于原图 320×240 范围内 |
| ONNX 导出 | 成功导出，onnx.checker 验证通过 |
| 模型打包 | ZIP 中 best.pt 与 manifest.json 的 SHA256 一致 |
| Python 3.8 语法 | 项目 Python 文件均通过 3.8 语法检查 |
| ARM64 / Python 3.8 依赖 | pip 跨平台 dry-run 成功解析运行时依赖和 NumPy 1.23.5 约束；不是实机安装验证 |
| 安装脚本 | Bash 语法和 PowerShell 解析检查通过 |

测试中发现并修复了 Windows 中文 YAML 路径的默认编码问题，以及新版 PyTorch inference_mode 下检测框原地修改的问题。输出压力测试使用 `--conf 0.001` 验证有目标时的数据和坐标；该阈值不用于正式杂草检测。

## 用户设备与待实测项

用户终端输出已确认：NVIDIA Orin Nano Developer Kit、L4T 35.5.0、系统 Python 3.8.10；APT 尚无 `libnvinfer8`。因此提供 JetPack 5.1.3 专用安装配方，但以下仍需后续验证：

- 在该 Jetson 上安装 GPU PyTorch/torchvision、OpenCV 和 TensorRT；通过 doctor。
- 电脑训练出的正式权重在板端加载，及 TensorRT 导出、FP16 精度一致性。
- C10pro 实际 RTSP 地址、解码、断流恢复、长时间存储和延迟。
- 杂草数据集的标注质量、正式训练、独立测试集精度、现场泛化表现。
- 目标 Jetson 的实时处理帧率、内存、功耗和温度。

原始日志保存在本地 `work/logs/`，未上传 GitHub。工程测试模型位于本地 `runs/`，同样未上传，以免被误当成杂草模型使用。
