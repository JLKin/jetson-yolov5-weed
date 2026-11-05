# TensorRT 加速与复杂画面识别

目标是分别验证速度和识别效果：训练中见过相近的拍摄条件，推理中保留质量信息，部署时确认 TensorRT 没有带来不可接受的精度下降。当前还没有真实杂草数据和 C10pro 视频，以下功能已实现，实际阈值和精度需用现场数据确定。

## 1. 训练集增强

在原始数据完成 train/val/test 划分后执行：

```bash
python -m weed_detector augment-data --data datasets/weeds/data.yaml --output datasets/weeds_aug --copies 2 --severity 0.4
python -m weed_detector train --data datasets/weeds_aug/data.yaml --hyp configs/hyp.weeds.yaml --model yolov5s --epochs 100 --batch 8 --device 0
```

每张原始训练图保留，再生成 2 张变体；val/test 原始图片和标注逐字节复制，不加训练增强。固定 `--seed` 可复现变体；`augmentation.json` 记录来源、条件、强度和随机种子。先划分再增强，避免同一图片的不同版本进入训练和验证两侧。

支持的条件：

| 条件 | 实现及边界 |
| --- | --- |
| `glare` | 局部椭圆高光模拟；不能完整模拟叶面反射、镜头眩光或偏振现象 |
| `gaussian-blur` | 模拟失焦模糊 |
| `motion-blur` | 随机方向线性运动模糊 |
| `mirror-horizontal` | 左右镜像，同时转换标注中心横坐标 |
| `mirror-vertical` | 上下镜像，同时转换标注中心纵坐标；默认未选入，需要时显式指定 |
| `dark` | Gamma 变换模拟暗画面 |
| `jpeg` | 模拟压缩细节损失 |

默认随机选择其中除上下镜像外的条件，不保证每张图包含全部条件。需要集中补充某种情况时显式指定，例如：

```bash
python -m weed_detector augment-data --data datasets/weeds/data.yaml --output datasets/weeds_blur --conditions gaussian-blur motion-blur --copies 2 --severity 0.3
```

`severity` 范围为 (0,1]，数值只是本项目的模拟强度，不是物理反光率或模糊等级。建议先查看样本：过强的增强可能遮蔽杂草，使保留的标注失去合理视觉依据。正式训练应补充真实反光、风吹叶片、云台运动、逆光和阴影图像。

`configs/hyp.weeds.yaml` 另外提供在线 HSV、轻度旋转、缩放、左右翻转及 Mosaic 配置。YOLOv5 在线增强会同步处理检测框；这些参数属于起点，需要验证集比较。[YOLOv5 官方超参数说明](https://docs.ultralytics.com/yolov5/tutorials/hyperparameter-evolution/)

## 2. 推理质量记录与可选处理

默认保持输入像素不变，额外记录质量指标。可开启指定的镜像校正、CLAHE 局部对比度调整或 Gamma 亮度调整：

```bash
# 明确知道相机画面左右镜像时使用
python -m weed_detector infer --weights weights/best.pt --source /data/video.mp4 --mirror horizontal

# 测试低对比度画面
python -m weed_detector infer --weights weights/best.pt --source /data/video.mp4 --enhance clahe

# Gamma 小于 1 提亮，大于 1 压暗
python -m weed_detector infer --weights weights/best.pt --source /data/video.mp4 --enhance gamma --gamma 0.8
```

镜像在识别前执行，检测框随后映射回**原始输入画面坐标**；保存的带框画面仍是原始输入，便于与地面站和后续控制程序对齐。镜像校正方向需要依据实际相机配置指定，程序不会猜测。

JSONL schema 已升级为 2，增加 `quality`、`quality_flags` 和 `preprocessing_ms`；CSV 同样增加质量标记和预处理耗时。原有字段保留。

| 指标／标记 | 含义 |
| --- | --- |
| `sharpness_laplacian` / `blur_risk` | 拉普拉斯方差较低；可能模糊，也可能画面本身缺少纹理 |
| `highlight_fraction` / `highlight_risk` | 三个色彩通道均接近饱和的像素比例；可能过曝、高光或白色背景 |
| `dark_fraction` / `dark_risk` | 暗像素占比较高 |

质量分析会将大图缩小到最长边 640 像素，阈值仍会受分辨率、场景和噪声影响。默认 `--blur-threshold 60 --highlight-threshold 0.05` 是待标定起点，不是可信度概率。质量异常帧继续识别并记录，不会自动当作“没有杂草”。可用 `--no-quality` 关闭诊断以测量其开销。

对比度处理可能同时放大噪声；Gamma 也无法恢复已经饱和的细节。严重反光或模糊时需要从曝光、快门、对焦、云台稳定性和采样条件改善输入。不要把锐化后看似清晰的画面当作已经恢复真实纹理。

## 3. 分场景压力测试

```bash
python -m weed_detector stress-test --weights weights/best.pt --data datasets/weeds/data.yaml --split val --imgsz 640
```

生成 baseline、高光、失焦模糊、运动模糊和左右镜像的独立评估结果，以及总表 `stress-report.json`。每个条件使用相同的原始划分、固定随机种子和同步变换的标注；原始数据不会改变。增加条件可使用 `--conditions`，例如追加 `mirror-vertical dark jpeg`。

这些是受控合成测试。真实现场测试集还应分为正常、高光、运动模糊、失焦等组分别报告 precision、recall、mAP；要重点关注小杂草漏检。开发时使用 val 调整增强，最后在未参与调整的 test 上确认效果。

## 4. TensorRT FP16 部署与精度回归

在 Jetson 上补齐其 JetPack 5.1.3 对应的 TensorRT 8.5 Python 环境后，构建固定输入尺寸、batch=1 的 FP16 引擎：

```bash
python -m weed_detector export --weights weights/best.pt --format engine --device 0 --half --imgsz 640 --workspace 2
```

使用终端实际输出目录中的 `model.engine` 和 `classes.yaml`。导出清单会记录 GPU、CUDA、PyTorch、TensorRT、L4T 和文件 SHA256。引擎在目标 Jetson 上构建，避免跨设备及运行库不匹配。

低精度可能改变数值和检测结果，所以需要使用相同数据检查。[NVIDIA 精度说明](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/accuracy-considerations.html)

```bash
python -m weed_detector compare-models --reference weights/best.pt --candidate runs/export/weed/model.engine --data datasets/weeds/data.yaml --split val --imgsz 640 --max-map-drop 0.01 --max-recall-drop 0.02 --require-pass
```

两侧都通过本项目实际推理路径：相同的方形 letterbox、batch=1、置信度下限 0.001、NMS IoU=0.6 和原图坐标。PyTorch 使用 FP32，TensorRT 使用引擎的原生精度。这里没有使用上游 `.pt` 的矩形批处理，以避免预处理差异污染对比。

`0.01` 指 mAP50-95 的绝对下降 1 个百分点，`0.02` 指 recall 下降 2 个百分点；只是可配置的回归门槛。`--require-pass` 超限返回非零退出码。两个都很差的模型也可能通过相对比较，绝对精度仍需单独验收。标准 `evaluate` 继续提供上游评估和图表；对照结果中的 protocol 字段区分两种评估流程。

当前提供 FP32 / FP16 路径。INT8 需要有代表性的现场校准数据及小目标精度验证，当前 CLI 未开放 INT8 导出。

## 5. 性能测试

```bash
python -m weed_detector benchmark --weights weights/best.pt --source datasets/weeds/images/val --imgsz 640 --iterations 100 --name pt
python -m weed_detector benchmark --weights runs/export/weed/model.engine --data runs/export/weed/classes.yaml --source datasets/weeds/images/val --imgsz 640 --iterations 100 --name trt
```

两次使用相同参数、设备功耗模式和测试图片，串行执行，并避免同时训练。报告包含预热次数、平均／P50／P95 延迟、处理吞吐量和环境版本。加入实际使用的 `--mirror`、`--enhance`、`--gamma` 才能计入对应处理成本。

`pipeline_fps` 衡量内存中图像的质量检查、预处理、模型和后处理，**不含相机取流／解码和结果写入**；不是相机端到端 FPS。真实流程另用 `infer` 的运行摘要与现场延迟测量。实际 TensorRT 加速比需要在用户 Jetson 上获得，仓库不预设某个 FPS。
