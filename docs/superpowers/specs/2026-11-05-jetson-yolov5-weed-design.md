# 杂草识别训练与 Jetson 部署设计

用户已确认：先在当前电脑训练，保留 Jetson 训练作为备选。使用原版 Ultralytics YOLOv5 检测模型，默认 yolov5s / 640，不能用 YOLOv5u 或 YOLO26 替代。

## 流程与边界

1. 输入：YOLO detection 图片与矩形框标注，类别由数据集提供。先校验再训练；原始数据不修改，未分割数据可复制分割；已有划分保留，按场景分组由使用者指定以避免相邻帧泄漏。
2. 训练：电脑 CUDA 优先，提供低批量的 Jetson 配置。调用固定版本上游 train.py，保存 best.pt / last.pt、CSV 和图表，支持断点续训；提供独立 val/test 评估入口。
3. 推理：图片、图片目录、本地视频、USB 相机、RTSP 视频流共用模型与结果输出。相机以解码后的帧输入；没有真实 C10pro 时以文件验证，实际厂家、码流与协议仍需设备资料确认。
4. 结果：原图坐标系检测框、类别、置信度、UTC 接收时间、帧序号、视频时间、推理耗时；JSONL 每帧一行（包括零检测），CSV 每目标一行；可视化图片／分段视频和 summary.json。逐帧检测数不能当作去重植株数量。
5. 加速：PyTorch 为基线；导出 ONNX，再在目标 Jetson 上构建 TensorRT。固定上游版本并检查 TensorRT 接口兼容性；模型旁保存类别元数据。目标板已确认 Orin Nano Developer Kit、L4T 35.5.0 / JetPack 5.1.3、Python 3.8.10；APT 缺少 libnvinfer8，安装和运行尚未实测。
6. 部署：环境诊断、Windows / Jetson 安装脚本、GitHub clone / pull 文档。数据、训练输出、权重、凭据默认不进入 Git；训练权重通过文件传输或显式发布的 Release 分发。GitHub 默认私有仓库，沿用已缓存账户。

## 组件

- weed_detector CLI：check-data / prepare-data / train / evaluate / export / infer / doctor / package-model。
- dataset：YOLO 格式校验、显式空标注、类别范围、坐标边界、划分重复检测。
- backend：固定 YOLOv5 的 DetectMultiBackend、letterbox、NMS、坐标恢复。
- sources：文件与实时视频读取、超时、有限重连、实时流只保留最新帧。
- results：结构化结果、图像／分段视频、运行摘要，失败也记录状态。
- scripts：安装、模型传输准备、运行演示及测试。

## 异常与验收

无模型、缺标注、空划分、未知类别和不可读输入明确报错；运行目录独立，避免覆盖旧结果；日志隐藏 RTSP 用户名、密码和查询参数。网络断流有时间限制并重连，超出限制非零退出。

校验器与坐标还原使用单元测试；本机 GPU 用小规模合成数据做真实训练、验证及图片／视频推理冒烟测试，只证明流程可用，不代表杂草识别精度。正式训练、精度报告、C10pro 联调和 Jetson FPS 等待数据集和目标硬件。保持实测项与待验证项清楚区分。

## 实现顺序

1. 固定上游 YOLOv5 与依赖策略，建立项目入口。
2. 数据校验／准备、训练和评估封装。
3. 推理输入、结构化输出和模型导出／打包。
4. 安装脚本、中文部署文档、CI。
5. 本机测试、代码审查、提交并上传 GitHub，核验远程。

参考：
- https://github.com/ultralytics/yolov5
- https://docs.ultralytics.com/yolov5/tutorials/train-custom-data/
- https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html
