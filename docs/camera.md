# C10pro 视频输入

本项目接收解码图像帧后执行 YOLOv5 检测。相机厂家、网络连接方式和确切 RTSP 地址尚待用户确认，因此不把某个示例 IP 当作设备实测配置。当前不发送云台控制指令。

## 接入顺序

1. 按相机／图传说明连接网口、供电和接收端，确认 Jetson 能访问相机所在网段。
2. 从地面站或厂家文档获取真实的 RTSP 地址、端口、认证信息、主／副码流及编码格式。
3. 先用 VLC 或 FFmpeg/ffprobe 验证码流，再运行检测。不要把包含密码的地址提交到 GitHub。
4. 先用 `--max-frames 100` 做短时验证，再持续运行。

Linux：

```bash
export CAMERA_SOURCE='rtsp://CAMERA_IP:554/STREAM_PATH'
ffprobe -v error -rtsp_transport tcp -show_entries stream=codec_name,width,height,r_frame_rate "$CAMERA_SOURCE"
bash scripts/run_camera.sh --max-frames 100
```

PowerShell：

```powershell
$env:CAMERA_SOURCE = 'rtsp://CAMERA_IP:554/STREAM_PATH'
.\.venv\Scripts\python.exe -m weed_detector infer --weights weights/best.pt --max-frames 100
```

地址是占位符。应用生成的 `run.json` / JSONL 会隐藏 URL 的用户名、密码、路径和查询参数；底层解码器或手动执行的诊断命令可能仍输出地址，分享日志前检查。

## 延迟、断流和录像

- RTSP 默认使用 OpenCV 的 FFmpeg 后端及 TCP 传输。当前实现未启用 Jetson 专用 GStreamer 硬件解码流水线。
- 读取线程只保留最新帧，推理不会不断追赶旧帧。输入帧率高于推理速度时会跳帧，JSONL 保留实际接收帧序号。
- `--timeout 5 --reconnects 5` 控制网络打开／读取超时与连续失败重试次数；恢复读取后重置连续失败次数。耗尽重试则非零退出，并写失败摘要。
- USB 相机采用系统驱动，阻塞行为由驱动决定；网络超时参数针对 FFmpeg 网络源。
- `--no-visual` 可减少编码和磁盘开销；`--segment-seconds 60` 按输出帧率分段写录像。固定帧率录像不是现场严格时间轴回放，精确事件时间使用 JSONL。
- 持续写入不会自动轮换删除旧数据，需要根据存储容量安排归档。

没有相机时，图片目录和本地视频走同一模型和结果模块，可以先验证系统功能。真实 C10pro 的网络恢复、持续延迟、编码兼容性需要实机联调。
