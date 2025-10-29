# FireRedASR 批量音频处理指南

本文档详细介绍如何使用FireRedASR进行批量音频处理，包括WebM转WAV格式和WAV文件的语音识别（ASR）。

## 📋 目录

- [系统要求](#系统要求)
- [环境配置](#环境配置)
- [工作流程](#工作流程)
  - [第一步：WebM转WAV批处理](#第一步webm转wav批处理)
  - [第二步：WAV转ASR批量识别](#第二步wav转asr批量识别)
- [文件命名规范](#文件命名规范)
- [高级功能](#高级功能)
- [故障排除](#故障排除)

---

## 🔧 系统要求

### 必需软件
- **Python**: 3.10+
- **FFmpeg**: 用于音频格式转换
- **CUDA** (可选): GPU加速需要

### Python依赖
```bash
pip install -r requirements.txt
```

### FFmpeg安装

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install ffmpeg
```

**验证安装:**
```bash
ffmpeg -version
```

---

## 🛠️ 环境配置

### 1. 创建Python虚拟环境
```bash
# 创建conda环境
conda create --name fireredasr python=3.10

# 激活环境
conda activate fireredasr

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置环境变量
```bash
# 在项目根目录执行
export PATH=$PWD/fireredasr/:$PWD/fireredasr/utils/:$PATH
export PYTHONPATH=$PWD/:$PYTHONPATH
```

**建议**: 将上述命令添加到 `~/.bashrc` 或 `~/.zshrc` 中，避免每次手动设置。

### 3. 下载预训练模型
从 [HuggingFace](https://huggingface.co/fireredteam) 下载模型文件，放置于 `pretrained_models/` 目录：

```
pretrained_models/
├── FireRedASR-AED-L/
│   ├── model.pth.tar
│   ├── train_bpe1000.model
│   ├── dict.txt
│   └── config.json
└── FireRedASR-LLM-L/  (可选)
```

---

## 📊 工作流程

完整的音频处理流程分为两个步骤：

```
WebM文件 --> [步骤1: 格式转换] --> WAV文件 --> [步骤2: 语音识别] --> 文本结果
```

---

## 🎵 第一步：WebM转WAV批处理

### 功能说明
将WebM音频文件批量转换为符合ASR要求的WAV格式：
- **采样率**: 16kHz
- **编码格式**: PCM_S16LE (16位)
- **声道**: 单声道 (Mono)

### 快速开始

#### 1. 修改配置参数

编辑 `fireredasr/transwav.py` 文件头部的路径配置：

```python
# 在 main() 函数中修改以下路径
INPUT_FOLDER = "/path/to/your/webm/files"        # WebM文件所在目录
OUTPUT_FOLDER = "/path/to/output/wav/files"      # WAV输出目录
PROGRESS_FILE = "/path/to/conversion_progress.txt"  # 进度记录文件
LOG_FILE = "/path/to/conversion.log"             # 日志文件
```

#### 2. 运行转换脚本

```bash
cd fireredasr
python transwav.py
```

### 转换特性

✅ **自动时间排序**: 自动解析文件名中的时间戳（HH.MM.SS）并按时间顺序处理  
✅ **进度追踪**: 实时记录转换进度到 `conversion_progress.txt`  
✅ **详细日志**: 完整的转换日志保存到 `conversion.log`  
✅ **错误处理**: 自动跳过损坏文件，继续处理其余文件  
✅ **双重输出**: 同时在控制台和日志文件中显示处理状态  

### 输出示例

**控制台输出:**
```
================================================================================
WebM 到 WAV 批量转换工具
================================================================================
输入文件夹: /home/user/webm_files
输出文件夹: /home/user/wav_files
FFmpeg 参数: -acodec pcm_s16le -ac 1 -ar 16000
================================================================================

找到 150 个 WebM 文件

[1/150] 正在处理: 09.40.23_小红_.webm
✓ 转换成功: 09.40.23_小红_.webm -> 09.40.23_小红_.wav

[2/150] 正在处理: 09.40.30_小明_.webm
✓ 转换成功: 09.40.30_小明_.webm -> 09.40.30_小明_.wav

...

================================================================================
转换完成！
================================================================================
总文件数: 150
成功: 148
失败: 2
================================================================================
```

---

## 🎤 第二步：WAV转ASR批量识别

### 功能说明
对WAV音频文件进行批量语音识别，输出时间戳格式的文本结果。

### 方式1：使用全局配置（推荐）

#### 1. 修改全局配置参数

编辑 `fireredasr/wavASR.py` 文件头部的配置：

```python
# ============================================================================
# 全局配置参数（可根据需要修改）
# ============================================================================

# 基本配置
ASR_TYPE = "aed"  # ASR模型类型: "aed" 或 "llm"
MODEL_DIR = "/home/jbj/openai/modle/FireRedASR-AED-L"  # 预训练模型目录
USE_GPU = 0  # 是否使用GPU: 1=使用GPU, 0=使用CPU

# 输入/输出配置
WAV_DIR = "/home/jbj/openai/modle/FireRedASR-AED-L/2025-10-24-wav"  # 音频文件目录
OUTPUT_FILE = "transcription_results.txt"  # 输出文件路径
CHECKPOINT_FILE = "transcription_checkpoint.json"  # 检查点文件路径

# 解码参数
BATCH_SIZE = 1  # 批处理大小
BEAM_SIZE = 1  # 束搜索宽度（1=贪婪搜索）
DECODE_MAX_LEN = 0  # 解码最大长度（0=无限制）

# AED模型专用参数
NBEST = 1  # 返回N个最佳结果
SOFTMAX_SMOOTHING = 1.0  # Softmax平滑系数
AED_LENGTH_PENALTY = 0.0  # AED长度惩罚系数
EOS_PENALTY = 1.0  # 结束符惩罚系数
```

#### 2. 直接运行

```bash
cd fireredasr
python wavASR.py
```

### 方式2：使用命令行参数

```bash
python fireredasr/wavASR.py \
    --wav_dir /path/to/wav/files \
    --asr_type aed \
    --model_dir /path/to/model \
    --use_gpu 0 \
    --output transcription_results.txt \
    --batch_size 1
```

### 方式3：使用Shell脚本

编辑 `examples/fireredasr/run_batch_transcribe.sh`：

```bash
#!/bin/bash

export PATH=$PWD/fireredasr/:$PWD/fireredasr/utils/:$PATH
export PYTHONPATH=$PWD/:$PYTHONPATH

python fireredasr/wavASR.py \
    --wav_dir /home/jbj/openai/modle/FireRedASR-AED-L/2025-10-24-wav \
    --asr_type aed \
    --model_dir /home/jbj/openai/modle/FireRedASR-AED-L \
    --use_gpu 0 \
    --output transcription_results.txt \
    --batch_size 1 \
    "$@"
```

运行脚本：
```bash
bash run_batch_transcribe.sh
```

### 断点续传功能

#### 自动恢复（从上次中断处继续）
```bash
python fireredasr/wavASR.py
```
脚本会自动检测 `transcription_checkpoint.json` 并从断点处继续。

#### 指定文件恢复
```bash
python fireredasr/wavASR.py --resume 09.40.23_小红_.wav
```

#### 重置并重新开始
```bash
python fireredasr/wavASR.py --reset
```

### 输出格式

**文本输出（transcription_results.txt）:**
```
09:40:23 小红:今天天气真好
09:40:30 小红:我们去公园散步吧
09:41:15 小明:好的没问题
09:42:05 小红:那我们现在就出发
09:42:20 小明:等我拿个外套
```

**检查点文件（transcription_checkpoint.json）:**
```json
{
  "last_processed": "09.42.20_小明_.wav",
  "completed_files": [
    "09.40.23_小红_.wav",
    "09.40.30_小红_.wav",
    "09.41.15_小明_.wav",
    "09.42.05_小红_.wav",
    "09.42.20_小明_.wav"
  ],
  "total_files": 150
}
```

---

## 📝 文件命名规范

### 标准格式
```
HH.MM.SS_说话人_.扩展名
```

### 示例
```
09.40.23_小红_.webm    ✅ 正确
09.40.30_小明_.webm    ✅ 正确
14.25.10_客服_.webm    ✅ 正确
23.59.59_系统_.wav     ✅ 正确

9.40.23_小红_.webm     ❌ 错误（小时应为两位数）
09-40-23_小红.webm     ❌ 错误（分隔符应为点号）
小红_09.40.23.webm     ❌ 错误（顺序错误）
```

### 命名规则说明
- **HH**: 00-23 (24小时制)
- **MM**: 00-59 (分钟)
- **SS**: 00-59 (秒)
- **说话人**: 任意UTF-8字符（中英文均可）
- **扩展名**: .webm 或 .wav

---

## 🚀 高级功能

### 1. 参数优化建议

#### 高精度场景（推荐会议转录）
```python
BEAM_SIZE = 3           # 使用束搜索
NBEST = 1              
SOFTMAX_SMOOTHING = 1.25
AED_LENGTH_PENALTY = 0.6
```

#### 高速度场景（实时处理）
```python
BEAM_SIZE = 1           # 贪婪搜索
BATCH_SIZE = 4          # 增加批处理（需要GPU）
USE_GPU = 1             # 启用GPU
```

### 2. GPU加速配置

检查GPU可用性：
```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

启用GPU：
```python
USE_GPU = 1  # 在配置文件中设置
```

### 3. 处理超长音频

**FireRedASR-AED:**
- 最佳长度: < 30秒
- 支持长度: < 60秒
- 最大长度: < 200秒（超过可能出错）

**解决方案**:
```bash
# 使用FFmpeg切分长音频
ffmpeg -i long_audio.wav -f segment -segment_time 30 -c copy output_%03d.wav
```

### 4. 批量处理优化

**大批量文件处理建议:**
1. 先测试小批量（10-20个文件）验证配置
2. 使用 `--batch_size` 增加并行处理（GPU场景）
3. 监控系统资源使用情况
4. 定期备份检查点文件

---

## 🔍 故障排除

### 常见问题

#### 1. FFmpeg未找到
```
错误: 未找到 ffmpeg 命令
```
**解决方案:**
```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg
```

#### 2. 模型文件缺失
```
错误: FileNotFoundError: model.pth.tar
```
**解决方案:**
- 确认模型已下载到 `pretrained_models/` 目录
- 检查 `MODEL_DIR` 配置路径是否正确

#### 3. CUDA内存不足
```
RuntimeError: CUDA out of memory
```
**解决方案:**
```python
# 减小批处理大小
BATCH_SIZE = 1

# 或切换到CPU
USE_GPU = 0
```

#### 4. 音频格式不匹配
```
错误: 音频格式不支持
```
**解决方案:**
```bash
# 使用FFmpeg转换为标准格式
ffmpeg -i input.wav -ar 16000 -ac 1 -acodec pcm_s16le output.wav
```

#### 5. 检查点文件损坏
```
⚠ 检查点文件损坏，将从头开始
```
**解决方案:**
```bash
# 删除损坏的检查点文件
rm transcription_checkpoint.json

# 重新运行
python fireredasr/wavASR.py --reset
```

### 日志分析

**查看转换日志:**
```bash
cat conversion.log | grep "✗"  # 查看失败的转换
```

**查看转录进度:**
```bash
cat transcription_checkpoint.json | jq '.completed_files | length'
```

---

## 📊 完整工作流程示例

### 场景：处理会议录音

**步骤1: 准备环境**
```bash
conda activate fireredasr
cd /home/user/FireRedASR
export PATH=$PWD/fireredasr/:$PWD/fireredasr/utils/:$PATH
export PYTHONPATH=$PWD/:$PYTHONPATH
```

**步骤2: WebM转WAV**
```bash
# 编辑 transwav.py 配置
# INPUT_FOLDER = "/home/user/meeting_2024/webm"
# OUTPUT_FOLDER = "/home/user/meeting_2024/wav"

python fireredasr/transwav.py
```

**步骤3: 批量转录**
```bash
# 编辑 wavASR.py 配置
# WAV_DIR = "/home/user/meeting_2024/wav"
# OUTPUT_FILE = "meeting_transcript.txt"
# BEAM_SIZE = 3  # 高精度
# SOFTMAX_SMOOTHING = 1.25
# AED_LENGTH_PENALTY = 0.6

python fireredasr/wavASR.py
```

**步骤4: 查看结果**
```bash
cat meeting_transcript.txt
```

---

## 📚 参考资料

- **项目主页**: [GitHub - FireRedASR](https://github.com/FireRedTeam/FireRedASR)
- **技术论文**: [ArXiv Paper](https://arxiv.org/pdf/2501.14350)
- **模型下载**: [HuggingFace Models](https://huggingface.co/fireredteam)
- **在线Demo**: [HuggingFace Space](https://huggingface.co/spaces/FireRedTeam/FireRedASR)

---

## 📄 许可证

本项目遵循 FireRedASR 原项目许可证。详见 [LICENSE](LICENSE) 文件。

---

## 🤝 贡献与反馈

如有问题或建议，请通过以下方式反馈：
- 提交 GitHub Issue
- 查看官方文档
- 联系开发团队

---

**最后更新**: 2025-10-29  
**文档版本**: 1.0

