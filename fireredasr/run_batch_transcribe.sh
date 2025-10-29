#!/bin/bash
# 批量音频转录运行脚本

# 运行批量转录
# 参数说明：
#   --wav_dir: 音频文件目录
#   --asr_type: 模型类型 (aed 或 llm)
#   --model_dir: 模型目录
#   --use_gpu: 是否使用GPU (1=是, 0=否)
#   --output: 输出文件路径
#   --batch_size: 批处理大小
#   --reset: 重置检查点（可选）
#   --resume: 从指定文件继续（可选）
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

echo "转录完成！结果已保存到 transcription_results.txt"

