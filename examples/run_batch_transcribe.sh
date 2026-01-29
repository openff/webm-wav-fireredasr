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
# cd /data/8T/modle/FireRedASR-AED-L/FireRedASR

#官方封装工具包fireredasr文件夹路劲，否则会报组件缺失。
export PATH=$PWD/fireredasr/:$PWD/fireredasr/utils/:$PATH
export PYTHONPATH=$PWD/:$PYTHONPATH

# #后续就可以执行wavASR转录程序
python fireredasr/wavASR.py \
    --wav_dir /data/8T/modle/audio-text/2025-10-24-wav \
    --asr_type aed \
    --model_dir /data/8T/modle/FireRedASR-AED-L \
    --use_gpu 1 \
    --output transcription_results.txt \
    --batch_size 1 
    
python fireredasr/speech2text.py \
    --wav_dir /data/8T/modle/audio-text/2025-10-24-wav \
    --asr_type llm \
    --model_dir /data/8T/modle/FireRedASR-LLM-L \
    --use_gpu 1 \
    --output transcription_results.txt \
    --batch_size 1 \
    --beam_size 3 


# python fireredasr/speech2text.py  \
# --wav_path /data/8T/modle/audio-text/out6.wav  \
# --asr_type llm  \
# --model_dir /data/8T/modle/FireRedASR-LLM-L  \
# --use_gpu 1  \
# --batch_size 4  \
# --beam_size 6  \
# --decode_max_len 0  \
# --temperature 1.0  \


#bk
# python fireredasr/speech2text.py  \
# --wav_path /data/8T/modle/audio-text/grds-60s.wav  \
# --asr_type llm  \
# --model_dir /data/8T/modle/FireRedASR-LLM-L  \
# --use_gpu 1  \
# --batch_size 4  \
# --beam_size 6  \
# --decode_max_len 0  \
# --temperature 1.0  \


echo "转录完成！结果已保存到 transcription_results.txt"

