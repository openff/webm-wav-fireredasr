#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FireRedASR 语音识别脚本
功能：将音频文件（WAV格式）转换为文本
支持模型类型：AED（Attention-based Encoder-Decoder）
2.批量音频转录
使用python虚拟环境命令是：conda activate fireredasr
输入文件夹：/home/jbj/openai/modle/FireRedASR-AED-L/2025-10-24-wav
文件命名：时间_说话人_.wav，时间由HH.MM.SS构成. 比如09.40.23_小红_.wav 表明上午9点40分23秒小红产生的音频。
格式要求：s16le 16000 单声道
工作内容：
1. 时间顺序处理
自动解析文件名中的时间戳（HH.MM.SS），按00:00:00到23:59:59严格顺序转录
2. 断点续传机制
    检查点系统：使用JSON文件(transcription_checkpoint.json)记录进度​
    记录最后处理的文件位置
    记录所有已完成的文件列表
    支持随时中断和恢复三种恢复方式：
    # 自动从上次中断处恢复
    python batch_transcribe.py
    # 指定从某个文件开始
    python batch_transcribe.py --resume 09.40.23_小红_.wav
    # 重置并从头开始
    python batch_transcribe.py --reset
3. 输出格式
严格按要求的格式输出：HH:MM:SS 说话人:转录数据 
比如:
09:40:23 小红:今天天气真好
09:40:30 小红:我们去公园散步吧
09:41:15 小明:好的没问题

使用方法：
方式1（推荐）：使用全局配置参数，直接运行
python fireredasr/wavASR.py

方式2：通过命令行参数覆盖全局配置
python /home/jbj/openai/modle/FireRedASR-AED-L/FireRedASR/fireredasr/wavASR.py \
    --wav_dir /home/jbj/openai/modle/FireRedASR-AED-L/2025-10-24-wav \
    --asr_type aed \
    --model_dir /home/jbj/openai/modle/FireRedASR-AED-L \
    --use_gpu 0 \
    --output transcription_results.txt \
    --batch_size 1 \
    "$@"

方式3（完整参数）：
python fireredasr/wavASR.py --wav_dir /home/jbj/openai/modle/FireRedASR-AED-L/2025-10-24-wav --asr_type aed --model_dir /home/jbj/openai/modle/FireRedASR-AED-L --use_gpu 0 --output transcription_results.txt --batch_size 1 
"""

import argparse
import glob
import os
import sys
import json
import re
from datetime import datetime

from fireredasr.models.fireredasr import FireRedAsr


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

# LLM模型专用参数
DECODE_MIN_LEN = 0  # 解码最小长度
REPETITION_PENALTY = 1.0  # 重复惩罚系数
LLM_LENGTH_PENALTY = 0.0  # LLM长度惩罚系数
TEMPERATURE = 1.0  # 温度参数


# ============================================================================
# 命令行参数解析器配置
# ============================================================================
parser = argparse.ArgumentParser(description="FireRedASR 语音识别工具")

# ----------------------------------------------------------------------------
# 必需参数：模型配置
# ----------------------------------------------------------------------------
parser.add_argument(
    '--asr_type', 
    type=str, 
    default=ASR_TYPE,
    choices=["aed", "llm"],
    help='ASR模型类型：\n'
         '  - aed: Attention-based Encoder-Decoder 模型\n'
         '  - llm: Large Language Model 模型\n'
         f'默认值: {ASR_TYPE}'
)

parser.add_argument(
    '--model_dir', 
    type=str, 
    default=MODEL_DIR,
    help=f'预训练模型所在目录的路径\n'
         f'默认值: {MODEL_DIR}'
)

# ----------------------------------------------------------------------------
# 输入/输出参数（四选一）
# ----------------------------------------------------------------------------
parser.add_argument(
    "--wav_path", 
    type=str,
    help='单个WAV音频文件的路径\n'
         '示例: --wav_path /path/to/audio.wav'
)

parser.add_argument(
    "--wav_paths", 
    type=str, 
    nargs="*",
    help='多个WAV音频文件的路径列表（空格分隔）\n'
         '示例: --wav_paths audio1.wav audio2.wav audio3.wav'
)

parser.add_argument(
    "--wav_dir", 
    type=str,
    default=WAV_DIR,
    help=f'包含WAV文件的目录路径（递归搜索所有子目录）\n'
         f'示例: --wav_dir /path/to/audio_folder\n'
         f'默认值: {WAV_DIR}'
)

parser.add_argument(
    "--wav_scp", 
    type=str,
    help='WAV文件列表文件（scp格式）\n'
         '每行格式: uttid wav_path\n'
         '示例文件内容:\n'
         '  utt001 /path/to/audio1.wav\n'
         '  utt002 /path/to/audio2.wav'
)

parser.add_argument(
    "--output", 
    type=str,
    default=OUTPUT_FILE,
    help=f'识别结果输出文件路径（可选）\n'
         f'格式: uttid<TAB>识别文本\n'
         f'如不指定，仅在控制台输出\n'
         f'默认值: {OUTPUT_FILE}'
)

# ----------------------------------------------------------------------------
# 解码通用参数
# ----------------------------------------------------------------------------
parser.add_argument(
    '--use_gpu', 
    type=int, 
    default=USE_GPU,
    help=f'是否使用GPU进行推理\n'
         f'  - 1: 使用GPU（需要CUDA支持）\n'
         f'  - 0: 使用CPU\n'
         f'默认值: {USE_GPU}'
)

parser.add_argument(
    "--batch_size", 
    type=int, 
    default=BATCH_SIZE,
    help=f'批处理大小（一次处理多少个音频）\n'
         f'较大的batch_size可以提高GPU利用率，但需要更多显存\n'
         f'默认值: {BATCH_SIZE}'
)

parser.add_argument(
    "--beam_size", 
    type=int, 
    default=BEAM_SIZE,
    help=f'束搜索（Beam Search）的束宽\n'
         f'  - 1: 贪婪搜索（最快，精度较低）\n'
         f'  - >1: 束搜索（更准确，但速度较慢）\n'
         f'推荐值: 3-10\n'
         f'默认值: {BEAM_SIZE}'
)

parser.add_argument(
    "--decode_max_len", 
    type=int, 
    default=DECODE_MAX_LEN,
    help=f'解码最大长度（输出文本的最大token数）\n'
         f'  - 0: 无限制（自动根据模型配置）\n'
         f'  - >0: 限制最大输出长度\n'
         f'默认值: {DECODE_MAX_LEN}'
)

# ----------------------------------------------------------------------------
# FireRedASR-AED 模型专用参数
# ----------------------------------------------------------------------------
parser.add_argument(
    "--nbest", 
    type=int, 
    default=NBEST,
    help=f'[AED模型] 返回N个最佳识别结果\n'
         f'仅在beam_size>1时有效\n'
         f'默认值: {NBEST}'
)

parser.add_argument(
    "--softmax_smoothing", 
    type=float, 
    default=SOFTMAX_SMOOTHING,
    help=f'[AED模型] Softmax平滑系数\n'
         f'用于调整输出概率分布的尖锐度\n'
         f'  - =1.0: 标准softmax\n'
         f'  - >1.0: 分布更平滑\n'
         f'  - <1.0: 分布更尖锐\n'
         f'默认值: {SOFTMAX_SMOOTHING}'
)

parser.add_argument(
    "--aed_length_penalty", 
    type=float, 
    default=AED_LENGTH_PENALTY,
    help=f'[AED模型] 长度惩罚系数\n'
         f'用于控制输出序列的长度\n'
         f'  - =0.0: 无惩罚\n'
         f'  - >0.0: 鼓励更长的输出\n'
         f'  - <0.0: 鼓励更短的输出\n'
         f'默认值: {AED_LENGTH_PENALTY}'
)

parser.add_argument(
    "--eos_penalty", 
    type=float, 
    default=EOS_PENALTY,
    help=f'[AED模型] 结束符（EOS）惩罚系数\n'
         f'用于控制模型何时结束解码\n'
         f'  - =1.0: 标准概率\n'
         f'  - >1.0: 更容易结束（输出更短）\n'
         f'  - <1.0: 不易结束（输出更长）\n'
         f'默认值: {EOS_PENALTY}'
)

# ----------------------------------------------------------------------------
# FireRedASR-LLM 模型专用参数
# ----------------------------------------------------------------------------
parser.add_argument(
    "--decode_min_len", 
    type=int, 
    default=DECODE_MIN_LEN,
    help=f'[LLM模型] 解码最小长度\n'
         f'强制模型至少生成指定数量的token\n'
         f'默认值: {DECODE_MIN_LEN}'
)

parser.add_argument(
    "--repetition_penalty", 
    type=float, 
    default=REPETITION_PENALTY,
    help=f'[LLM模型] 重复惩罚系数\n'
         f'用于减少输出中的重复内容\n'
         f'  - =1.0: 无惩罚\n'
         f'  - >1.0: 惩罚重复（推荐1.1-1.5）\n'
         f'  - <1.0: 鼓励重复（不推荐）\n'
         f'默认值: {REPETITION_PENALTY}'
)

parser.add_argument(
    "--llm_length_penalty", 
    type=float, 
    default=LLM_LENGTH_PENALTY,
    help=f'[LLM模型] 长度惩罚系数\n'
         f'与aed_length_penalty类似，用于LLM模型\n'
         f'默认值: {LLM_LENGTH_PENALTY}'
)

parser.add_argument(
    "--temperature", 
    type=float, 
    default=TEMPERATURE,
    help=f'[LLM模型] 温度参数\n'
         f'用于控制输出的随机性\n'
         f'  - =1.0: 标准采样\n'
         f'  - >1.0: 输出更随机、更多样化\n'
         f'  - <1.0: 输出更确定、更保守（接近beam search）\n'
         f'  - 接近0: 接近贪婪搜索\n'
         f'默认值: {TEMPERATURE}'
)

# ----------------------------------------------------------------------------
# 断点续传相关参数
# ----------------------------------------------------------------------------
parser.add_argument(
    "--resume", 
    type=str,
    help='从指定文件名继续处理（例如：09.40.23_小红_.wav）\n'
         '如不指定，自动从上次中断处恢复'
)

parser.add_argument(
    "--reset", 
    action='store_true',
    help='重置检查点，从头开始处理所有文件'
)

parser.add_argument(
    "--checkpoint_file", 
    type=str,
    default=CHECKPOINT_FILE,
    help=f'检查点文件路径\n'
         f'默认值: {CHECKPOINT_FILE}'
)


# ============================================================================
# 辅助函数：时间戳解析
# ============================================================================
def parse_timestamp_from_filename(filename):
    """
    从文件名中解析时间戳
    
    文件名格式: HH.MM.SS_说话人_.wav
    例如: 09.40.23_小红_.wav -> (9, 40, 23, "小红")
    
    Args:
        filename: 文件名（不含路径）
        
    Returns:
        tuple: (hour, minute, second, speaker) 或 None（如果解析失败）
    """
    # 匹配格式: HH.MM.SS_说话人_
    match = re.match(r'(\d{2})\.(\d{2})\.(\d{2})_(.+?)_', filename)
    if match:
        hour = int(match.group(1))
        minute = int(match.group(2))
        second = int(match.group(3))
        speaker = match.group(4)
        return (hour, minute, second, speaker)
    return None


def time_to_seconds(hour, minute, second):
    """
    将时间转换为秒数（用于排序）
    
    Args:
        hour: 小时
        minute: 分钟
        second: 秒
        
    Returns:
        int: 总秒数
    """
    return hour * 3600 + minute * 60 + second


def format_time(hour, minute, second):
    """
    格式化时间为 HH:MM:SS 格式
    
    Args:
        hour: 小时
        minute: 分钟
        second: 秒
        
    Returns:
        str: 格式化后的时间字符串
    """
    return f"{hour:02d}:{minute:02d}:{second:02d}"


# ============================================================================
# 辅助函数：检查点管理
# ============================================================================
def load_checkpoint(checkpoint_file):
    """
    加载检查点文件
    
    Args:
        checkpoint_file: 检查点文件路径
        
    Returns:
        dict: 检查点数据，包含以下字段：
            - last_processed: 最后处理的文件名
            - completed_files: 已完成的文件列表
            - total_files: 总文件数
    """
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint = json.load(f)
                print(f"✓ 加载检查点成功: {len(checkpoint.get('completed_files', []))} 个文件已完成")
                return checkpoint
        except Exception as e:
            print(f"⚠ 检查点文件损坏，将从头开始: {e}")
    
    return {
        "last_processed": None,
        "completed_files": [],
        "total_files": 0
    }


def save_checkpoint(checkpoint_file, checkpoint):
    """
    保存检查点文件
    
    Args:
        checkpoint_file: 检查点文件路径
        checkpoint: 检查点数据字典
    """
    try:
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(checkpoint, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"⚠ 保存检查点失败: {e}")


# ============================================================================
# 主函数：语音识别处理流程
# ============================================================================
def main(args):
    """
    主函数：执行批量语音识别
    
    处理流程：
    1. 加载检查点（如果存在）
    2. 获取待处理的音频文件列表（按时间排序）
    3. 根据检查点过滤已处理的文件
    4. 加载预训练的ASR模型
    5. 按batch_size分批处理音频文件
    6. 将识别结果输出到控制台和/或文件（格式化为 HH:MM:SS 说话人:文本）
    7. 每处理完一个文件，更新检查点
    
    Args:
        args: 命令行参数对象
    """
    # 步骤1: 处理检查点
    checkpoint_file = args.checkpoint_file
    
    # 如果指定了 --reset，删除检查点文件
    if args.reset:
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)
            print("✓ 检查点已重置")
        checkpoint = {
            "last_processed": None,
            "completed_files": [],
            "total_files": 0
        }
    else:
        # 加载现有检查点
        checkpoint = load_checkpoint(checkpoint_file)
    
    # 步骤2: 获取所有待识别的音频文件信息（按时间排序）
    all_wavs = get_wav_info(args)
    print(f"✓ 找到 {len(all_wavs)} 个音频文件")
    
    # 步骤3: 根据检查点或 --resume 参数过滤文件
    completed_set = set(checkpoint.get('completed_files', []))
    
    # 确定从哪个文件开始处理
    start_filename = args.resume if args.resume else checkpoint.get('last_processed')
    
    if start_filename:
        # 找到起始文件的位置
        start_idx = None
        for i, (uttid, wav_path) in enumerate(all_wavs):
            filename = os.path.basename(wav_path)
            if filename == start_filename or uttid == start_filename.replace('.wav', ''):
                start_idx = i + 1  # 从下一个文件开始
                break
        
        if start_idx is not None:
            wavs = all_wavs[start_idx:]
            print(f"✓ 从文件 '{start_filename}' 之后继续处理，剩余 {len(wavs)} 个文件")
        else:
            print(f"⚠ 未找到指定的起始文件 '{start_filename}'，从头开始")
            wavs = all_wavs
    else:
        # 过滤掉已完成的文件
        wavs = [(uttid, path) for uttid, path in all_wavs 
                if os.path.basename(path) not in completed_set]
        
        if len(wavs) < len(all_wavs):
            print(f"✓ 跳过 {len(all_wavs) - len(wavs)} 个已完成的文件，剩余 {len(wavs)} 个文件")
    
    if len(wavs) == 0:
        print("✓ 所有文件已处理完成！")
        return
    
    # 更新检查点的总文件数
    checkpoint['total_files'] = len(all_wavs)
    
    # 步骤4: 打开输出文件（如果指定了输出路径）
    fout = open(args.output, "a", encoding='utf-8') if args.output else None

    # 步骤5: 从预训练目录加载ASR模型
    # 根据 asr_type 自动选择 AED 或 LLM 模型
    print(f"正在加载 {args.asr_type.upper()} 模型...")
    model = FireRedAsr.from_pretrained(args.asr_type, args.model_dir)
    print("✓ 模型加载完成")

    # 步骤6: 批量处理音频文件
    batch_uttid = []      # 当前批次的音频ID列表
    batch_wav_path = []   # 当前批次的音频文件路径列表
    batch_info = []       # 当前批次的时间和说话人信息
    
    processed_count = len(completed_set)
    
    for i, wav in enumerate(wavs):
        uttid, wav_path = wav
        filename = os.path.basename(wav_path)
        
        # 解析文件名中的时间戳和说话人
        time_info = parse_timestamp_from_filename(filename)
        
        batch_uttid.append(uttid)
        batch_wav_path.append(wav_path)
        batch_info.append((filename, time_info))
        
        # 判断是否达到批次大小或已是最后一个文件
        # 如果批次未满且不是最后一个文件，继续累积
        if len(batch_wav_path) < args.batch_size and i != len(wavs) - 1:
            continue

        # 批次已满或到达最后，执行识别
        print(f"\n处理批次 ({i+1-len(batch_wav_path)+1}-{i+1}/{len(wavs)})...")
        
        # 构建模型推理参数字典
        decode_config = {
            # 通用参数
            "use_gpu": args.use_gpu,
            "beam_size": args.beam_size,
            "decode_max_len": args.decode_max_len,
            
            # AED模型专用参数
            "nbest": args.nbest,
            "softmax_smoothing": args.softmax_smoothing,
            "aed_length_penalty": args.aed_length_penalty,
            "eos_penalty": args.eos_penalty,
            
            # LLM模型专用参数
            "decode_min_len": args.decode_min_len,
            "repetition_penalty": args.repetition_penalty,
            "llm_length_penalty": args.llm_length_penalty,
            "temperature": args.temperature
        }
        
        # 调用模型进行语音识别
        # 返回格式: [{"uttid": "xxx", "text": "识别文本", ...}, ...]
        results = model.transcribe(
            batch_uttid,
            batch_wav_path,
            decode_config
        )

        # 步骤7: 输出识别结果并更新检查点
        for j, result in enumerate(results):
            filename, time_info = batch_info[j]
            text = result.get('text', '')
            
            # 格式化输出
            if time_info:
                hour, minute, second, speaker = time_info
                formatted_time = format_time(hour, minute, second)
                formatted_output = f"{formatted_time} {speaker}:{text}"
            else:
                # 如果文件名格式不匹配，使用原始格式
                formatted_output = f"{result['uttid']}\t{text}"
            
            # 打印到控制台
            print(formatted_output)
            
            # 写入输出文件
            if fout is not None:
                fout.write(formatted_output + "\n")
                fout.flush()
            
            # 更新检查点
            checkpoint['last_processed'] = filename
            if filename not in checkpoint['completed_files']:
                checkpoint['completed_files'].append(filename)
            
            processed_count += 1
            
            # 每处理一个文件就保存一次检查点
            save_checkpoint(checkpoint_file, checkpoint)
        
        print(f"进度: {processed_count}/{checkpoint['total_files']} ({processed_count*100//checkpoint['total_files']}%)")
        
        # 清空当前批次，准备处理下一批
        batch_uttid = []
        batch_wav_path = []
        batch_info = []
    
    # 关闭输出文件
    if fout is not None:
        fout.close()
    
    print(f"\n✓ 全部处理完成！共处理 {len(wavs)} 个文件")


# ============================================================================
# 辅助函数：获取音频文件信息
# ============================================================================
def get_wav_info(args):
    """
    根据命令行参数获取待处理的音频文件列表
    按时间戳排序（从文件名中解析HH.MM.SS）
    
    支持四种输入方式（优先级从高到低）：
    1. --wav_path: 单个音频文件
    2. --wav_paths: 多个音频文件列表
    3. --wav_scp: scp格式的文件列表
    4. --wav_dir: 包含音频文件的目录
    
    Args:
        args: 命令行参数对象
        
    Returns:
        wavs: 音频信息列表，每个元素为 (uttid, wav_path) 元组
              - uttid: 音频的唯一标识符（从文件名提取，不含.wav扩展名）
              - wav_path: 音频文件的完整路径
              按时间戳从早到晚排序
              
    Raises:
        ValueError: 如果没有提供任何有效的音频输入参数
        
    示例返回值:
        [
            ("09.40.23_小红_", "/path/to/09.40.23_小红_.wav"),
            ("09.40.25_小明_", "/path/to/09.40.25_小明_.wav"),
            ...
        ]
    """
    # Lambda函数：从完整路径中提取文件名（不含扩展名）作为uttid
    # 例如: "/path/to/audio.wav" -> "audio"
    base = lambda p: os.path.basename(p).replace(".wav", "")
    
    # 方式1: 单个音频文件
    if args.wav_path:
        wavs = [(base(args.wav_path), args.wav_path)]
        
    # 方式2: 多个音频文件列表
    elif args.wav_paths and len(args.wav_paths) >= 1:
        wavs = [(base(p), p) for p in args.wav_paths]
        
    # 方式3: SCP格式文件列表
    # SCP文件格式（Kaldi风格）：
    # uttid1 /path/to/audio1.wav
    # uttid2 /path/to/audio2.wav
    elif args.wav_scp:
        wavs = [line.strip().split() for line in open(args.wav_scp)]
        
    # 方式4: 从目录递归搜索所有WAV文件
    elif args.wav_dir:
        # 递归搜索指定目录及其所有子目录中的.wav文件
        wavs = glob.glob(f"{args.wav_dir}/**/*.wav", recursive=True)
        wavs = [(base(p), p) for p in wavs]
        
    # 错误处理: 没有提供任何有效输入
    else:
        raise ValueError("请提供有效的音频输入参数：wav_path, wav_paths, wav_scp 或 wav_dir")
    
    # 按时间戳排序
    def get_sort_key(wav_tuple):
        """
        提取排序键：按时间戳排序
        返回 (秒数, 文件名)，秒数相同时按文件名排序
        """
        uttid, wav_path = wav_tuple
        filename = os.path.basename(wav_path)
        time_info = parse_timestamp_from_filename(filename)
        
        if time_info:
            hour, minute, second, speaker = time_info
            return (time_to_seconds(hour, minute, second), filename)
        else:
            # 无法解析时间戳的文件放在最后，按文件名排序
            return (999999, filename)
    
    wavs.sort(key=get_sort_key)
    
    # 打印找到的音频文件数量
    print(f"#wavs={len(wavs)}")
    
    return wavs


# ============================================================================
# 程序入口
# ============================================================================
if __name__ == "__main__":
    # 解析命令行参数
    args = parser.parse_args()
    
    # 打印所有参数配置（用于调试和日志记录）
    print(args)
    
    # 执行主函数
    main(args)
