#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FireRedASR 语音识别脚本
功能：将音频文件（WAV格式）转换为文本
支持两种模型类型：AED（Attention-based Encoder-Decoder）和 LLM（Large Language Model）
"""

import argparse
import glob
import os
import sys

from fireredasr.models.fireredasr import FireRedAsr


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
    required=True, 
    choices=["aed", "llm"],
    help='ASR模型类型：\n'
         '  - aed: Attention-based Encoder-Decoder 模型\n'
         '  - llm: Large Language Model 模型'
)

parser.add_argument(
    '--model_dir', 
    type=str, 
    required=True,
    help='预训练模型所在目录的路径'
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
    help='包含WAV文件的目录路径（递归搜索所有子目录）\n'
         '示例: --wav_dir /path/to/audio_folder'
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
    help='识别结果输出文件路径（可选）\n'
         '格式: uttid<TAB>识别文本\n'
         '如不指定，仅在控制台输出'
)

# ----------------------------------------------------------------------------
# 解码通用参数
# ----------------------------------------------------------------------------
parser.add_argument(
    '--use_gpu', 
    type=int, 
    default=1,
    help='是否使用GPU进行推理\n'
         '  - 1: 使用GPU（需要CUDA支持）\n'
         '  - 0: 使用CPU\n'
         '默认值: 1'
)

parser.add_argument(
    "--batch_size", 
    type=int, 
    default=1,
    help='批处理大小（一次处理多少个音频）\n'
         '较大的batch_size可以提高GPU利用率，但需要更多显存\n'
         '默认值: 1'
)

parser.add_argument(
    "--beam_size", 
    type=int, 
    default=1,
    help='束搜索（Beam Search）的束宽\n'
         '  - 1: 贪婪搜索（最快，精度较低）\n'
         '  - >1: 束搜索（更准确，但速度较慢）\n'
         '推荐值: 3-10\n'
         '默认值: 1'
)

parser.add_argument(
    "--decode_max_len", 
    type=int, 
    default=0,
    help='解码最大长度（输出文本的最大token数）\n'
         '  - 0: 无限制（自动根据模型配置）\n'
         '  - >0: 限制最大输出长度\n'
         '默认值: 0'
)

# ----------------------------------------------------------------------------
# FireRedASR-AED 模型专用参数
# ----------------------------------------------------------------------------
parser.add_argument(
    "--nbest", 
    type=int, 
    default=1,
    help='[AED模型] 返回N个最佳识别结果\n'
         '仅在beam_size>1时有效\n'
         '默认值: 1'
)

parser.add_argument(
    "--softmax_smoothing", 
    type=float, 
    default=1.0,
    help='[AED模型] Softmax平滑系数\n'
         '用于调整输出概率分布的尖锐度\n'
         '  - =1.0: 标准softmax\n'
         '  - >1.0: 分布更平滑\n'
         '  - <1.0: 分布更尖锐\n'
         '默认值: 1.0'
)

parser.add_argument(
    "--aed_length_penalty", 
    type=float, 
    default=0.0,
    help='[AED模型] 长度惩罚系数\n'
         '用于控制输出序列的长度\n'
         '  - =0.0: 无惩罚\n'
         '  - >0.0: 鼓励更长的输出\n'
         '  - <0.0: 鼓励更短的输出\n'
         '默认值: 0.0'
)

parser.add_argument(
    "--eos_penalty", 
    type=float, 
    default=1.0,
    help='[AED模型] 结束符（EOS）惩罚系数\n'
         '用于控制模型何时结束解码\n'
         '  - =1.0: 标准概率\n'
         '  - >1.0: 更容易结束（输出更短）\n'
         '  - <1.0: 不易结束（输出更长）\n'
         '默认值: 1.0'
)

# ----------------------------------------------------------------------------
# FireRedASR-LLM 模型专用参数
# ----------------------------------------------------------------------------
parser.add_argument(
    "--decode_min_len", 
    type=int, 
    default=0,
    help='[LLM模型] 解码最小长度\n'
         '强制模型至少生成指定数量的token\n'
         '默认值: 0'
)

parser.add_argument(
    "--repetition_penalty", 
    type=float, 
    default=1.0,
    help='[LLM模型] 重复惩罚系数\n'
         '用于减少输出中的重复内容\n'
         '  - =1.0: 无惩罚\n'
         '  - >1.0: 惩罚重复（推荐1.1-1.5）\n'
         '  - <1.0: 鼓励重复（不推荐）\n'
         '默认值: 1.0'
)

parser.add_argument(
    "--llm_length_penalty", 
    type=float, 
    default=0.0,
    help='[LLM模型] 长度惩罚系数\n'
         '与aed_length_penalty类似，用于LLM模型\n'
         '默认值: 0.0'
)

parser.add_argument(
    "--temperature", 
    type=float, 
    default=1.0,
    help='[LLM模型] 温度参数\n'
         '用于控制输出的随机性\n'
         '  - =1.0: 标准采样\n'
         '  - >1.0: 输出更随机、更多样化\n'
         '  - <1.0: 输出更确定、更保守（接近beam search）\n'
         '  - 接近0: 接近贪婪搜索\n'
         '默认值: 1.0'
)


# ============================================================================
# 主函数：语音识别处理流程
# ============================================================================
def main(args):
    """
    主函数：执行批量语音识别
    
    处理流程：
    1. 获取待处理的音频文件列表
    2. 加载预训练的ASR模型
    3. 按batch_size分批处理音频文件
    4. 将识别结果输出到控制台和/或文件
    
    Args:
        args: 命令行参数对象
    """
    # 步骤1: 获取所有待识别的音频文件信息
    # wavs 是一个列表，每个元素是 (uttid, wav_path) 元组
    wavs = get_wav_info(args)
    print(wavs)
    # 步骤2: 打开输出文件（如果指定了输出路径）
    fout = open(args.output, "w") if args.output else None

    # 步骤3: 从预训练目录加载ASR模型
    # 根据 asr_type 自动选择 AED 或 LLM 模型
    model = FireRedAsr.from_pretrained(args.asr_type, args.model_dir)

    # 步骤4: 批量处理音频文件
    batch_uttid = []      # 当前批次的音频ID列表
    batch_wav_path = []   # 当前批次的音频文件路径列表
    
    for i, wav in enumerate(wavs):
        uttid, wav_path = wav
        batch_uttid.append(uttid)
        batch_wav_path.append(wav_path)
        
        # 判断是否达到批次大小或已是最后一个文件
        # 如果批次未满且不是最后一个文件，继续累积
        if len(batch_wav_path) < args.batch_size and i != len(wavs) - 1:
            continue

        # 批次已满或到达最后，执行识别
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

        # 步骤5: 输出识别结果
        for result in results:
            # 打印到控制台（完整的结果字典）
            print(result)
            
            # 写入输出文件（仅保存 uttid 和识别文本）
            if fout is not None:
                fout.write(f"{result['uttid']}\t{result['text']}\n")

        # 清空当前批次，准备处理下一批
        batch_uttid = []
        batch_wav_path = []


# ============================================================================
# 辅助函数：获取音频文件信息
# ============================================================================
def get_wav_info(args):
    """
    根据命令行参数获取待处理的音频文件列表
    
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
        # 对文件路径进行排序，确保处理顺序一致
        wavs = [(base(p), p) for p in sorted(args.wav_paths)]
        
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
        # 排序并构建 (uttid, path) 元组列表
        wavs = [(base(p), p) for p in sorted(wavs)]
        
    # 错误处理: 没有提供任何有效输入
    else:
        raise ValueError("请提供有效的音频输入参数：wav_path, wav_paths, wav_scp 或 wav_dir")
    
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
