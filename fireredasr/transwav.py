#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量转换 WebM 文件到 WAV 格式
使用 FFmpeg 转换音频文件，并按时间戳顺序处理

python transwav.py   

"""

import os
import sys
import subprocess
import logging
from pathlib import Path
from datetime import datetime
import re


def setup_logging(log_file="conversion.log"):
    """配置双重日志记录（控制台 + 文件）"""
    # 创建日志记录器
    logger = logging.getLogger('WebM2WAV')
    logger.setLevel(logging.INFO)
    
    # 文件处理器
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    
    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    
    # 添加处理器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def parse_time_from_filename(filename):
    """
    从文件名中解析时间戳
    格式: HH.MM.SS_说话人_.webm
    返回: datetime.time 对象，用于排序
    """
    try:
        # 提取时间部分 (HH.MM.SS)
        time_match = re.match(r'^(\d{2})\.(\d{2})\.(\d{2})_', filename)
        if time_match:
            hour = int(time_match.group(1))
            minute = int(time_match.group(2))
            second = int(time_match.group(3))
            
            # 验证时间有效性
            if 0 <= hour <= 23 and 0 <= minute <= 59 and 0 <= second <= 59:
                return hour * 3600 + minute * 60 + second  # 返回总秒数用于排序
            else:
                return None
        else:
            return None
    except Exception as e:
        return None


def get_sorted_webm_files(input_folder):
    """
    获取输入文件夹中的所有 webm 文件，并按时间戳排序
    """
    input_path = Path(input_folder)
    
    if not input_path.exists():
        raise FileNotFoundError(f"输入文件夹不存在: {input_folder}")
    
    # 获取所有 .webm 文件
    webm_files = list(input_path.glob("*.webm"))
    
    # 按时间戳排序
    files_with_time = []
    files_without_time = []
    
    for file in webm_files:
        time_seconds = parse_time_from_filename(file.name)
        if time_seconds is not None:
            files_with_time.append((time_seconds, file))
        else:
            files_without_time.append(file)
    
    # 排序有时间戳的文件
    files_with_time.sort(key=lambda x: x[0])
    sorted_files = [f[1] for f in files_with_time]
    
    # 将无法解析时间戳的文件放在最后
    sorted_files.extend(files_without_time)
    
    return sorted_files


def convert_webm_to_wav(input_file, output_file, logger):
    """
    使用 FFmpeg 转换单个 WebM 文件到 WAV
    参数: -acodec pcm_s16le -ac 1 -ar 16000
    """
    # FFmpeg 命令
    cmd = [
        'ffmpeg',
        '-i', str(input_file),
        '-acodec', 'pcm_s16le',  # 16位 PCM 编码
        '-ac', '1',               # 单声道
        '-ar', '16000',           # 16kHz 采样率
        '-y',                     # 覆盖已存在的文件
        str(output_file)
    ]
    
    try:
        # 执行转换
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=300  # 5分钟超时
        )
        
        if result.returncode == 0:
            logger.info(f"✓ 转换成功: {input_file.name} -> {output_file.name}")
            return True, None
        else:
            error_msg = result.stderr[-500:] if result.stderr else "未知错误"
            logger.error(f"✗ 转换失败: {input_file.name}")
            logger.error(f"  错误信息: {error_msg}")
            return False, error_msg
            
    except subprocess.TimeoutExpired:
        error_msg = "转换超时（超过5分钟）"
        logger.error(f"✗ 转换失败: {input_file.name} - {error_msg}")
        return False, error_msg
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"✗ 转换失败: {input_file.name} - {error_msg}")
        return False, error_msg


def write_progress(progress_file, status_msg):
    """写入进度文件"""
    with open(progress_file, 'a', encoding='utf-8') as f:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        f.write(f"[{timestamp}] {status_msg}\n")


def main():
    # 配置路径
    INPUT_FOLDER = "/home/jbj/openai/modle/FireRedASR-AED-L/2025-10-24"
    OUTPUT_FOLDER = "/home/jbj/openai/modle/FireRedASR-AED-L/2025-10-24-wav"
    PROGRESS_FILE = "/home/jbj/openai/modle/FireRedASR-AED-L/conversion_progress.txt"
    LOG_FILE = "/home/jbj/openai/modle/FireRedASR-AED-L/conversion.log"
    
    # 设置日志
    logger = setup_logging(LOG_FILE)
    
    logger.info("=" * 80)
    logger.info("WebM 到 WAV 批量转换工具")
    logger.info("=" * 80)
    logger.info(f"输入文件夹: {INPUT_FOLDER}")
    logger.info(f"输出文件夹: {OUTPUT_FOLDER}")
    logger.info(f"FFmpeg 参数: -acodec pcm_s16le -ac 1 -ar 16000")
    logger.info("=" * 80)
    
    # 检查 FFmpeg 是否可用
    try:
        subprocess.run(['ffmpeg', '-version'], 
                      stdout=subprocess.PIPE, 
                      stderr=subprocess.PIPE, 
                      check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("错误: 未找到 ffmpeg 命令，请确保已安装 FFmpeg")
        sys.exit(1)
    
    # 创建输出文件夹
    output_path = Path(OUTPUT_FOLDER)
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"输出文件夹已创建: {OUTPUT_FOLDER}\n")
    
    # 初始化进度文件
    with open(PROGRESS_FILE, 'w', encoding='utf-8') as f:
        f.write(f"WebM 到 WAV 转换进度记录\n")
        f.write(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'=' * 80}\n\n")
    
    # 获取排序后的文件列表
    try:
        webm_files = get_sorted_webm_files(INPUT_FOLDER)
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)
    
    if not webm_files:
        logger.warning("未找到任何 .webm 文件")
        sys.exit(0)
    
    logger.info(f"找到 {len(webm_files)} 个 WebM 文件\n")
    
    # 统计信息
    total_files = len(webm_files)
    success_count = 0
    failed_count = 0
    failed_files = []
    
    # 批量转换
    for index, input_file in enumerate(webm_files, 1):
        # 构建输出文件名（保持原文件名，只改扩展名）
        output_filename = input_file.stem + ".wav"
        output_file = output_path / output_filename
        
        logger.info(f"[{index}/{total_files}] 正在处理: {input_file.name}")
        
        # 记录进度
        write_progress(PROGRESS_FILE, f"[{index}/{total_files}] 开始转换: {input_file.name}")
        
        # 执行转换
        success, error_msg = convert_webm_to_wav(input_file, output_file, logger)
        
        if success:
            success_count += 1
            write_progress(PROGRESS_FILE, f"✓ 成功: {input_file.name} -> {output_filename}")
        else:
            failed_count += 1
            failed_files.append((input_file.name, error_msg))
            write_progress(PROGRESS_FILE, f"✗ 失败: {input_file.name} - {error_msg}")
        
        logger.info("")  # 空行分隔
    
    # 最终统计
    logger.info("=" * 80)
    logger.info("转换完成！")
    logger.info("=" * 80)
    logger.info(f"总文件数: {total_files}")
    logger.info(f"成功: {success_count}")
    logger.info(f"失败: {failed_count}")
    logger.info("=" * 80)
    
    # 记录失败的文件
    if failed_files:
        logger.warning("\n失败的文件列表:")
        for filename, error in failed_files:
            logger.warning(f"  - {filename}: {error}")
    
    # 写入最终进度
    with open(PROGRESS_FILE, 'a', encoding='utf-8') as f:
        f.write(f"\n{'=' * 80}\n")
        f.write(f"转换完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总文件数: {total_files}\n")
        f.write(f"成功: {success_count}\n")
        f.write(f"失败: {failed_count}\n")
        f.write(f"{'=' * 80}\n")
    
    logger.info(f"\n详细日志已保存到: {LOG_FILE}")
    logger.info(f"进度记录已保存到: {PROGRESS_FILE}")
    
    # 返回退出代码
    sys.exit(0 if failed_count == 0 else 1)


if __name__ == "__main__":
    main()

