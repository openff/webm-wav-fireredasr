#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速测试numpy数组输入功能
使用真实音频文件进行测试
"""

import numpy as np
import sys
import os
import time

# 添加模块路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fireredasr.models.fireredasr import FireRedAsr

def test_numpy_input():
    """测试numpy数组输入，使用真实音频文件"""
    
    print("=" * 70)
    print("测试 FireRedASR numpy数组输入功能")
    print("=" * 70)
    
    # 检查模型路径
    model_dir = "/data/8T/modle/FireRedASR-LLM-L"
    if not os.path.exists(model_dir):
        print(f"❌ 错误: 模型路径不存在: {model_dir}")
        print("请修改 model_dir 变量为正确的模型路径")
        return
    
    # 检查测试音频文件
    test_audio_path = "/data/8T/modle/audio-text/out6.wav"
    if not os.path.exists(test_audio_path):
        print(f"❌ 错误: 测试音频文件不存在: {test_audio_path}")
        return
    
    try:
        # 1. 加载模型
        print(f"\n【步骤1】 加载模型")
        print(f"   模型路径: {model_dir}")
        print(f"   模型类型: LLM")
        model = FireRedAsr.from_pretrained("llm", model_dir)
        print("✅ 模型加载成功")
        
        # 2. 读取真实音频文件
        print(f"\n【步骤2】 读取测试音频文件")
        print(f"   文件路径: {test_audio_path}")
        
        import kaldiio
        sample_rate, audio_data = kaldiio.load_mat(test_audio_path)
        
        duration = len(audio_data) / sample_rate
        print(f"   ✅ 音频读取成功")
        print(f"   - 采样率: {sample_rate} Hz")
        print(f"   - 时长: {duration:.2f} 秒")
        print(f"   - 采样数: {len(audio_data)}")
        print(f"   - 数据类型: {audio_data.dtype}")
        print(f"   - 形状: {audio_data.shape}")
        print(f"   - 数值范围: [{audio_data.min()}, {audio_data.max()}]")
        
        # 确保是 int16 格式
        if audio_data.dtype != np.int16:
            print(f"   ⚠️  转换数据类型: {audio_data.dtype} -> int16")
            audio_data = audio_data.astype(np.int16)
        
        # 解码配置
        decode_config = {
            "use_gpu": 1,  # 使用GPU
            "beam_size": 1,
            "decode_max_len": 512,
            "temperature": 1.0,
            "repetition_penalty": 1.0,
        }
        
        # 3. 测试方法1：使用文件路径（传统方式）
        print(f"\n【步骤3】 方法1 - 使用文件路径输入（传统方式）")
        batch_uttid_1 = ["test_file_path"]
        batch_wav_input_1 = [test_audio_path]
        
        print("   执行识别...")
        start_time = time.time()
        results_1 = model.transcribe(batch_uttid_1, batch_wav_input_1, decode_config)
        elapsed_1 = time.time() - start_time
        
        print("✅ 识别完成")
        print(f"\n   结果:")
        print(f"   - 音频ID: {results_1[0]['uttid']}")
        print(f"   - 识别文本: {results_1[0]['text']}")
        print(f"   - 模型RTF: {results_1[0]['rtf']}")
        print(f"   - 总耗时: {elapsed_1:.4f} 秒")
        
        # 4. 测试方法2：使用numpy数组（新方式）
        print(f"\n【步骤4】 方法2 - 使用numpy数组输入（新方式）")
        batch_uttid_2 = ["test_numpy_array"]
        batch_wav_input_2 = [(sample_rate, audio_data)]
        
        print("   执行识别...")
        start_time = time.time()
        results_2 = model.transcribe(batch_uttid_2, batch_wav_input_2, decode_config)
        elapsed_2 = time.time() - start_time
        
        print("✅ 识别完成")
        print(f"\n   结果:")
        print(f"   - 音频ID: {results_2[0]['uttid']}")
        print(f"   - 识别文本: {results_2[0]['text']}")
        print(f"   - 模型RTF: {results_2[0]['rtf']}")
        print(f"   - 总耗时: {elapsed_2:.4f} 秒")
        
        # 5. 对比结果
        print(f"\n【步骤5】 结果对比")
        print("   " + "=" * 66)
        print(f"   {'方法':<20} {'耗时(秒)':<12} {'识别文本'}")
        print("   " + "-" * 66)
        print(f"   {'文件路径输入':<20} {elapsed_1:<12.4f} {results_1[0]['text'][:30]}...")
        print(f"   {'numpy数组输入':<20} {elapsed_2:<12.4f} {results_2[0]['text'][:30]}...")
        print("   " + "=" * 66)
        
        # 检查识别结果是否一致
        text1 = results_1[0]['text']
        text2 = results_2[0]['text']
        
        if text1 == text2:
            print(f"\n   ✅ 识别结果完全一致！")
        else:
            print(f"\n   ⚠️  识别结果略有差异")
            print(f"   文件路径: {text1}")
            print(f"   numpy数组: {text2}")
        
        # 性能对比
        speedup = ((elapsed_1 - elapsed_2) / elapsed_1) * 100
        if speedup > 0:
            print(f"   📊 numpy数组方式快 {speedup:.1f}%")
        else:
            print(f"   📊 两种方式性能相当 (差异 {abs(speedup):.1f}%)")
        
        print("\n" + "=" * 70)
        print("✅ 测试通过！numpy数组输入功能正常工作")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_numpy_input()

