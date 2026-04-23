#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
目标：
1. 从 FineWeb 流式抽样到约 1.5GB
2. 读取 MNBVC gov 子集到约 1.0GB
3. 从 The Stack v2 Python 流式抽样到约 0.3GB
4. 清洗、去重、写入 data/cleaned/cpt/*.jsonl

当前版本为占位文件，后续可继续补充：
- 长度过滤
- exact hash / minhash 去重
- 中英混合比例控制
- 代码样本单独配额
"""

def main():
    print("TODO: build CPT corpus from raw data.")


if __name__ == "__main__":
    main()
