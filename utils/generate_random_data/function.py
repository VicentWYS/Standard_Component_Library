import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import random
import numpy as np
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 生成一个随机张量
def generate_random_image(size):
    """
    输出：
        type: torch.FloatTensor
        分布：0~1
        shape: [size]
        内容（size=4时）：tensor([0.0612, 0.8588, 0.4811, 0.4368])
    """
    random_data = torch.rand(size)
    return random_data


# 生成一个由正态分布（高斯分布）中随机采样的随机张量
def generate_random_seed(size):
    random_data = torch.randn(size)
    return random_data
