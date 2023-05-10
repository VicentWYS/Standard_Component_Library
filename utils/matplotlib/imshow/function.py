"""
显示一张图片
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
import matplotlib.pyplot as plt
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def plot_image(self, index):
    # 获取数据
    image = self.data.iloc[index, 1:].values.reshape(28, 28)
    plt.figure()
    plt.title('label = ' + str(self.data.iloc[index, 0]))
    plt.imshow(image, interpolation='none', cmap='Blues')
    plt.savefig('./save/image/image_' + str(self.data.iloc[index, 0]) + '.png', dpi=1000)
    plt.show()
    plt.close()
