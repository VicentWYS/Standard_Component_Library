import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def plot_d_train_loss(self):
    x = self.counter_list
    y = self.train_loss_list

    plt.figure()
    # 纯折线图，蓝色
    plt.plot(x, y, linewidth=0.2)
    # 点图，带折线
    # plt.plot(x, y, marker=".", markersize=3.0, alpha=0.5, linewidth=0.1, linestyle='dashed')
    plt.title("D train loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.savefig("./save/image/D_train_loss_improved.png", dpi=1000)
    plt.show()
    plt.close()
