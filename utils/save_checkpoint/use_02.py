import torch
import matplotlib.pyplot as plt
import os
from generator import Generator
from discriminator import Discriminator
from dataset import MnistDataset
from utils.get_data import generate_random_seed
from utils.save_checkpoints import save_checkpoint

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

D = Discriminator()
G = Generator()

train_data = MnistDataset()

epochs = 4

for epoch in range(epochs):
    print('epoch = ', epoch + 1)
    for label, image, target in train_data:
        # 用真实数据训练D
        D.train_D(image, torch.FloatTensor([1.0]))

        # G中输入符合正态分布的随机数
        D.train_D(G.forward(generate_random_seed(100)).detach(), torch.FloatTensor([0.0]))

        # G中输入符合正态分布的随机数
        G.train_G(D, generate_random_seed(100), torch.FloatTensor([1.0]))

# 保存训练模型
save_checkpoint(D, './save/checkpoint/D_train_mnist_improved.pth')
save_checkpoint(G, './save/checkpoint/G_train_mnist_improved.pth')
