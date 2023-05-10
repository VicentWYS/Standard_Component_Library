import torch
import os
import matplotlib.pyplot as plt
from utils.get_data import generate_random_seed
from generator import Generator

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ---------------------- 可视化G生成效果 ----------------------
G = Generator()
checkpoint = torch.load("./save/checkpoint/G_train_mnist_improved.pth")  # 加载检查点
G.load_state_dict(checkpoint["model_state_dict"])  # 模型加载参数

f, ax = plt.subplots(2, 3, figsize=(16, 8))  # f是整个图形窗口的对象,ax是一个2x3的NumPy数组，其中包含6个子图对象
for i in range(2):
    for j in range(3):
        g_out = G.forward(generate_random_seed(100))
        image = g_out.detach().numpy().reshape(28, 28)
        ax[i][j].imshow(image, interpolation='none', cmap='Blues')
plt.savefig("./save/image/G_output.png", dpi=1000)
plt.show()
plt.close()
