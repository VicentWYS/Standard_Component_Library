import torch
from torch.utils.data import Dataset
import pandas as pd
import matplotlib.pyplot as plt
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class MnistDataset(Dataset):
    def __init__(self, csv_path=r'./data/mnist_train.csv'):
        self.data = pd.read_csv(csv_path, header=None)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        label = self.data.iloc[index, 0]  # 标签
        target = torch.zeros(10)  # 标签 one-hot格式
        target[label] = 1.0

        image = torch.FloatTensor(self.data.iloc[index, 1:].values) / 255.0

        return label, image, target

    def plot_image(self, index):
        # 获取数据
        image = self.data.iloc[index, 1:].values.reshape(28, 28)
        plt.figure()
        plt.title('label = ' + str(self.data.iloc[index, 0]))
        plt.imshow(image, interpolation='none', cmap='Blues')
        plt.savefig('./save/image/image_' + str(self.data.iloc[index, 0]) + '.png', dpi=1000)
        plt.show()
        plt.close()


if __name__ == '__main__':
    train_data = MnistDataset()
    train_data.plot_image(17)
