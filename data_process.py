import multiprocessing
from PIL import Image
import os
from torchvision import transforms
import torch
from tqdm import tqdm
import threading
from torch.utils.data import DataLoader, Subset, TensorDataset ,random_split
from gc import collect
import numpy as np
class data():
    def __init__(self,batch_size,size):
        self.data = []
        self.target = []
        self.all_num = 0
        self.complete_num = 0 # 完成的线程数
        self.condition = threading.Condition()
        self.transformers = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.Grayscale(num_output_channels=3),  # 确保所有图像都有 3 个通道
            transforms.ToTensor(),
        ])
        # 读取猫的图片,先自动识别为文件夹内图片的数量
        covid_num, covid_list = self.count_images('./COVID-19')
        normal_num, normal_list = self.count_images('./NORMAL')
        self.all_num = covid_num + normal_num
        #读取covid图片
        for i in tqdm(range(covid_num)):
            img = Image.open(f"./COVID-19/{covid_list[i]}")
            img = self.transformers(img)
            self.data.append(img)
            self.target.append(torch.tensor([1.0,0.0]))
        #读取normal图片
        for i in tqdm(range(normal_num)):
            img = Image.open(f"./NORMAL/{normal_list[i]}")
            img = self.transformers(img)
            self.data.append(img)
            self.target.append(torch.tensor([0.0,1.0]))

        print("数据加载完成，开始处理数据,completenum:", self.complete_num)
        self.data_set = torch.stack(tuple(self.data), dim=0)
        self.target_set = torch.stack(tuple(self.target), dim=0)
        del self.data, self.target
        collect()
        self.transformers2 = self.count_mean_std(self.data_set)
        self.data_set = self.transformers2(self.data_set)
        print(self.data_set.shape)
        self.dataset = TensorDataset(self.data_set, self.target_set)
        del self.data_set, self.target_set
        collect()
        # 定义训练集和验证集的大小ddd
        train_size = int(0.8 * len(self.dataset))  # 假设训练集占80%
        val_size = len(self.dataset) - train_size  # 剩余的部分用作验证集
        # 使用random_split分割数据集
        self.train_dataset, self.val_dataset = random_split(self.dataset, [train_size, val_size])
        self.train_dataset = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=5)
        self.val_dataset = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=True, num_workers=5)

    #读取图片数量
    def count_images(self,path):
        list = os.listdir(path)
        num = len(list)
        return num, list


    def get_data(self):
        return self.train_dataset,self.val_dataset

    def count_mean_std(self,data):
        print("开始计算归一化参数")
        nb_samples = 0.
        channel_mean = torch.zeros(3)
        channel_std = torch.zeros(3)
        N, C, H, W = data.shape[:4]
        data = data.view(N, C, -1)
        channel_mean += data.mean(2).sum(0)
        channel_std += data.std(2).sum(0)
        nb_samples += N
        channel_mean /= nb_samples
        channel_std /= nb_samples
        print(f"归一化参数计算完成mean为{channel_mean}，std为{channel_std},开始进行归一化操作")
        return transforms.Compose([transforms.Normalize(channel_mean, channel_std)])


if __name__ == "__main__":
    data = data(batch_size=32,size=224)
    train_data,val_data = data.get_data()
    print("train_data_len:",len(train_data))
    print("val_data_len:",len(val_data))
