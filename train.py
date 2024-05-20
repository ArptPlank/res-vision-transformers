import torch
from data_process import data
from vit_original import vit
from tqdm import tqdm
import os
import shutil
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import threading
class train():
    def __init__(self):
        if os.path.exists('./logs') and os.path.isdir('./logs'):
            shutil.rmtree('./logs')
        if os.path.exists('./model') and os.path.isdir('./model'):
            shutil.rmtree('./model')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = 24
        self.size = 224
        data_set = data(self.batch_size,self.size)
        self.train_data, self.val_data = data_set.get_data()
        # 定义模型
        self.model = vit(pretrained=True,num_classes=2).to(self.device)
        print(f"The model has {self.count_parameters():,} parameters.")
        print("train on ", self.device)
        # 定义损失函数
        self.criterion = torch.nn.CrossEntropyLoss()
        # 定义优化器
        self.optimizer = torch.optim.Adam([
            {'params': self.model.pos_embed, 'lr': 1e-3},
            {'params': (param for name, param in self.model.named_parameters() if 'pos_embed' not in name),
             'lr': 1e-4}
        ])
        # 按照损失是否减小定义学习率衰减
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=100,verbose=True,min_lr=1e-8)
        self.writer = SummaryWriter(log_dir='logs')

    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters())

    def train(self):
        self.pbar = tqdm(range(1000))
        e = 0
        for epoch in self.pbar:
            self.accuracy = self.test(epoch)
            self.writer.add_scalar(tag="accuracy",  # 可以暂时理解为图像的名字
                              scalar_value=self.accuracy,  # 纵坐标的值
                              global_step=epoch # 当前是第几次轮，可以理解为横坐标的值
                              )
            #self.pbar.set_postfix({"准确率": self.accuracy,"剩余显存大小":self.get_available_memory()})
            t = threading.Thread(target=self.set_available_memory)
            t.start()
            self.model.train()
            for index,(batch_data,batch_target) in enumerate(self.train_data):
                torch.cuda.empty_cache()
                batch_data = batch_data.to(self.device)
                batch_target = batch_target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model.forward(batch_data)
                loss = self.criterion(output,batch_target)
                loss.backward()
                self.optimizer.step()
                self.scheduler.step(loss)
                torch.cuda.empty_cache()
                self.pbar.set_postfix({"准确率": self.accuracy,"loss": loss.item(), "剩余显存大小": self.get_available_memory()})
                self.writer.add_scalar(tag="loss",
                                       scalar_value=loss.item(),
                                       global_step=e
                                       )
                e += 1

    def test(self,epoch):
        self.model.eval()
        num = 0
        correct = 0
        for index, (batch_data, batch_target) in enumerate(self.val_data):
            batch_data = batch_data.to(self.device)
            batch_target = batch_target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model.forward(batch_data)
            #对每一行使用softmax
            output = F.softmax(output,dim=1)
            output = torch.round(output)
            correct += (output == batch_target).all(dim=1).sum().item()  # all(dim=1) 检查每一行是否全部正确
            num += self.batch_size
        #保存模型
        if not os.path.exists('./vit_original'):
            os.mkdir('./vit_original')
        torch.save(self.model.state_dict(), f'./vit_original/model_{epoch}.pth')
        return correct/num

    def get_available_memory(self,device_id=0):
        device = torch.device(f"cuda:{device_id}")
        prop = torch.cuda.get_device_properties(device)
        cached = torch.cuda.memory_reserved(device) / (1024 * 1024)  # 这里是预留/缓存的显存
        total = prop.total_memory / (1024 * 1024)  # 总显存

        available_memory = total - cached  # 计算可用显存
        return available_memory  # 返回可用显存大小，单位：MB

    def set_available_memory(self):
        #while True:
        self.pbar.set_postfix({"准确率": self.accuracy, "剩余显存大小": self.get_available_memory()})


if __name__ == "__main__":
    train = train()
    train.train()