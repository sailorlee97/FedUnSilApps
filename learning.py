import pandas as pd
from sklearn.metrics import classification_report
from torch import nn, optim
from torch.optim.lr_scheduler import StepLR

from dataset import BatchflowData
from utils.args import parser_args
from utils.datasets import *
import copy
import random
import dill
import datetime
from tqdm import tqdm
import numpy as np
import math
import torch
from torch.utils.data import DataLoader
import time
from models.alexnet import AlexNet
from models.alexnet_ul import AlexNet_UL
from utils.base import basetrain
from utils.plot_ml import plot_conf
from utils.trainer_private import TrainerPrivate, TesterPrivate
from utils.flowfeatures import flowfeatures
from utils.exemplar import Exemplar
from models.cnnmodel import ResNet
from models.model import BiasLayer

class learn(basetrain):
    def __init__(self, total_cls, num_list):
        super().__init__(num_list)
        self.total_cls = total_cls
        self.seen_cls = 0
        #self.dataset = flowfeatures()
        self.model = ResNet(classes=self.total_cls).cuda()
        # print(self.model)
        # self.model = nn.DataParallel(self.model, device_ids=[0,1])
        #self.bias_layer1 = BiasLayer().cuda()
        #self.bias_layer2 = BiasLayer().cuda()
        #self.bias_layer3 = BiasLayer().cuda()
        #self.bias_layer4 = BiasLayer().cuda()
        # self.bias_layer5 = BiasLayer().cuda()
        #self.bias_layers=[self.bias_layer1, self.bias_layer2, self.bias_layer3, self.bias_layer4]
        #self.sample = num_list
        #self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        #print("Solver total trainable parameters : ", total_params)

    def train(self,batch_size, epoches, lr, dataset,test, inc, status):
        total_cls = self.total_cls
        optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=2e-4)
        scheduler = StepLR(optimizer, step_size=70, gamma=0.1)
        criterion = nn.CrossEntropyLoss()
        #exemplar = Exemplar(8000, total_cls)
        #previous_model = None
        #dataset = self.dataset
        #status = dataset.multi_dict
        #train, val, test = dataset.getNextClasses(0)
        #print('train:', len(train), 'val:', len(val), 'test:', len(test))
        train_x, train_y = zip(*dataset)
        #val_x, val_y = zip(*val)
        test_x, test_y = zip(*test)
        test_accs = []

        train_data = DataLoader(BatchflowData(train_x, train_y),
                                batch_size=batch_size, shuffle=True, drop_last=True)
        test_data = DataLoader(BatchflowData(test_x, test_y),
                                batch_size=batch_size, shuffle=True, drop_last=True)
        #exemplar.update(self.sample[0], (train_x, train_y), (val_x, val_y))

        self.seen_cls = 10
        print("seen cls number : ", self.seen_cls)
        # val_xs, val_ys = exemplar.get_exemplar_val()
        # val_bias_data = DataLoader(BatchflowData(val_xs, val_ys), batch_size=16, shuffle=True, drop_last=False)
        test_acc = []

        for epoch in range(epoches):
            print("---" * 50)
            print("Epoch", epoch)

            cur_lr = self.get_lr(optimizer)
            print("Current Learning Rate : ", cur_lr)
            self.model.train()
            for _ in range(len(self.bias_layers)):
                self.bias_layers[_].eval()
            self.stage(train_data, criterion, optimizer, 0, status)
            scheduler.step()
            # acc =  self.test_data(test_data,label_mapping, inc= inc_i, status = status)
        #for i, layer in enumerate(self.bias_layers):
        #    layer.printParam(i)

        # 一次增量结束之后，保存本次模型
        #self.previous_model = copy.deepcopy(self.model)
        acc = self.test_data(test_data, test_y, inc=0, status=status)
        test_acc.append(acc)
        test_accs.append(max(test_acc))
        #print('test_accs:', test_accs)
        if not os.path.exists(f'./saved_models/client{inc}'):
            os.makedirs(f'./saved_models/client{inc}')
        torch.save(self.model.state_dict(), f'./saved_models/client{inc}/model+{inc}.pth')


        return self.model.state_dict()

    def stage(self,train_data, criterion, optimizer, inc_i, status):
        print("Training ... ")
        losses = []
        for i, (x, label) in enumerate(tqdm(train_data)):
            x = x.type(torch.FloatTensor)
            x = x.cuda()
            label = label.view(-1).cuda()
            p = self.model(x)
            p, num_output = self.bias_forward_new(p, 0, status)
            # a = p[:, :self.seen_cls]
            loss = criterion(p[:, :self.seen_cls], label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print("stage loss :", np.mean(losses))

    def test_data(self, testdata, mappping, inc, status):

        print("test data number : ", len(testdata))
        self.model.eval()
        correct = 0
        wrong = 0
        pred_list = []
        label_list = []

        for i, (x, label) in enumerate(testdata):
            x = x.type(torch.FloatTensor)
            x = x.cuda()
            label = label.view(-1).cuda()
            p = self.model(x)
            p, num_output = self.bias_forward_new(p, inc, status)
            pred = p[:, :self.seen_cls].argmax(dim=-1)
            #pred_leverage = self.inverse_label_mapping(pred, mappping)
            correct += sum(pred == label).item()
            wrong += sum(pred != label).item()

            pred_list.append(pred)
            label_list.append(label)

        pred_py = torch.cat(pred_list, dim=0)
        label_py = torch.cat(label_list, dim=0)

        #pred_arr = pred_py.detach().cpu().numpy()
        #label_arr = label_py.detach().cpu().numpy()
        #print('predict_label : {}'.format(list(set(pred_arr))))
        #print('true_label : {}'.format(list(set(label_arr))))
        #res_key = list(map(str, mappping.keys()))
        #plot_conf(pred_arr, label_arr, res_key, name=inc)
        #report = classification_report(label_arr, pred_arr, digits=4, target_names=res_key, output_dict=True)

        #print(report)
        #df = pd.DataFrame(report).transpose()
        #df.to_csv("{}.csv".format(inc), index=True)

        acc = correct / (wrong + correct)
        print("Test Acc: {}".format(acc * 100))
        self.model.train()
        print("---------------------------------------------")
        return acc

if __name__ == '__main__':
    trainlib = learn(10,[10,0])
    trainlib.train(256,50,0.001,8000)