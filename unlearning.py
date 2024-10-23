import os
from copy import deepcopy

import numpy as np
import torch
from torch import optim, nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import BatchflowData
from utils.base import basetrain
from utils.flowfeatures import flowfeatures
from utils.exemplar import Exemplar
from models.cnnmodel import ResNet


class unlearn(basetrain):

    def __init__(self, total_cls, num_list):
        super().__init__(num_list)
        self.previous_model = None
        self.total_cls = total_cls
        self.seen_cls = total_cls
        # self.dataset = flowfeatures()
        self.model = ResNet(classes=self.total_cls).cuda()
        #total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        #print("Solver total trainable parameters : ", total_params)

    def stage(self, train_data, criterion, optimizer, inc_i, status):
        print("Training ... ")
        losses_fine_tune = []
        for i, (x, label) in enumerate(tqdm(train_data)):
            x = x.type(torch.FloatTensor)
            x = x.cuda()
            label = label.view(-1).cuda()
            p = self.model(x)
            p, num_output = self.bias_forward_new(p, 1, status)
            # a = p[:, :self.seen_cls]
            loss = criterion(p[:, :self.seen_cls - len(status[1][2])], label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses_fine_tune.append(loss.item())
        print("stage fine_tune loss :", np.mean(losses_fine_tune))

    def train(self, batch_size, epoches, lr, dataset, test, inc, status):

        #total_cls = self.total_cls
        optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=2e-4)
        scheduler = StepLR(optimizer, step_size=70, gamma=0.1)
        criterion = nn.CrossEntropyLoss()
        # dataset = self.dataset
        # status = dataset.multi_dict
        # train, val, test = dataset.getNextClasses(0)
        # print('train:', len(train), 'val:', len(val), 'test:', len(test))

        train_x, train_y = zip(*dataset)
        train_ys_ft_new, label_mapping_fine_tune = self.automate_label_mapping(train_y)
        train_loader = DataLoader(BatchflowData(train_x, train_ys_ft_new),
                                  batch_size=batch_size, shuffle=True, drop_last=True)
        # val_x, val_y = zip(*val)
        test_x, test_y = zip(*test)
        test_ys_ft_new, label_mapping_fine_tune = self.automate_label_mapping(test_y)
        test_data = DataLoader(BatchflowData(test_x, test_ys_ft_new),
                               batch_size=batch_size, shuffle=True, drop_last=True)

        state_dict = torch.load(f'./saved_models/client{inc}/model+{inc}.pth')
        self.model.load_state_dict(state_dict)
        # test_x = [x for x, y in zip(test_x, test_y) if y not in status[1][2]]
        # = [y for y in test_y if y not in status[1][2]]
        # train_xs_ft = [x for x, y in zip(train_x, train_y) if y not in status[1][2]]
        # train_ys_ft = [y for y in train_y if y not in status[1][2]]
        # test_data = DataLoader(BatchflowData(test_x, test_y),
        # batch_size=batch_size, shuffle=False)

        self.model.to(self.device)
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.fc.parameters():
            param.requires_grad = True
        for epoch in range(epoches):  # 假设训练5个epoch

            cur_lr = self.get_lr(optimizer)
            print("Current Learning Rate : ", cur_lr)
            self.model.train()
            for _ in range(len(self.bias_layers)):
                self.bias_layers[_].eval()
            self.stage_reduce(train_loader, criterion, optimizer, 1, status,
                              label_mapping_fine_tune)
            scheduler.step()
        for param in self.model.parameters():
            param.requires_grad = True
        self.previous_model = deepcopy(self.model)

        if not os.path.exists(f'./saved_models/client{inc}'):
            os.makedirs(f'./saved_models/client{inc}')
        torch.save(self.model.state_dict(), f'./saved_models/client{inc}/model+{inc}.pth')
        acc = self.test_data(test_data, label_mapping_fine_tune, inc=0, status=status)

        return self.model.state_dict()

    def stage_reduce(self, train_data, criterion, optimizer, inc_i, status, label_mapping_fine_tune):
        print("Training ... ")
        losses_fine_tune = []
        for i, (x, label) in enumerate(tqdm(train_data)):
            x = x.type(torch.FloatTensor)
            x = x.cuda()
            label = label.view(-1).cuda()
            p = self.model(x)
            #p, num_output = self.bias_forward_new(p, 1, status)
            # a = p[:, :self.seen_cls]

            loss = criterion(p[:, :self.seen_cls - len(status[1][2])], label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses_fine_tune.append(loss.item())
        print("stage fine_tune loss :", np.mean(losses_fine_tune))

    def test_data(self, testdata, mappping, inc, status):
        #print("test data number : ", len(testdata))
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
            pred = p[:, :self.seen_cls - len(status[1][2])].argmax(dim=-1)
            #pred_leverage = self.inverse_label_mapping(pred, mappping)
            correct += sum(pred == label).item()
            wrong += sum(pred != label).item()

            pred_list.append(pred)
            label_list.append(label)

        pred_py = torch.cat(pred_list, dim=0)
        label_py = torch.cat(label_list, dim=0)

        pred_arr = pred_py.detach().cpu().numpy()
        label_arr = label_py.detach().cpu().numpy()
        #print('predict_label : {}'.format(list(set(pred_arr))))
        #print('true_label : {}'.format(list(set(label_arr))))
        acc = correct / (wrong + correct)
        print("Test UL client Acc: {}".format(acc * 100))
        self.model.train()
        return acc
