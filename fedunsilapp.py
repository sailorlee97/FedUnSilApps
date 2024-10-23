import copy
import os
import time
from collections import defaultdict

import torch
from torch.utils.data import DataLoader

from dataset import BatchflowData
from utils.Aggregator import FedAggregator
from utils.flowfeatures import flowfeatures
from models.cnnmodel import ResNet
from learning import learn
from unlearning import unlearn
from utils.base import basetrain

class fedunsilapps(basetrain):
    def __init__(self, epochs, num_class, incremental_num_list):
        super().__init__(incremental_num_list)
        self.epochs = epochs
        # self.total_cls = num_class
        self.idxs_users = 10
        self.ul_clients = [2]
        self.dataset = flowfeatures(self.ul_clients)
        self.learn = learn(num_class, [num_class, 0])
        self.unlearn = unlearn(num_class, [num_class, 0])
        self.model = ResNet(classes=num_class).cuda()
        self.model_ul = ResNet(classes=num_class).cuda()
        self.w_t = copy.deepcopy(self.model.state_dict())
        self.first_num = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
        self.second_num = {2, 3, 4, 5, 6, 7, 8, 9}
        #print(len(self.second_num))

    def get_data(self, mode):

        if mode == 'remain':
            train, val, test = self.dataset.getRemainClass()
        elif mode == 'ul':
            train, val, test = self.dataset.getul()
        elif mode == 'all':
            train, val, test = self.dataset.getallclass()
        else:
            raise RuntimeError('no mode!!!')

        return train, val, test, self.dataset.multi_dict

    def splitdata(self, train):

        train, label = zip(*train)

        return train, label

    def fedtrain(self, batch_size, lr):
        # 获取数据

        # 这里用于存放遗忘学习的参数
        #ul_state_dicts = []
        # for epoch in range(self.epochs):
        #local_models_per_epoch = []
        #global_state_dict = copy.deepcopy(self.model.state_dict())
        local_ws = defaultdict(list)
        clients_train_data, clients_val_data, clients_test_data, multi_dict = self.get_data('all')
        ul_clients_train_data, ul_clients_test_data, ul_clients_val_data, multi_dict = self.get_data('remain')

        start = time.time()
        for idx in range(self.idxs_users):
            print("---" * 15,f"client:{idx}","---" * 15)
            if (idx in self.ul_clients) == False:
                # learn
                start_ul = time.time()
                local_w = self.learn.train(batch_size, self.epochs, lr, clients_train_data[idx], clients_test_data[idx],
                                           idx, multi_dict)
                local_ws[idx].extend(copy.deepcopy(local_w))
                end_ul = time.time()

            else:
                # unlearn
                # self.model_ul.load_state_dict(ul_state_dicts[idx])
                # ul_model除W2外替换为global model的参数
                # self.model_ul.load_state_dict(global_state_dict, strict=False)
                start_ul = time.time()
                local_w = self.learn.train(batch_size, self.epochs, lr, clients_train_data[idx], clients_test_data[idx],
                                           idx, multi_dict)
                local_w_ul = self.unlearn.train(batch_size, self.epochs, lr, ul_clients_train_data[idx], ul_clients_test_data[idx],
                                                idx, multi_dict)
                end_ul = time.time()
                local_ws[idx].extend(copy.deepcopy(local_w_ul))

        #client_weights = []
        #for i in range(self.idxs_users):
        #    client_weights.append(1 / self.idxs_users)

        #print(local_ws)
        #w_avg1, w_avg2 = self.fed_avg_features_fc(local_ws, client_weights, 1, self.ul_clients)
        aggregator = FedAggregator(model_path='./saved_models/', forget_clients=self.ul_clients, feature_key='features', fc_key='fc')
        non_forget_model, forget_model = aggregator.aggregate_models()
        
        if not os.path.exists(f'./saved_models/globalModel/'):
            os.makedirs(f'./saved_models/globalModel/')
        torch.save(non_forget_model, f'./saved_models/globalModel/non_forget_model.pth')
        torch.save(forget_model, f'./saved_models/globalModel/forget_model.pth')
        # 测试 w_avg2 这里的测试需要将遗忘的类别去掉
        testdata = self.dataset.getgobalclass(self.first_num,self.second_num) 
        test_x, test_y = zip(*testdata)
        new_labels, label_mapping = self.automate_label_mapping(test_y)
        test_data = DataLoader(BatchflowData(test_x, new_labels),
                               batch_size=batch_size, shuffle=True, drop_last=True)

        self.test_globalmodel(forget_model,test_data,label_mapping)

    def test_globalmodel(self, w_avg2, testdata,mappping):

        self.model.load_state_dict(w_avg2)
        correct = 0
        wrong = 0
        for i, (x, label) in enumerate(testdata):
            x = x.type(torch.FloatTensor)
            x = x.cuda()
            label = label.view(-1).cuda()
            p = self.model(x)
            pred = p[:, :len(self.second_num)].argmax(dim=-1)
            pred_leverage = self.inverse_label_mapping(pred, mappping)
            correct += sum(pred == label).item()
            wrong += sum(pred != label).item()

        acc = correct / (wrong + correct)
        print("Test Ul Acc: {}".format(acc * 100))

    def train_test(self):
        local_ws = []
        state_dict = torch.load('./saved_models/model+0.pth')
        self.model.load_state_dict(state_dict)
        for i in range(10):
            local_ws.append(state_dict)

        client_weights = []

        for i in range(self.idxs_users):
            client_weights.append(1 / self.idxs_users)
            if i in self.ul_clients:
                client_weights[i] = 0.1
        self.fed_avg(local_ws, client_weights, 1)

    def fed_avg_features_fc(self, local_ws, client_weights, lr_outer, forget_client_idx):
        state_dict = torch.load('./saved_models/model+0.pth')
        self.model.load_state_dict(state_dict)

        w_avg1 = copy.deepcopy(local_ws[0])
        w_avg2 = copy.deepcopy(local_ws[0])

        # 计算非遗忘客户端和遗忘客户端的数量
        num_non_forget_clients = self.idxs_users-len(self.ul_clients)
        num_forget_clients = len(self.ul_clients)

        for k in w_avg1:
            # 如果层的名称中包含 'fc'，则分别聚合非遗忘客户端和遗忘客户端的参数
            # 如果层的名称中包含 'fc'，则分别聚合非遗忘客户端和遗忘客户端的参数
            if isinstance(local_ws[0], dict) and 'fc' in k:
                w_avg1[k] = torch.zeros_like(local_ws[0][k])
                w_avg2[k] = torch.zeros_like(local_ws[0][k])
                for i in range(len(local_ws)):
                    if i != forget_client_idx:
                        w_avg1[k] += local_ws[i][k]
                    else:
                        w_avg2[k] += local_ws[i][k]
                if num_non_forget_clients > 0:
                    w_avg1[k] /= num_non_forget_clients
                if num_forget_clients > 0:
                    w_avg2[k] /= num_forget_clients
            else:
                print(local_ws[0][k])
                w_avg1[k] = local_ws[0][k] * client_weights[0]
                w_avg2[k] = local_ws[0][k] * client_weights[0]
                for i in range(1, len(local_ws)):
                    w_avg1[k] += local_ws[i][k] * client_weights[i] * lr_outer
                    w_avg2[k] += local_ws[i][k] * client_weights[i] * lr_outer

        return w_avg1, w_avg2

    def fed_avg(self, local_ws, client_weights, lr_outer):

        w_avg = copy.deepcopy(local_ws[0])

        # client_weight=1.0/len(local_ws)
        # print('client_weights:',client_weights)

        for k in w_avg.keys():
            w_avg[k] = w_avg[k] * client_weights[0]

            for i in range(1, len(local_ws)):
                w_avg[k] += local_ws[i][k] * client_weights[i] * lr_outer

            self.w_t[k] = w_avg[k]


# uint test
if __name__ == '__main__':
    fus = fedunsilapps(50, 10, [10,0])
    fus.fedtrain(256, 0.01)
