import copy
import time
from collections import defaultdict

import torch
from utils.flowfeatures import flowfeatures
from models.cnnmodel import ResNet
from learning import learn
from unlearning import unlearn

class fedunsilapps():
    def __init__(self,epochs,num_class):
        self.epochs = epochs
        #self.total_cls = num_class
        self.idxs_users = 10
        self.ul_clients = [5]
        self.learn = learn(num_class,[num_class,0])
        self.unlearn = unlearn(num_class,[num_class,0])
        self.model = ResNet(classes=num_class).cuda()
        self.model_ul = ResNet(classes=num_class).cuda()
        self.w_t = copy.deepcopy(self.model.state_dict())

    def get_data(self,mode):
        dataset = flowfeatures()
        if mode == 'remain':
            train, val, test = dataset.getRemainClass()
        elif mode == 'ul':
            train, val, test = dataset.getul()
        elif mode == 'all':
            train, val, test = dataset.getallclass()
        else:
            raise RuntimeError('no mode!!!')

        return train, val, test,  dataset.multi_dict

    def splitdata(self,train):

        train, label = zip(*train)

        return train,label

    def fedtrain(self, batch_size, lr):
        #获取数据


        # 这里用于存放遗忘学习的参数
        ul_state_dicts = []
        #for epoch in range(self.epochs):
        local_models_per_epoch = []
        global_state_dict = copy.deepcopy(self.model.state_dict())
        local_ws = defaultdict(list)
        clients_train_data, clients_val_data, clients_test_data, multi_dict = self.get_data('all')
        ul_clients_train_data, ul_clients_test_data ,ul_clients_val_data, multi_dict = self.get_data('ul')

        start = time.time()
        for idx in range(self.idxs_users):

            if (idx in self.ul_clients) == False:
                # learn
                start_ul = time.time()
                local_w = self.learn.train(batch_size, self.epochs, lr, clients_train_data[idx], clients_test_data[idx], idx, multi_dict)
                local_ws[idx].extend(copy.deepcopy(local_w))
                end_ul = time.time()

            else:
                # unlearn
                #self.model_ul.load_state_dict(ul_state_dicts[idx])
                # ul_model除W2外替换为global model的参数
                #self.model_ul.load_state_dict(global_state_dict, strict=False)
                start_ul=time.time()
                local_w = self.learn.train(batch_size, self.epochs, lr, clients_train_data[idx], clients_test_data[idx], idx, multi_dict)
                local_w_ul = self.unlearn.train(batch_size, self.epochs, lr, ul_clients_train_data[idx], idx, multi_dict)
                end_ul=time.time()
                local_ws[idx].extend(copy.deepcopy(local_w_ul))

        client_weights = []
        for i in range(self.idxs_users):
            client_weights.append(1 / self.idxs_users)
        w_avg1, w_avg2 = self.fed_avg_features_fc(local_ws, client_weights, 1,self.ul_clients)


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
        self.fed_avg(local_ws,client_weights,1)

    def fed_avg_features_fc(self, local_ws, client_weights, lr_outer, forget_client_idx):
        w_avg1 = copy.deepcopy(local_ws[0])
        w_avg2 = copy.deepcopy(local_ws[0])

        for k in w_avg1.keys():
            # 如果层的名称中包含 'fc'，则分别聚合非遗忘客户端和遗忘客户端的参数
            if 'fc' in k:
                w_avg1[k] = torch.zeros_like(w_avg1[k])
                w_avg2[k] = torch.zeros_like(w_avg2[k])
                for i in range(len(local_ws)):
                    if i != forget_client_idx:
                        w_avg1[k] += local_ws[i][k] * client_weights[i] * lr_outer
                    else:
                        w_avg2[k] += local_ws[i][k] * client_weights[i] * lr_outer
            else:
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

#uint test
if __name__ == '__main__':
    fus = fedunsilapps(2,10)
    fus.fedtrain(256,0.01)