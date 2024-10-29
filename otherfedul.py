import copy
import math
import os
import time
from collections import defaultdict

from utils.datasets import DatasetSplit
from utils.trainer_private import TrainerPrivate, TesterPrivate
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.alexnet import AlexNet
from models.alexnet_ul import AlexNet_UL
from dataset import BatchflowData
from utils.Aggregator import FedAggregator
from utils.flowfeatures import flowfeatures
from models.cnnmodel import ResNet
from learning import learn
from unlearning import unlearn
from utils.base import basetrain

class fedau(basetrain):

    def __init__(self, epochs, num_class, incremental_num_list):
        super().__init__(incremental_num_list)
        self.epochs = epochs
        self.total_cls = num_class
        self.num_users = 10
        self.idxs_users = [0, 1, 3, 4, 5, 6, 7, 8, 9]
        self.total_users = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.ul_clients = [2]
        self.dataset = flowfeatures(self.ul_clients)
        self.learn = learn(num_class, [num_class, 0])
        self.unlearn = unlearn(num_class, [num_class, 0])
        #self.model = ResNet(classes=num_class).cuda()
        #self.model_ul = ResNet(classes=num_class).cuda()
        self.w_t = AlexNet(num_classes=self.total_cls, in_channels=1).state_dict()
        self.first_num = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
        self.second_num = {2, 3, 4, 5, 6, 7, 8, 9}
        self.lr = 0.01
        self.local_ep = 1
        self.sigma = 0.1
        self.dp =False
        self.optim = 'sgd'
        self.ul_mode = 'ul_class'
        self.model = None
        self.model_ul = None
        self.construct_model()
        self.trainer = TrainerPrivate(self.model, self.device, self.dp, self.sigma,self.total_cls,'none')
        self.trainer_ul=TrainerPrivate(self.model_ul, self.device, self.dp, self.sigma,self.total_cls,self.ul_mode)
        self.tester = TesterPrivate(self.model, self.device)



    def get_data(self, mode):

        if mode == 'remain':
            train, val, test = self.dataset.getRemainClass({0, 1, 2, 3, 4, 5, 6, 7, 8, 9},{ 2, 3, 4, 5, 6, 7, 8, 9})
        elif mode == 'ul':
            train, val, test = self.dataset.getul({0, 1, 2, 3, 4, 5, 6, 7, 8, 9},{ 2, 3, 4, 5, 6, 7, 8, 9})
        elif mode == 'all':
            train, val, test = self.dataset.getallclass()
        else:
            raise RuntimeError('no mode!!!')

        return train, val, test, self.dataset.multi_dict

    def construct_model(self):
        model = AlexNet(num_classes=self.total_cls, in_channels=1)
        self.model = model.to(self.device)
        model_ul = AlexNet_UL(num_classes=self.total_cls, in_channels=1)
        self.model_ul = model_ul.to(self.device)
        print('Total params: %.2f' % (sum(p.numel() for p in model.parameters())))


    def fedtrain(self, batch_size, lr):

        self.dataset = flowfeatures(self.ul_clients,self.idxs_users)
        local_ws = defaultdict(list)
        clients_train_data, clients_val_data, clients_test_data, multi_dict = self.get_data('remain')
        ul_clients_train_data, ul_clients_test_data, ul_clients_val_data, multi_dict = self.get_data('ul')
        test = self.dataset.getglobalallclass({0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, {2, 3, 4, 5, 6, 7, 8, 9})
        idxs_users = [0,1,2,3,4,5,6,7,8,9]
        ul_state_dicts = {}
        for i in self.ul_clients:
            ul_state_dicts[i] = copy.deepcopy(self.model_ul.state_dict())

        for epoch in range(self.epochs):
            local_models_per_epoch = []
            global_state_dict = copy.deepcopy(self.model.state_dict())
            local_ws, local_losses, = [], []
            for idx in tqdm(self.total_users, desc='Epoch:%d, lr:%f' % (epoch, self.lr)):
                if (idx in self.ul_clients) == False:

                    # print(idx,"True1000000")
                    self.model.load_state_dict(global_state_dict)  # 还原 global model
                    start_normal = time.time()
                    local_w, local_loss = self.trainer._local_update(clients_train_data[idx], self.local_ep, self.lr,
                                                                     self.optim)
                    end_normal = time.time()
                    local_ws.append(copy.deepcopy(local_w))
                    local_losses.append(local_loss)
                    if 'federaser' in self.ul_mode or 'amnesiac_ul_samples' in self.ul_mode:
                        local_models_per_epoch.append(copy.deepcopy(local_w))
                else:
                    if self.ul_mode.startswith('ul_'):

                        # print("ul-idx:",idx)
                        self.model_ul.load_state_dict(ul_state_dicts[idx])
                        # ul_model除W2外替换为global model的参数
                        #self.model_ul.load_state_dict(global_state_dict, strict=False)
                        # ul_client时， W2基于W1训练：
                        if self.ul_mode == 'ul_samples_whole_client' or (
                                self.ul_mode == 'ul_samples_backdoor' and self.dataset == 'cifar100'):
                            # print('Learn based on W1...')
                            gamma = 0.5
                            temp_state_dict = copy.deepcopy(self.model_ul.state_dict())
                            temp_state_dict['classifier_ul.weight'] = (1 - gamma) * temp_state_dict[
                                'classifier.weight'] + gamma * temp_state_dict['classifier_ul.weight']
                            temp_state_dict['classifier_ul.bias'] = (1 - gamma) * temp_state_dict[
                                'classifier.bias'] + gamma * temp_state_dict['classifier_ul.bias']
                            self.model_ul.load_state_dict(temp_state_dict)
                        # 参数替换完毕，开始训练
                        start_ul = time.time()
                        local_w_ul, local_loss, classify_loss, normalize_loss = self.trainer_ul._local_update_ul(
                            ul_clients_train_data[idx], self.local_ep, self.lr, self.optim, self.ul_clients)
                        end_ul = time.time()
                        # 本次ul_model结果保存（用于下轮更新W2）
                        ul_state_dicts[idx] = copy.deepcopy(local_w_ul)
                        # 提取W1 (全局模型加载W1，保存到待avg列表中)
                        self.model.load_state_dict(local_w_ul, strict=False)

                        # class_loss,class_acc=self.trainer.test(ul_ldr)
                        # print('**** local class loss: {:.4f}  local class acc: {:.4f}****'.format(class_loss,class_acc))

                        local_ws.append(copy.deepcopy(self.model.state_dict()))



        client_weights = [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]

        self.fed_avg(local_ws, client_weights, 1)
        self.model.load_state_dict(self.w_t)

        test_x, test_y = zip(*test)
        test_loader = DataLoader(BatchflowData(test_x, test_y),
                             batch_size=256, shuffle=True, drop_last=True)

        loss_val_mean, acc_val_mean = self.trainer.test(test_loader)
        print('ACC: ',acc_val_mean)


    def fed_avg(self, local_ws, client_weights, lr_outer):

        w_avg = copy.deepcopy(local_ws[0])

        # client_weight=1.0/len(local_ws)
        # print('client_weights:',client_weights)

        for k in w_avg.keys():
            w_avg[k] = w_avg[k] * client_weights[0]

            for i in range(1, len(local_ws)):
                w_avg[k] += local_ws[i][k] * client_weights[i] * lr_outer

            self.w_t[k] = w_avg[k]


if __name__ == '__main__':
    fd = fedau(50,10,[10,0])
    fd.fedtrain(256,0.01)