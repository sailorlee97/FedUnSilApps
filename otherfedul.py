import copy
import math
import os
import time
from collections import defaultdict
from utils.utils import CpuGpuMonitor
from models.fedrecovery_base import fedrecovery_operation
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

class fedul(basetrain):

    def __init__(self, epochs, num_class, incremental_num_list, mode):
        super().__init__(incremental_num_list)
        self.epochs = epochs
        self.total_cls = num_class
        self.num_users = 10
        self.idxs_users = [0, 1, 3, 4, 5, 6, 7, 8, 9]
        self.total_users = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.ul_clients = [2]
        self.dataset_name = 'mirage'
        self.dataset = flowfeatures(self.ul_clients,self.dataset_name,[0,1,3,4,5,6,7,8,9])
        self.learn = learn(num_class, [num_class, 0],self.dataset_name)
        self.unlearn = unlearn(num_class, [num_class, 0],self.dataset_name)
        #self.model = ResNet(classes=num_class).cuda()
        #self.model_ul = ResNet(classes=num_class).cuda()
        self.w_t = AlexNet(num_classes=self.total_cls, in_channels=1, dataset_name = self.dataset_name).state_dict()
        # self.w_t = AlexNet(num_classes=self.total_cls, in_channels=1).state_dict()
        self.first_num = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
        self.second_num = {2, 3, 4, 5, 6, 7, 8, 9}
        self.lr = 0.01
        self.local_ep = 1
        self.sigma = 0.1
        self.dp =False
        self.optim = 'sgd'
        self.ul_mode = mode
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
        model = AlexNet(num_classes=self.total_cls, in_channels=1, dataset_name = self.dataset_name)
        # model = AlexNet(num_classes=self.total_cls, in_channels=1)
        self.model = model.to(self.device)
        model_ul = AlexNet_UL(num_classes=self.total_cls, in_channels=1, dataset_name = self.dataset_name)
        # model_ul = AlexNet_UL(num_classes=self.total_cls, in_channels=1)
        self.model_ul = model_ul.to(self.device)
        print('Total params: %.2f' % (sum(p.numel() for p in model.parameters())))


    def fedtrain(self, batch_size, lr):

        global local_models_per_epoch, old_local_model_list, old_global_model_list
        #self.dataset = flowfeatures(self.ul_clients,self.idxs_users)
        local_ws = defaultdict(list)
        if 'fedau' in self.ul_mode:

            clients_train_data, clients_val_data, clients_test_data, multi_dict = self.get_data('remain')
            ul_clients_train_data, ul_clients_test_data, ul_clients_val_data, multi_dict = self.get_data('ul')

        elif 'fedrecovery'in self.ul_mode or 'amnesiac' in self.ul_mode:
            clients_train_data, clients_val_data, clients_test_data, multi_dict = self.get_data('all')
            ul_clients_train_data, ul_clients_test_data, ul_clients_val_data, multi_dict = self.get_data('ul')

        # elif 'amnesiac' in self.ul_mode:
        #     clients_train_data, clients_val_data, clients_test_data, multi_dict = self.get_data('remain')
        #     ul_clients_train_data, ul_clients_test_data, ul_clients_val_data, multi_dict = self.get_data('ul')
        else:
            raise RuntimeError('no ul mode!!!')
        ul_state_dicts = {}
        test = self.dataset.getglobalclass({0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, {2, 3, 4, 5, 6, 7, 8, 9})
        if 'fedrecovery' in self.ul_mode:
            old_global_model_list=[]
            old_local_model_list=[]
            old_global_model_list.append(copy.deepcopy(self.model.state_dict()))

        if 'amnesiac' in self.ul_mode:
            update_list=[]
            update_epochs={} # 形状和模型参数字典相同
            for param_tensor in self.model.state_dict():
                if "weight" in param_tensor or "bias" in param_tensor:
                    update_epochs[param_tensor] = torch.zeros_like(self.model.state_dict()[param_tensor]).to(self.device)
            # update_list用于保存各epoch的private update
            update_list.append(update_epochs)
            # 初始化private update的累计量为0
            update_sum=update_list[0]

        for i in self.ul_clients:
            ul_state_dicts[i] = copy.deepcopy(self.model_ul.state_dict())

        monitor = CpuGpuMonitor()
        monitor.start()
        clients_dict = {f"client{i}": 0 for i in range(10)}
        start_total = time.time()
        for epoch in range(self.epochs):
            local_models_per_epoch = []
            if 'amnesiac' in self.ul_mode:
                global_update_epoch=copy.deepcopy(update_list[0])
            global_state_dict = copy.deepcopy(self.model.state_dict())
            local_ws, local_losses, = [], []
            for idx in tqdm(self.total_users, desc='Epoch:%d, lr:%f' % (epoch, self.lr)):
                if (idx in self.ul_clients) == False:
                    # 不是需要遗忘的客户端
                    # print(idx,"True1000000")
                    self.model.load_state_dict(global_state_dict)  # 还原 global model
                    start_normal = time.time()
                    local_w, local_loss = self.trainer._local_update(clients_train_data[idx], self.local_ep, self.lr,
                                                                     self.optim)
                    end_normal = time.time()
                    train_time = end_normal - start_normal
                    clients_dict[f"client{idx}"]+=train_time
                    local_ws.append(copy.deepcopy(local_w))
                    local_losses.append(local_loss)
                    if 'fedrecovery' in self.ul_mode or 'amnesiac_ul_samples' in self.ul_mode:
                        local_models_per_epoch.append(copy.deepcopy(local_w))
                else:
                    if self.ul_mode.startswith('fedau'):

                        # print("ul-idx:",idx)
                        self.model_ul.load_state_dict(ul_state_dicts[idx])
                        # ul_model除W2外替换为global model的参数
                        #self.model_ul.load_state_dict(global_state_dict, strict=False)
                        # ul_client时， W2基于W1训练：
                        # 参数替换完毕，开始训练
                        start_ul = time.time()
                        local_w_ul, local_loss, classify_loss, normalize_loss = self.trainer_ul._local_update_ul(
                            ul_clients_train_data[idx], self.local_ep, self.lr, self.optim, self.ul_clients)
                        end_ul = time.time()
                        ul_time = end_ul - start_ul
                        clients_dict[f"client{idx}"]+=ul_time

                        # 本次ul_model结果保存（用于下轮更新W2）
                        ul_state_dicts[idx] = copy.deepcopy(local_w_ul)
                        # 提取W1 (全局模型加载W1，保存到待avg列表中)
                        self.model.load_state_dict(local_w_ul, strict=False)

                        # class_loss,class_acc=self.trainer.test(ul_ldr)
                        # print('**** local class loss: {:.4f}  local class acc: {:.4f}****'.format(class_loss,class_acc))

                        local_ws.append(copy.deepcopy(self.model.state_dict()))

                    elif 'fedrecovery' in self.ul_mode:
                        start_ul = time.time()
                        self.model.load_state_dict(global_state_dict)

                        local_w, local_loss= self.trainer._local_update(clients_train_data[idx], self.local_ep, self.lr, self.optim,self.ul_mode )
                        local_ws.append(copy.deepcopy(local_w))
                        local_models_per_epoch.append(copy.deepcopy(local_w))
                        end_ul = time.time()
                        ul_time = end_ul - start_ul
                        clients_dict[f"client{idx}"]+=ul_time
                    elif 'amnesiac' in self.ul_mode:
                        start_ul = time.time()
                        self.model.load_state_dict(global_state_dict)
                        # 根据敏感batch 计算对应update之和
                        # print('amnesiac learning')
                        local_w, local_loss, local_update_epoch = self.trainer._local_update(ul_clients_train_data[idx],
                                                                                             self.local_ep, self.lr,
                                                                                             self.optim, self.ul_mode)
                        local_ws.append(copy.deepcopy(local_w))
                        local_losses.append(local_loss)

                        for key in local_update_epoch:
                            global_update_epoch[key] += local_update_epoch[key] * 1 / self.num_users
                        update_list.append(global_update_epoch)
                        end_ul = time.time()
                        ul_time = end_ul - start_ul
                        clients_dict[f"client{idx}"]+=ul_time

        client_weights = [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]

        self.fed_avg(local_ws, client_weights, 1)

        dataset_name = self.dataset_name
        ul_mode = self.ul_mode

        if not os.path.exists(f'./saved_models/data_{dataset_name}/global_{ul_mode}'):
            os.makedirs(f'./saved_models/data_{dataset_name}/global_{ul_mode}')
        with open(f'./saved_models/data_{dataset_name}/global_{ul_mode}/clients_dict.txt', 'w') as f:
            for key, value in clients_dict.items():
                # 自定义格式，比如 "key: value\n"
                f.write(f"{key}: {value}\n")


        cpu, gpu = monitor.end()
        end_total = time.time()
        total_time = end_total -  start_total
        print("train time:",total_time)
        
        if not os.path.exists(f'./saved_models/data_{dataset_name}/global_{ul_mode}'):
                os.makedirs(f'./saved_models/data_{dataset_name}/global_{ul_mode}')
        with open(f'./saved_models/data_{dataset_name}/global_{ul_mode}/data.txt', 'w') as f:
            # f.write(f"unlearning time: {unlearning_time:.2f} s   ")
            f.write(f"train time: {total_time:.2f} s   ")
            f.write(f"Average CPU usage: {cpu}  ")
            f.write(f"Average GPU usage: {gpu}%")

        self.model.load_state_dict(self.w_t)
        if 'fedrecovery' in self.ul_mode:
            old_global_model_list.append(copy.deepcopy(self.model.state_dict()))
            old_local_model_list.append(local_models_per_epoch)

            for std in [0.020, 0.022, 0.025]:  # ,0.028,0.030,0.032,0.034,0.036,0.038,0.040]:
                recovery_state_dict = fedrecovery_operation(old_local_model_list, old_global_model_list, self.ul_clients, std)
                self.model.load_state_dict(recovery_state_dict)

                # ers_end = time.time()
                # ers_interval_time = ers_end - ers_start
                # ers_total_time+=ers_interval_time
                recovery_trainer = TrainerPrivate(self.model, self.device, self.dp, self.sigma, self.total_cls,
                                                  'none')
                test_x, test_y = zip(*test)
                test_loader = DataLoader(BatchflowData(test_x, test_y),
                                         batch_size=256, shuffle=True, drop_last=True)

                precision, recall, f1, accuracy = recovery_trainer.test_new(test_loader)
                #print('ACC: ', acc_val_mean)
                #loss_eraser_train_mean, acc_eraser_train_mean = recovery_trainer.test(train_ldr)


        if 'fedau' in self.ul_mode:
            if not os.path.exists(f'./saved_models/{dataset_name}/fedau'):
                os.makedirs(f'./saved_models/{dataset_name}/fedau')
            torch.save(self.model.state_dict(), f'./saved_models/{dataset_name}/fedau/model_global_forget.pth')
            test_x, test_y = zip(*test)
            test_loader = DataLoader(BatchflowData(test_x, test_y),
                                 batch_size=256, shuffle=True, drop_last=True)

            loss_val_mean, acc_val_mean = self.trainer.test(test_loader)
            print('ACC: ',acc_val_mean)

        if 'amnesiac' in self.ul_mode:
            if not os.path.exists(f'./saved_models/{dataset_name}/amnesiac'):
                os.makedirs(f'./saved_models/{dataset_name}/amnesiac')
            torch.save(self.model.state_dict(), f'./saved_models/{dataset_name}/amnesiac/model_global_forget.pth')
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

    def test_fedau(self):
        test = self.dataset.getglobalclass({0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, {2, 3, 4, 5, 6, 7, 8, 9})
        state_dict = torch.load('./saved_models/mirage/fedau/model_global_forget.pth')
        self.model.load_state_dict(state_dict)
        test_x, test_y = zip(*test)
        test_loader = DataLoader(BatchflowData(test_x, test_y),
                                 batch_size=256, shuffle=True, drop_last=True)

        precision, recall, f1, accuracy = self.trainer.test_new(test_loader)

    def test_amnesiac(self):
        test = self.dataset.getglobalclass({0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, {2, 3, 4, 5, 6, 7, 8, 9})
        state_dict = torch.load('./saved_models/amnesiac_ul/mirage_model_global_forget.pth')
        self.model.load_state_dict(state_dict)
        test_x, test_y = zip(*test)
        test_loader = DataLoader(BatchflowData(test_x, test_y),
                                 batch_size=256, shuffle=True, drop_last=True)

        precision, recall, f1, accuracy = self.trainer.test_new(test_loader)



if __name__ == '__main__':

    # mode = fedau, fedrecovery, amnesiac
    fd = fedul(150,10,[10,0],'fedau')
    # fd.test_amnesiac()

    fd.fedtrain(256,0.01)