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
from utils.utils import CpuGpuMonitor
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

class fedunsilapps(basetrain):
    def __init__(self, epochs, num_class, incremental_num_list,data):
        super().__init__(incremental_num_list)
        self.epochs = epochs
        # self.total_cls = num_class
        self.idxs_users = 10
        self.ul_clients = [2]
        self.dataset_name = data
        self.dataset = flowfeatures(self.ul_clients,data=data)
        self.learn = learn(num_class, [num_class, 0],data)
        self.unlearn = unlearn(num_class, [num_class, 0],data)
        self.model = ResNet(classes=num_class).cuda()
        self.model_ul = ResNet(classes=num_class).cuda()
        self.w_t = copy.deepcopy(self.model.state_dict())
        self.first_num = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
        self.second_num = {2, 3, 4, 5, 6, 7, 8, 9}
        self.dele_class = [0,1]
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

    def remove_categories_from_clients(self, clients_data, client_ids, categories):
        """
        从指定客户端的数据中删除指定类别的数据。

        参数:
        - clients_data: 客户端数据字典（格式：{客户端ID: [(数据, 标签), ...] }）
        - client_ids: 要修改的客户端ID列表
        - category: 要删除的类别

        返回:
        - 修改后的客户端数据字典
        """
        for client_id in client_ids:
            if client_id in clients_data:
                # 过滤掉指定类别的数据
                clients_data[client_id] = [
                    (data, label) for data, label in clients_data[client_id] if label not in categories
                ]

                print(f"已从客户端 {client_id} 中删除类别 {categories} 的数据。")
            else:
                print(f"客户端 {client_id} 不存在于数据中。")

        return clients_data

    def fedtrain(self, batch_size, lr):
        # 获取数据

        # 这里用于存放遗忘学习的参数
        #ul_state_dicts = []
        # for epoch in range(self.epochs):
        #local_models_per_epoch = []
        #global_state_dict = copy.deepcopy(self.model.state_dict())
        dataset_name = self.dataset_name
        local_ws = defaultdict(list)
        clients_train_data, clients_val_data, clients_test_data, multi_dict = self.get_data('all')
        
        monitor = CpuGpuMonitor()
        monitor_total = CpuGpuMonitor()
        monitor_total.start()
        start_total = time.time()
        for idx in range(self.idxs_users):
            print("---" * 15,f"client:{idx}","---" * 15)
            if (idx in self.ul_clients) == False:
                # learn
                
                # monitor = CpuGpuMonitor()
                #        start = time.time()
                monitor.start()
                start = time.time()
                local_w = self.learn.train(batch_size, self.epochs, lr, clients_train_data[idx], clients_test_data[idx],
                                           idx, multi_dict)
                end = time.time()
                local_ws[idx].extend(copy.deepcopy(local_w))
                cpu, gpu = monitor.end()
                training_time = end - start
                print(f"second model time: {training_time:.2f} s")
                if not os.path.exists(f'./saved_models/{dataset_name}/client{idx}/'):
                    os.makedirs(f'./saved_models/{dataset_name}/client{idx}/')
                with open(f'./saved_models/{dataset_name}/client{idx}/data.txt', 'w') as f:
                    f.write(f"train time:{training_time:.2f} s   ")
                    f.write(f"Average CPU usage: {cpu}   ")
                    f.write(f"Average GPU usage: {gpu}%")

            else:
                # unlearn
                # self.model_ul.load_state_dict(ul_state_dicts[idx])
                # ul_model除W2外替换为global model的参数
                # self.model_ul.load_state_dict(global_state_dict, strict=False)
                # monitor = CpuGpuMonitor()
                monitor.start()
                start = time.time()
                local_w = self.learn.train(batch_size, self.epochs, lr, clients_train_data[idx], clients_test_data[idx],
                                           idx, multi_dict)
                end = time.time()
                training_time = end - start
                clients_ul_train_data = self.remove_categories_from_clients(clients_train_data,self.ul_clients,self.dele_class)
                clients_ul_test_data = self.remove_categories_from_clients(clients_test_data,self.ul_clients,self.dele_class)
                start_ul = time.time()
                local_w_ul = self.unlearn.train(batch_size, self.epochs, lr, clients_ul_train_data[idx], clients_ul_test_data[idx],
                                                idx, multi_dict)
                end_ul = time.time()
                unlearning_time = end_ul - start_ul
                cpu, gpu = monitor.end()
                local_ws[idx].extend(copy.deepcopy(local_w_ul))
                if not os.path.exists(f'./saved_models/{dataset_name}/client{idx}/'):
                    os.makedirs(f'./saved_models/{dataset_name}/client{idx}/')
                with open(f'./saved_models/{dataset_name}/client{idx}/data.txt', 'w') as f:
                    f.write(f"unlearning time: {unlearning_time:.2f} s   ")
                    f.write(f"train time: {training_time:.2f} s   ")
                    f.write(f"Average CPU usage: {cpu}  ")
                    f.write(f"Average GPU usage: {gpu}%")

        #client_weights = []
        #for i in range(self.idxs_users):
        #    client_weights.append(1 / self.idxs_users)

        #print(local_ws)
        #w_avg1, w_avg2 = self.fed_avg_features_fc(local_ws, client_weights, 1, self.ul_clients)
        fedavg_start_time = time.time()
        aggregator = FedAggregator(model_path=f'./saved_models/{dataset_name}/', forget_clients=self.ul_clients, feature_key='features', fc_key='fc')
        non_forget_model, forget_model = aggregator.aggregate_models()
        fedavg_end_time = time.time()
        fedavg_time = fedavg_end_time - fedavg_start_time

        end_total = time.time()
        total_time = end_total - start_total

        cpu_total, gpu_total = monitor_total.end()
       
        if not os.path.exists(f'./saved_models/{dataset_name}/globalModel/'):
            os.makedirs(f'./saved_models/{dataset_name}/globalModel/')
        torch.save(non_forget_model, f'./saved_models/{dataset_name}/globalModel/non_forget_model.pth')
        torch.save(forget_model, f'./saved_models/{dataset_name}/globalModel/forget_model.pth')
        # 测试 w_avg2 这里的测试需要将遗忘的类别去掉
        with open(f'./saved_models/{dataset_name}/globalModel/data.txt', 'w') as f:
                f.write(f"fedavg time: {fedavg_time:.2f}s   ")
                f.write(f"total time: {total_time:.2f}s   ")
                f.write(f"Average CPU usage: {cpu_total}  ")
                f.write(f"Average GPU usage: {gpu_total}%")
        testdata = self.dataset.getglobalclass(self.first_num,self.second_num) 
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

    def unit_test_globalmodel(self):
        dataset_name = self.dataset_name
        state_dict = torch.load(f'./saved_models/{dataset_name}/globalModel/forget_model.pth')
        self.model.load_state_dict(state_dict)

        testdata = self.dataset.getglobalclass(self.first_num,self.second_num) 
        test_x, test_y = zip(*testdata)
        new_labels, label_mapping = self.automate_label_mapping(test_y)
        testdata = DataLoader(BatchflowData(test_x, new_labels),
                               batch_size=256, shuffle=True, drop_last=True)

        correct = 0
        wrong = 0
        all_preds = []  # 存储所有预测结果
        all_targets = []  # 存储所有真实标签

        for i, (x, label) in enumerate(testdata):
            x = x.type(torch.FloatTensor)
            x = x.cuda()
            label = label.view(-1).cuda()
            p = self.model(x)
            pred = p[:, :len(self.second_num)].argmax(dim=-1)
            # pred_leverage = self.inverse_label_mapping(pred, label_mapping)
            correct += sum(pred == label).item()
            wrong += sum(pred != label).item()

            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(label.cpu().numpy())

        precision = precision_score(all_targets, all_preds, average='macro')
        recall = recall_score(all_targets, all_preds, average='macro')
        f1 = f1_score(all_targets, all_preds, average='macro')
        accuracy = accuracy_score(all_targets, all_preds)

        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, Accuracy: {accuracy:.4f}")
        acc = correct / (wrong + correct)
        print("Test Ul Acc: {}".format(acc * 100))


# uint test
if __name__ == '__main__':
    fus = fedunsilapps(50, 10, [10,0],'mirage')
    # fus.fedtrain(256, 0.01)
    fus.unit_test_globalmodel()