"""
@Time    : 2023/9/4 17:16
-------------------------------------------------
@Author  : sailorlee(lizeyi)
@email   : sailorlee31@gmail.com
-------------------------------------------------
@FileName: FlowFeatures.py
@Software: PyCharm
"""
# coding:utf-8
import pickle
import logging
from itertools import cycle
from imblearn.over_sampling import SMOTE
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from collections import defaultdict
from sklearn.utils import shuffle
import random
from collections import defaultdict


class DataProcessor:
    def __init__(self, data, labels, remain_clients=None, ul_clients=None):
        """
        初始化训练、验证或测试数据集。
        参数:
        - data: 数据样本列表
        - labels: 数据样本对应的标签列表
        """
        if remain_clients is None:
            remain_clients = [0, 1, 2, 3, 4, 6, 7, 8, 9]
        if ul_clients is None:
            ul_clients = [5]
        self.data = data
        self.labels = labels
        self.remain_client = remain_clients
        self.ul_clients = ul_clients

    def split_data_for_clients(self, client_ids, samples_per_class=1000):
        """
        将数据按类别划分并为每个客户端分配指定数量的数据。
        参数:
        - num_clients: 客户端数量
        - samples_per_class: 每个客户端需要每个类别的数据量

        返回:
        - clients_data: 每个客户端的数据（字典格式 {客户端ID: [(数据, 标签), ...] }）
        """
        # 将数据按类别存储
        num_clients = len(client_ids)

        class_data = defaultdict(list)
        for item, label in zip(self.data, self.labels):
            class_data[label].append((item, label))
        # 检查每个类别是否有足够的数据
        for label, items in class_data.items():
            # print(f"类别 {label}:",len(items))
            if len(items) < num_clients * samples_per_class:
                raise ValueError(f"类别 {label} 的数据不足，至少需要 {num_clients * samples_per_class} 个样本")
        # 为每个客户端分配数据
        clients_data = defaultdict(list)
        for label, items in class_data.items():
            # 随机打乱该类别的数据
            random.shuffle(items)
            # 均匀分配给每个客户端
            for idx, client_id in enumerate(client_ids):
                start_idx = idx * samples_per_class
                end_idx = start_idx + samples_per_class
                clients_data[client_id].extend(items[start_idx:end_idx])
        return clients_data


# 示例调用
# 模拟数据集：8个类别，每类10000个样本，共计80000个样本
# data = [f"data_{i}" for i in range(80000)]  # 伪数据样本
# labels = [i // 10000 for i in range(80000)]  # 每10000个数据为一类

# 初始化数据处理器并分配数据
# processor = DataProcessor(data, labels)
# clients_data = processor.split_data_for_clients(num_clients=10, samples_per_class=1000)

# 打印每个客户端的数据量
# for client_id, client_data in clients_data.items():
#     print(f"客户端 {client_id} 的数据量：{len(client_data)}")

# 验证每个客户端的数据是否均匀分布在所有类别中
# for client_id, client_data in clients_data.items():
#     label_counts = defaultdict(int)
#     for _, label in client_data:
#         label_counts[label] += 1
#     print(f"客户端 {client_id} 的类别分布：{dict(label_counts)}")

class flowfeatures():

    def __init__(self, ul_clients_id, data, clients_id=None):
        if clients_id is None:
            clients_id = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        if data == 'njupt':
            self.train_data, self.test_data, self.val_data, self.train_labels, self.test_labels, self.val_labels = self.processdata()
        elif data == 'mirage':
            self.train_data, self.test_data, self.val_data, self.train_labels, self.test_labels, self.val_labels = self.processMIRAGEdata()
        elif data == 'iscx':
            self.train_data, self.test_data, self.val_data, self.train_labels, self.test_labels, self.val_labels = self.processISCXdata()
        elif data == 'cic':
            self.train_data, self.test_data, self.val_data, self.train_labels, self.test_labels, self.val_labels = self.processCICdata()
        else:
            raise RuntimeError('no DATA!!!')
        # self.train_groups, self.val_groups, self.test_groups, self.multi_dict = self.evolveinitialize()
        self.multi_dict = self.evolveinitialize()
        # 持续的次数
        self.batch_num = 2
        self.clients_id = clients_id
        self.ul_clients_id = ul_clients_id
        self.total_clients = 10

    def _process_index_label(self, labels):

        le = LabelEncoder()
        labels_en = le.fit_transform(labels).astype(np.int64)
        res = {}
        for cl in le.classes_:
            res.update({cl: le.transform([cl])[0]})
        print(res)

    def processdata(self):

        df = pd.read_csv('./data/newdataframe19.csv')
        scaler = StandardScaler()
        numeric_columns = df.columns[:-1]
        df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
        # dataframe0 = df.drop(df.columns[0], axis=1)
        # train, test = train_test_split(df, test_size=0.1)
        train = df.groupby('appname').head(10000)
        val = df.groupby('appname').head(1000)
        test = df.groupby('appname').tail(1000)
        # train, val = train_test_split(train, test_size=0.1)

        val_labels = val.pop('appname')
        train_labels = train.pop('appname')
        # print("The train is :{}".format(list(set(train_labels))))
        test_labels = test.pop('appname')
        # print("The test is :{}".format(list(set(test_labels))))
        return train.values, test.values, val.values, train_labels, test_labels, val_labels

    def smote_data(self,df,num):

        df_features = df.iloc[:, 1:-1]  # 原始数据的特征（去掉第一列和最后一列）
        df_labels = df.iloc[:, -1]  # 原始数据的标签

        # 合并特征和标签，得到处理后的原始数据集
        df_processed = pd.concat([df_features, df_labels], axis=1)

        # 初始化 SMOTE 对象，指定每个类别增强到 1000 条
        smote = SMOTE(sampling_strategy={label: num for label in df_labels.unique()}, random_state=42)

        # 使用 SMOTE 进行增强
        X_resampled, y_resampled = smote.fit_resample(df_features, df_labels)

        # 将增强后的特征和标签合并为新的 DataFrame
        df_resampled = pd.DataFrame(X_resampled, columns=df.columns[1:-1])
        df_resampled['appname'] = y_resampled

        # 将处理后的原始数据和增强后的数据合并
        df_combined = pd.concat([df_processed, df_resampled], ignore_index=False)

        # 打乱数据集
        df_combined = shuffle(df_combined, random_state=42).reset_index(drop=True)

        return df_combined
    
    def processCICdata(self):
        df = pd.read_csv('./data/CIC_dataset_2022.csv')
        keep = [
        "Camera_simcam",
        "Camera_homeeyecam",
        "Camera_arloqcam",
        "Camera_arlobasecam",
        "Camera_luohecam",
        "Camera_amcrest",
        "Camera_dlinkcam",
        "Camera_heimvisioncam",
        "Home Automation_eufyhomebase",
        "Camera_netatmocam"
        ]
        enc=LabelEncoder() 
        df_filtered = df[df["source"].isin(keep)].reset_index(drop=True)
        data = enc.fit_transform(df_filtered["source"])
        df_filtered["source"] = data
        scaler = StandardScaler()
        numeric_columns = df_filtered.columns[:-1]
        df_filtered[numeric_columns] = scaler.fit_transform(df_filtered[numeric_columns])
        
        train = df_filtered.groupby('source').head(10000)
        val = df_filtered.groupby('source').head(1000)
        test = df_filtered.groupby('source').tail(1000)
        # train, val = train_test_split(train, test_size=0.1)

        val_labels = val.pop('source')
        train_labels = train.pop('source')
        # print("The train is :{}".format(list(set(train_labels))))
        test_labels = test.pop('source')
        # print("The test is :{}".format(list(set(test_labels))))
        return train.values, test.values, val.values, train_labels, test_labels, val_labels



    def processMIRAGEdata(self):

        df = pd.read_csv('./data/MIRAGE.csv')
        df_combined = self.smote_data(df,10000)
        #label_counts = df_combined.groupby('appname').size()
        #print(label_counts)
        scaler = StandardScaler()
        numeric_columns = df_combined.columns[:-1]
        df_combined[numeric_columns] = scaler.fit_transform(df_combined[numeric_columns])
        # dataframe0 = df.drop(df.columns[0], axis=1)
        # train, test = train_test_split(df, test_size=0.1)
        train = df_combined.groupby('appname').head(10000)
        val = df_combined.groupby('appname').head(1000)
        test = df_combined.groupby('appname').tail(1000)
        # train, val = train_test_split(train, test_size=0.1)

        val_labels = val.pop('appname')
        train_labels = train.pop('appname')
        # print("The train is :{}".format(list(set(train_labels))))
        test_labels = test.pop('appname')
        # print("The test is :{}".format(list(set(test_labels))))
        return train.values, test.values, val.values, train_labels, test_labels, val_labels

    def processISCXdata(self):

        df = pd.read_csv('../data/new_ISCX-VPN2016_smote.csv')
        label_counts = df.groupby('appname').size()
        print(label_counts)
        scaler = StandardScaler()
        numeric_columns = df.columns[:-1]
        df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
        # dataframe0 = df.drop(df.columns[0], axis=1)
        # train, test = train_test_split(df, test_size=0.1)
        train = df.groupby('appname').head(10000)
        val = df.groupby('appname').head(1000)
        test = df.groupby('appname').tail(1000)
        # train, val = train_test_split(train, test_size=0.1)

        val_labels = val.pop('appname')
        train_labels = train.pop('appname')
        # print("The train is :{}".format(list(set(train_labels))))
        test_labels = test.pop('appname')
        # print("The test is :{}".format(list(set(test_labels))))
        return train.values, test.values, val.values, train_labels, test_labels, val_labels

    def get_status(self, set1, set2):
        '''

        :param set1: 上一次的应用列表  app list in pre set
        :param set2:  app list in set
        :return:
        '''
        # 查看有没有增加的元素
        incremental_elements = set2 - set1
        # 查看有没有减少的元素
        Reduce_elements = set1 - set2
        if len(Reduce_elements) > 0 and len(incremental_elements) == 0:
            status = 0
            incremental_elements = None

            # print('fine-tune')

        elif len(Reduce_elements) > 0 and len(incremental_elements) > 0:
            status = 1
            # print('先 fine-tune')
            # print('再 bias- incremental')

        elif len(incremental_elements) > 0 and len(Reduce_elements) == 0:
            status = 2
            # print('bias incremental')
        else:
            status = 3
            # print('no change!')

        return status, incremental_elements, Reduce_elements

    def evolveinitialize(self, first_num=None, second_num=None):
        if first_num is None:
            first_num = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
        if second_num is None:
            second_num = {2, 3, 4, 5, 6, 7, 8, 9}

        # a = [second_num,third_num,forth_num]
        a = [second_num]
        """
        先判断状态
        """
        multi_dict = defaultdict(list)
        lista = [first_num, second_num]
        for i in range(len(lista) - 1):
            status, incremental_elements, Reduce_elements = self.get_status(lista[i], lista[i + 1])
            multi_dict[i + 1].extend([status, incremental_elements, Reduce_elements, len(a[i])])
        return multi_dict

        # lista = [first_num, second_num]
        # for i in range(len(lista) - 1):
        #     status, incremental_elements, Reduce_elements = self.get_status(lista[i], lista[i + 1])
        #     multi_dict[i + 1].extend([status, incremental_elements, Reduce_elements, len(a[i])])
        #     # statuslist.append(status)
        #     # incremental_sets.append(incremental_elements)
        #     # Reduce_sets.append(Reduce_elements)
        #
        # train_groups = [[] for _ in range(len(lista))]
        # for train_data, train_label in zip(self.train_data, self.train_labels):
        #     # 第一次需要训练的元素放入
        #     if train_label in list(first_num):
        #         train_groups[0].append((train_data, train_label))
        #     for key, values in multi_dict.items():
        #         # 依次把标签为4 为5 6 为 7 8 9的放进label中
        #         if values[1] and train_label in list(values[1]):
        #             train_groups[key].append((train_data, train_label))
        #
        # val_groups = [[] for _ in range(len(lista))]
        # for val_data, val_label in zip(self.val_data, self.val_labels):
        #     if val_label in list(first_num):
        #         val_groups[0].append((val_data, val_label))
        #     for key, values in multi_dict.items():
        #         # 依次把标签为4 为5 6 为 7 8 9的放进label中
        #         if values[1] and val_label in list(values[1]):
        #             val_groups[key].append((val_data, val_label))
        #
        # test_groups = [[] for _ in range(len(lista))]
        # for test_data, test_label in zip(self.test_data, self.test_labels):
        #     if test_label in list(first_num):
        #         test_groups[0].append((test_data, test_label))
        #     for key, values in multi_dict.items():
        #         # 依次把标签为4 为5 6 为 7 8 9的放进label中
        #         if values[1] and test_label in list(values[1]):
        #             test_groups[key].append((test_data, test_label))

        #return train_groups, val_groups, test_groups, multi_dict
    def getNextClasses(self, i):
        return self.train_groups[i], self.val_groups[i], self.test_groups[i]

    def splitclientdata(self, list, data, labels, samples_per_class, clients_id):
        train = []
        for train_data, train_label in zip(data, labels):
            # 第一次需要训练的元素放入
            if train_label in list:
                train.append((train_data, train_label))
        train_data, train_label = zip(*train)
        processor = DataProcessor(train_data, train_label)
        clients_traindata = processor.split_data_for_clients(client_ids=clients_id,
                                                             samples_per_class=samples_per_class)
        return clients_traindata

    def getul(self,
              first_num=None,
              second_num=None):
        if first_num is None:
            first_num = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
        if second_num is None:
            second_num = {2, 3, 4, 5, 6, 7, 8, 9}

        lista = [first_num, second_num]

        status, incremental_elements, Reduce_elements = self.get_status(lista[0], lista[1])

        ul_clients_train_data = self.splitclientdata(Reduce_elements, self.train_data, self.train_labels,
                                                     int((10000*len(Reduce_elements) / len(Reduce_elements)) / len(self.ul_clients_id)),
                                                     self.ul_clients_id)
        ul_clients_test_data = self.splitclientdata(Reduce_elements, self.test_data, self.test_labels,
                                                    int((1000*len(Reduce_elements) / len(Reduce_elements)) / len(self.ul_clients_id)),
                                                    self.ul_clients_id)
        ul_clients_val_data = self.splitclientdata(Reduce_elements, self.val_data, self.val_labels,
                                                   int((1000*len(Reduce_elements) / len(Reduce_elements)) / len(self.ul_clients_id)),
                                                   self.ul_clients_id)

        return ul_clients_train_data, ul_clients_test_data, ul_clients_val_data

    def getglobalclass(self, first_num, second_num):
        """
        get remain classes, all data
        :param first_num:
        :param second_num:
        :return:
        """
        if first_num is None:
            first_num = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
        if second_num is None:
            second_num = {2, 3, 4, 5, 6, 7, 8, 9}
        a = [second_num]
        multi_dict = defaultdict(list)
        lista = [first_num, second_num]
        for i in range(len(lista) - 1):
            status, incremental_elements, Reduce_elements = self.get_status(lista[i], lista[i + 1])
            multi_dict[i + 1].extend([status, incremental_elements, Reduce_elements, len(a[i])])
        test = []
        for test_data, test_label in zip(self.test_data, self.test_labels):
            if test_label in second_num:
                test.append((test_data, test_label))
        return test

    def getglobalallclass(self, first_num, second_num):
        if first_num is None:
            first_num = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
        if second_num is None:
            second_num = {2, 3, 4, 5, 6, 7, 8, 9}
        a = [second_num]
        multi_dict = defaultdict(list)
        lista = [first_num, second_num]
        for i in range(len(lista) - 1):
            status, incremental_elements, Reduce_elements = self.get_status(lista[i], lista[i + 1])
            multi_dict[i + 1].extend([status, incremental_elements, Reduce_elements, len(a[i])])
        test = []
        for test_data, test_label in zip(self.test_data, self.test_labels):
            if test_label in first_num:
                test.append((test_data, test_label))
        return test

    def getglobalulclass(self, first_num, second_num):
        if first_num is None:
            first_num = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
        if second_num is None:
            second_num = {2, 3, 4, 5, 6, 7, 8, 9}
        a = [second_num]
        multi_dict = defaultdict(list)
        lista = [first_num, second_num]
        for i in range(len(lista) - 1):
            status, incremental_elements, Reduce_elements = self.get_status(lista[i], lista[i + 1])
            multi_dict[i + 1].extend([status, incremental_elements, Reduce_elements, len(a[i])])
        test = []
        for test_data, test_label in zip(self.test_data, self.test_labels):
            if test_label in multi_dict[1][2]:
                test.append((test_data, test_label))
        return test

    def getallclass(self, first_num=None, second_num=None):
        if first_num is None:
            first_num = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
        if second_num is None:
            second_num = {2, 3, 4, 5, 6, 7, 8, 9}
        clients_train_data = self.splitclientdata(first_num, self.train_data, self.train_labels,
                                                  int((100000 / len(first_num)) / self.total_clients),
                                                  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        clients_val_data = self.splitclientdata(second_num, self.val_data, self.val_labels,
                                                int((10000 / len(first_num)) / (self.total_clients)),
                                                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        clients_test_data = self.splitclientdata(second_num, self.test_data, self.test_labels,
                                                 int((10000 / len(first_num)) / (self.total_clients)),
                                                 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        return clients_train_data, clients_val_data, clients_test_data

    def getUlClass(self,
                   first_num=None,
                   second_num=None):

        if first_num is None:
            first_num = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
        if second_num is None:
            second_num = {2, 3, 4, 5, 6, 7, 8, 9}
        a = [second_num]
        multi_dict = defaultdict(list)
        lista = [first_num, second_num]
        for i in range(len(lista) - 1):
            status, incremental_elements, Reduce_elements = self.get_status(lista[i], lista[i + 1])
            multi_dict[i + 1].extend([status, incremental_elements, Reduce_elements, len(a[i])])
        # print(multi_dict)
        # train_ul = [[] for _ in range(len(lista))]
        train_ul = []
        for train_data, train_label in zip(self.train_data, self.train_labels):
            # 第一次需要训练的元素放入
            if train_label in multi_dict[1][2]:
                train_ul.append((train_data, train_label))
        val_ul = []
        for val_data, val_label in zip(self.val_data, self.val_labels):
            if val_label in multi_dict[1][2]:
                val_ul.append((val_data, val_label))
        test_ul = []
        for test_data, test_label in zip(self.test_data, self.test_labels):
            if test_label in multi_dict[1][2]:
                test_ul.append((test_data, test_label))
        return train_ul, val_ul, test_ul

    def getRemainClass(self,
                       first_num=None,
                       second_num=None):

        if first_num is None:
            first_num = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
        if second_num is None:
            second_num = {2, 3, 4, 5, 6, 7, 8, 9}
        a = [second_num]
        multi_dict = defaultdict(list)
        lista = [first_num, second_num]
        for i in range(len(lista) - 1):
            status, incremental_elements, Reduce_elements = self.get_status(lista[i], lista[i + 1])
            multi_dict[i + 1].extend([status, incremental_elements, Reduce_elements, len(a[i])])
        # print(multi_dict)
        # train_ul = [[] for _ in range(len(lista))]

        clients_train_data = self.splitclientdata(second_num, self.train_data, self.train_labels,
                                                  int((10000*len(second_num) / len(second_num)) / (
                                                              self.total_clients - len(self.ul_clients_id))),
                                                  self.clients_id)
        clients_val_data = self.splitclientdata(second_num, self.val_data, self.val_labels,
                                                int((1000*len(second_num) / len(second_num)) / (
                                                            self.total_clients - len(self.ul_clients_id))),
                                                self.clients_id)
        clients_test_data = self.splitclientdata(second_num, self.test_data, self.test_labels,
                                                 int((1000*len(second_num) / len(second_num)) / (
                                                             self.total_clients - len(self.ul_clients_id))),
                                                 self.clients_id)

        # train = []
        # for train_data, train_label in zip(self.train_data, self.train_labels):
        #     # 第一次需要训练的元素放入
        #     if train_label in second_num:
        #         train.append((train_data, train_label))
        # train_data, train_label = zip(*train)
        # processor = DataProcessor(train_data, train_label)
        # clients_traindata = processor.split_data_for_clients(num_clients=9, samples_per_class=int((80000/8)/9))

        # for client_id, client_data in clients_data.items():
        #     print(f"客户端 {client_id} 的数据量：{len(client_data)}")
        # val = []
        # for val_data, val_label in zip(self.val_data, self.val_labels):
        #     if val_label in second_num:
        #         val.append((val_data, val_label))
        # val_data, val_label = zip(*val)
        # processor = DataProcessor(val_data, val_label)
        # clients_valdata = processor.split_data_for_clients(num_clients=9, samples_per_class=int((80000 / 8) / 9))
        #
        # test = []
        # for test_data, test_label in zip(self.test_data, self.test_labels):
        #     if test_label in second_num:
        #         test.append((test_data, test_label))
        # test_data, val_label = zip(*val)
        # processor = DataProcessor(val_data, val_label)
        # clients_testdata = processor.split_data_for_clients(num_clients=9, samples_per_class=int((80000 / 8) / 9))

        return clients_train_data, clients_val_data, clients_test_data


def process_csv():
    df = pd.read_csv('../data/dataframe15.csv')
    # class_counts = df['appname'].value_counts()
    del_list = ['QQ音乐', '爱奇艺', '百度贴吧', '金铲铲之战']
    for i in del_list:
        df = df[df['appname'] != i]

    # class_counts = df['appname'].value_counts()
    # print(class_counts)
    df.drop(df.columns[[0]], axis=1, inplace=True)

    # propress labels
    labels = df.pop('appname')
    le = preprocessing.LabelEncoder()
    numlabel = le.fit_transform(labels)
    df['appname'] = numlabel
    res = {}
    for cl in le.classes_:
        res.update({cl: le.transform([cl])[0]})
    print(res)
    WorldNet = open("./log/label.txt", "w", encoding="utf-8")
    WorldNet.write(str(res))
    WorldNet.close()
    result_df = df.groupby('appname').head(11000)

    result_df.to_csv('./data/newdataframe15.csv', index=False)


if __name__ == '__main__':
    ff = flowfeatures([2],'mirage')
    #ff.processMIRAGEdata()

    # ul_clients_train_data, ul_clients_test_data, ul_clients_val_data = ff.getallclass()
    # for client_id, client_data in ul_clients_train_data.items():
    #     print(f"客户端 {client_id} 的数据量：{len(client_data)}")
    # # test,label  = zip(*test_ul)
    # for client_id, client_data in ul_clients_train_data.items():
    #     label_counts = defaultdict(int)
    #     for _, label in client_data:
    #         label_counts[label] += 1
    #     print(f"客户端 {client_id} 的类别分布：{dict(label_counts)}")
    # process_csv()
    # df  = pd.read_csv('./data/dataframe24.csv')
    # class_counts = df['appname'].value_counts()
    # print(class_counts)
    # ff = flowfeatures()
    # train_groups, val_groups, test_groups, multi_dict = ff.evolveinitialize()
    # print(multi_dict)
    # train_groups, val_groups, test_groups = ff.newinitialize()
    # train_groups, val_groups, test_groups = ff.getNextClasses(2)
    # print(len(train_groups))
    # dataframe = df.copy()
    # labels = dataframe.pop('appname')
    # dataframe.drop(dataframe.columns[[0]], axis=1, inplace=True)
    # # propress labels
    # le = preprocessing.LabelEncoder()
    # numlabel = le.fit_transform(labels)
    # dataframe['appname'] = numlabel
    #
    # result_df = dataframe.groupby('appname').head(10000)
    # result_df.to_csv('./data/newdataframe40.csv',index=False)
    # print(numlabel)
    # ()
