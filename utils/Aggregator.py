import os
import torch
import copy

class FedAggregator:
    def __init__(self, model_path='../saved_models/', forget_clients=[5], feature_key='features', fc_key='fc'):
        self.model_path = model_path  # 模型存放路径
        self.forget_clients = forget_clients  # 遗忘客户端列表
        self.feature_key = feature_key  # feature 层关键字
        self.fc_key = fc_key  # fc 层关键字

    def load_model(self, client_idx):
        """加载客户端的模型 state_dict"""
        model_file = os.path.join(self.model_path, f'client{client_idx}', f'model+{client_idx}.pth')
        return torch.load(model_file)

    def aggregate_models(self):
        """遍历所有客户端并聚合模型参数"""
        local_ws = []  # 所有客户端的 state_dict
        non_forget_ws = []  # 非遗忘客户端的 state_dict
        forget_ws = []  # 遗忘客户端的 state_dict

        # 遍历所有客户端，分类为遗忘和非遗忘客户端
        for i in range(10):  # 遍历 10 个客户端
            state_dict = self.load_model(i)  # 加载客户端模型
            local_ws.append(state_dict)  # 所有客户端的模型

            if i in self.forget_clients:
                forget_ws.append(state_dict)  # 遗忘客户端模型
            else:
                non_forget_ws.append(state_dict)  # 非遗忘客户端模型

        # 聚合所有客户端的 feature 层
        w_avg_features = self.aggregate_features(local_ws)

        # 分别聚合非遗忘客户端和遗忘客户端的 fc 层
        w_fc_non_forget = self.aggregate_fc(non_forget_ws)
        w_fc_forget = self.aggregate_fc(forget_ws)

        # 组合 feature 和 fc 层，生成 w_avg1 和 w_avg2
        w_avg1 = self.combine_feature_fc(w_avg_features, w_fc_non_forget)
        w_avg2 = self.combine_feature_fc(w_avg_features, w_fc_forget)

        return w_avg1, w_avg2

    def aggregate_features(self, clients_ws):
        """聚合所有客户端的 feature 层参数"""
        if not clients_ws:  # 如果客户端列表为空
            return None

        # 使用第一个客户端的 state_dict 作为模板
        w_avg_features = copy.deepcopy(clients_ws[0])

        # 遍历所有参数，初始化并累加 feature 层
        for k in w_avg_features.keys():
            if self.feature_key in k:  # 仅处理 feature 层
                w_avg_features[k] = torch.zeros_like(w_avg_features[k], dtype=torch.float32)

                # 累加所有客户端的 feature 参数
                for ws in clients_ws:
                    w_avg_features[k] += ws[k].to(torch.float32)

                # 取平均
                w_avg_features[k] /= len(clients_ws)

        return w_avg_features

    def aggregate_fc(self, clients_ws):
        """聚合客户端的 fc 层参数"""
        if not clients_ws:  # 如果客户端列表为空
            return None

        # 使用第一个客户端的 state_dict 作为模板
        w_avg_fc = copy.deepcopy(clients_ws[0])

        # 遍历所有参数，初始化并累加 fc 层
        for k in w_avg_fc.keys():
            if self.fc_key in k:  # 仅处理 fc 层
                w_avg_fc[k] = torch.zeros_like(w_avg_fc[k], dtype=torch.float32)

                # 累加所有客户端的 fc 参数
                for ws in clients_ws:
                    w_avg_fc[k] += ws[k].to(torch.float32)

                # 取平均
                w_avg_fc[k] /= len(clients_ws)

        return w_avg_fc

    def combine_feature_fc(self, w_features, w_fc):
        """将 feature 层和 fc 层组合成一个模型"""
        combined_model = copy.deepcopy(w_features)

        for k in w_fc.keys():
            if self.fc_key in k:  # 仅替换 fc 层的参数
                combined_model[k] = w_fc[k]

        return combined_model
if __name__ == '__main__':

# 示例使用
    aggregator = FedAggregator(forget_clients=[5])  # 指定客户端 5 为遗忘客户端
    w_avg1, w_avg2 = aggregator.aggregate_models()

    print("w_avg1（非遗忘客户端 FC 层 + 所有客户端 Feature 层）聚合完成")
    print("w_avg2（遗忘客户端 FC 层 + 所有客户端 Feature 层）聚合完成")
