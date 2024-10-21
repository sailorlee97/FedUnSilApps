import torch

from models.model import BiasLayer


class basetrain:
    def __init__(self, incremental_num_list):
        self.bias_layer1 = BiasLayer().cuda()
        self.bias_layer2 = BiasLayer().cuda()
        self.bias_layer3 = BiasLayer().cuda()
        self.bias_layer4 = BiasLayer().cuda()
        # self.bias_layer5 = BiasLayer().cuda()
        self.bias_layers = [self.bias_layer1, self.bias_layer2, self.bias_layer3, self.bias_layer4]
        self.sample = incremental_num_list
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def bias_forward_new(self, input, inc, status):
        # 0 1 2
        if inc > 0:
            if status[inc][2]:
                # 老应用个数
                in1 = input[:, :status[inc][3] - len(status[inc][2])]
                in2 = input[:, status[inc][3] - len(status[inc][2]):status[inc][3]]
                out1 = self.bias_layer1(in1)
                out2 = self.bias_layer2(in2)
                if in1.shape[1] + in2.shape[1] == input.shape[1]:
                    return torch.cat([out1, out2], dim=1), 2
                else:
                    in3 = input[:, status[inc][3]:]
                    out3 = self.bias_layer3(in3)
                    return torch.cat([out1, out2, out3], dim=1), 3
            else:
                in1 = input[:, :status[inc][3]]
                in2 = input[:, status[inc][3]:status[inc][3] + len(status[inc][1])]
                out1 = self.bias_layer1(in1)
                out2 = self.bias_layer2(in2)
                if in1.shape[1] + in2.shape[1] == input.shape[1]:
                    return torch.cat([out1, out2], dim=1), 2
                else:
                    in3 = input[:, status[inc][3] + len(status[inc][1]):]
                    out3 = self.bias_layer3(in3)
                    return torch.cat([out1, out2, out3], dim=1), 3
        else:
            in1 = input[:, :self.sample[0]]
            in2 = input[:, self.sample[0]:]
            out1 = self.bias_layer1(in1)
            out2 = self.bias_layer2(in2)
            return torch.cat([out1, out2], dim=1), 2

    def get_lr(self, optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    def automate_label_mapping(self, original_labels):
        # 训练前用
        unique_labels = sorted(set(original_labels))
        label_mapping = {label: index for index, label in enumerate(unique_labels)}
        new_labels = [label_mapping[label] for label in original_labels]
        return new_labels, label_mapping

    def inverse_label_mapping(self, new_labels, label_mapping):
        # 测试的时候用
        new_labels_cuda = torch.tensor(new_labels).cuda()
        new_labels_list = new_labels_cuda.cpu().tolist()

        inverse_mapping = {v: k for k, v in label_mapping.items()}
        original_labels = [inverse_mapping[label] for label in new_labels_list]
        return torch.tensor(original_labels).to('cuda')

    def stage(self,train_data, criterion, optimizer, inc_i, status):
        return NotImplementedError

    def stage_reduce(self,train_loader, criterion, optimizer, inc_i, status,
                                  label_mapping_fine_tune):
        return NotImplementedError

    def test_data(self, testdata, mappping, inc, status):
        return NotImplementedError