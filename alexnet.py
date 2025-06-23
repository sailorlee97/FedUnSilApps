import torch
import torch.nn as nn
from pyexpat import features

from models.layers.conv2d import ConvBlock
# from models.layers.passportconv2d_private import PassportPrivateBlock

class AlexNet(nn.Module):

    def __init__(self, num_classes,in_channels,dataset_name): #in_channels,
        super().__init__()
        self.num_classes=num_classes
        maxpoolidx = [1, 3, 7]
        layers = []
        inp = in_channels #in_channels
        oups = {
            0: 64,
            2: 192,
            4: 384,
            5: 256,
            6: 256
        }
        kp = {
            0: (5, 2),
            2: (5, 2),
            4: (3, 1),
            5: (3, 1),
            6: (3, 1)
        }
        for layeridx in range(8):
            if layeridx in maxpoolidx:
                layers.append(nn.MaxPool2d((2,1), (2,1)))
            else:
                k = kp[layeridx][0]
                p = kp[layeridx][1]
                # if passport_kwargs[str(layeridx)]['flag']:
                #     layers.append(PassportPrivateBlock(inp, oups[layeridx], k, 1, p))
                # else:
                layers.append(ConvBlock(inp, oups[layeridx], k, 1, p))
                inp = oups[layeridx]

        self.features = nn.Sequential(*layers)
        if dataset_name == 'mirage':
            in_feature = 3584
        elif dataset_name == 'njupt':
            in_feature = 2816
        elif dataset_name == 'cic':
            in_feature = 1536
        else:
            raise RuntimeError('no modal')
        self.classifier = nn.Linear(in_feature, num_classes)

    def forward(self, x):
        for m in self.features:
            x = m(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        # a=torch.nn.functional.softmax(x[:,0:int(self.num_classes/2)]) #softmax
        # #print(a.size())
        # b=torch.nn.functional.softmax(x[:,int(self.num_classes/2):self.num_classes])
        #print(b.size())
        # z=torch.cat((a,b),dim=1)
        return x

def alexnet(**kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    """
    model = AlexNet(**kwargs)
    return model
