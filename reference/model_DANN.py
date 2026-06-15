from typing import List, Dict
import torch.nn as nn


class CFE(nn.Module):
    def __init__(self):
        super(CFE, self).__init__()
        self.module = nn.Sequential(
            nn.Linear(310, 512),
            nn.BatchNorm1d(512, momentum=0.1, affine=False),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 320),
            nn.BatchNorm1d(320, momentum=0.1, affine=False),
            nn.ReLU(True),
            nn.Dropout(p=0.5)
        )

    def forward(self, x):
        x = self.module(x)
        return x

class DSFE(nn.Module):
    def __init__(self):
        super(DSFE, self).__init__()
        self.module = nn.Sequential(
            nn.Linear(320, 320),
            nn.BatchNorm1d(320, momentum=0.1, affine=False),
            nn.ReLU(True),
            nn.Dropout(p=0.5)
        )

    def forward(self, x):
        x = self.module(x)
        return x



class MSMDAERNet(nn.Module):
    def __init__(self, number_of_source=-1, number_of_category=-1):
        super(MSMDAERNet, self).__init__()
        # 一个共享的CFE
        self.number_of_source = number_of_source
        self.number_of_category = number_of_category
        self.sharedNet = CFE()
        # N个DSFE，N个DSC
        for i in range(self.number_of_source):
            exec('self.DSFE' + str(i) + '=DSFE()')
            exec('self.cls_fc_DSC' + str(i) +
                 '=nn.Linear(320,' + str(number_of_category) + ')')

    def forward(self, data, index):
        if self.training == True:
            data = data.reshape(data.size(0), -1)
            feature1 = self.sharedNet(data)
            DSFE_name = 'self.DSFE' + str(index)
            feature2 = eval(DSFE_name)(feature1)
            DSC_name = 'self.cls_fc_DSC' + str(index)
            output = eval(DSC_name)(feature2)
            return feature1, feature2, output
        else:
            data = data.reshape(data.size(0), -1)
            feature = self.sharedNet(data)
            pred = []
            for i in range(self.number_of_source):
                DSFE_name = 'self.DSFE' + str(i)
                DSC_name = 'self.cls_fc_DSC' + str(i)
                feature_DSFE_i = eval(DSFE_name)(feature)
                pred.append(eval(DSC_name)(feature_DSFE_i))
            return feature, sum(pred)/len(pred)
    def get_parameters(self) -> List[Dict]:
        params = [
            {"params": self.sharedNet.parameters(), "lr_mult": 1},
        ]
        for i in range(self.number_of_source):
            params.append({"params": eval('self.DSFE' + str(i) + '.parameters()'), "lr_mult": 1})
            params.append({"params": eval('self.cls_fc_DSC' + str(i) + '.parameters()'), "lr_mult": 1})
        return params