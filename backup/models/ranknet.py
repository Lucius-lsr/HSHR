from torch import nn
from preprocess.data_helper import *


class RankNetLossFuncV1(nn.Module):
    def __init__(self):
        super(RankNetLossFuncV1, self).__init__()
        return

    def forward(self, gd_time1, gd_time2, pred_surv1, pred_surv2):  # longer_st_time_data is a list of hazard
        P_label = torch.exp(gd_time1) / (torch.exp(gd_time1) + torch.exp(gd_time2))  # this definition is unsure
        O_12 = pred_surv1 - pred_surv2
        loss = -P_label * O_12 + torch.log(torch.add(torch.exp(O_12), 1))
        return loss


class RankNetLossFuncV2(nn.Module):  # https://pdfs.semanticscholar.org/0df9/c70875783a73ce1e933079f328e8cf5e9ea2.pdf
    def __init__(self):
        super(RankNetLossFuncV2, self).__init__()
        return

    def forward(self, gd_time1, gd_time2, pred_surv1, pred_surv2):  # longer_st_time_data is a list of hazard
        if gd_time1 > gd_time2:
            S = 1
        elif gd_time1 == gd_time2:
            S = 0
        else:
            S = -1

        sigmoid_res = torch.sigmoid(pred_surv1 - pred_surv2)
        loss = 0.5*(1-S)*sigmoid_res+torch.log(1+torch.exp(-sigmoid_res))
        return loss


criterion = RankNetLossFuncV2()
