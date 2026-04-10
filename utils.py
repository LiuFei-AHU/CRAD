import torch
from torch import nn
import torch.nn.functional as F


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class DistillationOrthogonalLoss(nn.Module):
    def __init__(self):
        super(DistillationOrthogonalLoss, self).__init__()

    @staticmethod
    def forward(features, features_teacher):
        flatten = nn.Flatten(1)
        features, features_teacher = flatten(features), flatten(features_teacher)
        #  features are normalized
        features = F.normalize(features, p=2, dim=1)
        features_teacher = F.normalize(features_teacher, p=2, dim=1)
        # dot products calculated
        dot_prod = torch.matmul(features, features.t())
        dot_prod_teacher = torch.matmul(features_teacher, features_teacher.t())
        tau = 1
        loss = abs(F.kl_div(
            dot_prod / tau,
            dot_prod_teacher / tau,
            reduction='sum',
            log_target=True
        ) * (tau * tau) / dot_prod_teacher.numel())

        return loss


class DistillKL(nn.Module):
    def __init__(self):
        super(DistillKL, self).__init__()
        self.T = 1.0

    def forward(self, y_s, y_t):
        y_s, y_t = y_s.view(y_s.size(0), -1), y_t.view(y_t.size(0), -1)
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        loss = F.kl_div(p_s, p_t.detach(), reduction='sum') * (self.T**2) / y_s.shape[0]
        return loss
