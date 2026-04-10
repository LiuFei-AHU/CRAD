"""
For paper accepted in CMIG 2025:
paper: "CRAD: Cognitive Aware Feature Refinement with Missing Modalities for Early Alzheimer’s Progression Prediction",
url: https://www.sciencedirect.com/science/article/pii/S0895611125001739
"""

import torch.nn.functional as F
import torch
from torch import nn


class ChannelAttention(nn.Module):
    def __init__(self, c_in, m_in):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.fc1 = nn.Linear(c_in, m_in)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(m_in, c_in)
        self.flatten = nn.Flatten()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_x = self.avg_pool(x)
        max_x = self.max_pool(x)
        avg_x = torch.reshape(avg_x, [avg_x.shape[0], -1])
        max_x = torch.reshape(max_x, [max_x.shape[0], -1])
        avg_out = self.fc2(self.relu1(self.fc1(avg_x)))
        max_out = self.fc2(self.relu1(self.fc1(max_x)))
        out = (avg_out + max_out).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        return self.sigmoid(out)


class ChannelAttention_wn(nn.Module):
    def __init__(self, c_in, m_in, _lambda=1e-4):
        super(ChannelAttention_wn, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.theta = nn.ReLU()
        self.flatten = nn.Flatten(2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = F.normalize(x, p=2, dim=1)
        avg_x = self.avg_pool(x).squeeze(-1).squeeze(-1)
        max_x = self.max_pool(x).squeeze(-1).squeeze(-1)
        out = self.theta(avg_x + max_x)
        out = torch.matmul(out, out.permute(0, 2, 1))
        out = torch.diagonal(out, dim1=-2, dim2=-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        return self.sigmoid(out)


class ChannelAttention_wnma(nn.Module):
    def __init__(self, c_in, m_in, _lambda=1e-4):
        super(ChannelAttention_wnma, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.flatten = nn.Flatten(2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = F.normalize(x, p=2, dim=1)
        avg_x = self.avg_pool(x).squeeze(-1).squeeze(-1)
        max_x = self.max_pool(x).squeeze(-1).squeeze(-1)
        out = torch.matmul(avg_x, max_x.permute(0, 2, 1))
        out = torch.diagonal(out, dim1=-2, dim2=-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        return self.sigmoid(out)


class ChannelAttention_wnma_paca(nn.Module):
    def __init__(self, c_in, m_in, _lambda=1e-4):
        super(ChannelAttention_wnma_paca, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.theta = nn.ReLU()
        self.flatten = nn.Flatten(2)
        self.sigmoid = nn.Sigmoid()
        self.e_lambda = _lambda

    def forward(self, x):
        x = F.normalize(x, p=2, dim=1)
        b, c, d, h, w = x.size()
        n = d * w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[2, 3, 4], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3, 4], keepdim=True) / n + self.e_lambda)) + 0.5

        avg_x = self.avg_pool(x).squeeze(-1).squeeze(-1)
        max_x = self.max_pool(x).squeeze(-1).squeeze(-1)

        out = self.theta(avg_x + max_x)
        out = torch.matmul(out, out.permute(0, 2, 1))
        out = torch.diagonal(out, dim1=-2, dim2=-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        return self.sigmoid(out) * y


class LayerScores(nn.Module):
    def __init__(self, channels=(64, 128, 256, 512), opt=None):
        super(LayerScores, self).__init__()
        if opt is not None and opt.layerscore == "PFAC_wn":
            self.layer1 = LayerScore_PFCAE_wn(channels[0])
            self.layer2 = LayerScore_PFCAE_wn(channels[1])
            self.layer3 = LayerScore_PFCAE_wn(channels[2])
            self.layer4 = LayerScore_PFCAE_wn(channels[3])
        elif opt is not None and opt.layerscore == "PFAC_wnma":
            self.layer1 = LayerScore_PFCAE_wnma(channels[0])
            self.layer2 = LayerScore_PFCAE_wnma(channels[1])
            self.layer3 = LayerScore_PFCAE_wnma(channels[2])
            self.layer4 = LayerScore_PFCAE_wnma(channels[3])
        elif opt is not None and opt.layerscore == "PFAC_wnma_paca":
            self.layer1 = LayerScore_PFCAE_wnma_paca(channels[0])
            self.layer2 = LayerScore_PFCAE_wnma_paca(channels[1])
            self.layer3 = LayerScore_PFCAE_wnma_paca(channels[2])
            self.layer4 = LayerScore_PFCAE_wnma_paca(channels[3])
        else:
            self.layer1 = LayerScore(channels[0])
            self.layer2 = LayerScore(channels[1])
            self.layer3 = LayerScore(channels[2])
            self.layer4 = LayerScore(channels[3])

    def forward(self, x, m=None, re_att=False):
        x1 = self.layer1(x[0], m, re_att)
        x2 = self.layer2(x[1], m, re_att)
        x3 = self.layer3(x[2], m, re_att)
        x4 = self.layer4(x[3], m, re_att)

        return x1, x2, x3, x4


class LayerScore(nn.Module):
    def __init__(self, block_num: int, cls_num=1):
        super(LayerScore, self).__init__()
        self.global_att = ChannelAttention(block_num, block_num)
        self.fc = nn.Sequential(nn.Linear(block_num, cls_num))
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.sigmoid = nn.Sigmoid()
        self.block_num = block_num

    def forward(self, x, m=None, att=False):
        g_att = self.global_att(x) * x
        x = self.pool(x)
        rst = self.fc(x.view(x.shape[0], -1))
        return g_att, rst


class LayerScore_PFCAE_wn(nn.Module):
    def __init__(self, block_num: int, cls_num=1):
        super(LayerScore_PFCAE_wn, self).__init__()
        self.global_att = ChannelAttention_wn(block_num, 3)
        self.fc = nn.Sequential(nn.Linear(block_num, cls_num))
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.sigmoid = nn.Sigmoid()
        self.block_num = block_num

    def forward(self, x, m=None, re_att=False):
        score = self.global_att(x)
        g_att = score * x
        x = self.pool(g_att)
        rst = self.fc(x.view(x.shape[0], -1))
        if not re_att:
            return g_att, rst
        else:
            return g_att, score


class LayerScore_PFCAE_wnma(nn.Module):
    def __init__(self, block_num: int, cls_num=1):
        super(LayerScore_PFCAE_wnma, self).__init__()
        self.global_att = ChannelAttention_wnma(block_num, 3)
        self.fc = nn.Sequential(nn.Linear(block_num, cls_num))
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.sigmoid = nn.Sigmoid()
        self.block_num = block_num

    def forward(self, x, m=None, re_att=False):
        score = self.global_att(x)
        g_att = score * x
        x = self.pool(g_att)
        rst = self.fc(x.view(x.shape[0], -1))
        if not re_att:
            return g_att, rst
        else:
            return g_att, score


class LayerScore_PFCAE_wnma_paca(nn.Module):
    def __init__(self, block_num: int, cls_num=1):
        super(LayerScore_PFCAE_wnma_paca, self).__init__()
        self.global_att = ChannelAttention_wnma_paca(block_num, 3)
        self.fc = nn.Sequential(nn.Linear(block_num, cls_num))
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.sigmoid = nn.Sigmoid()
        self.block_num = block_num

    def forward(self, x, m=None):
        g_att = self.global_att(x) * x
        x = self.pool(g_att)
        rst = self.fc(x.view(x.shape[0], -1))

        return g_att, rst


def fusion_fi(output_dim):
    return nn.Sequential(nn.Conv3d(output_dim * 2, output_dim, kernel_size=1, bias=False),
                         nn.Conv3d(output_dim, output_dim, padding=1, kernel_size=3, bias=False),
                         nn.ReLU(),
                         nn.Conv3d(output_dim, output_dim, padding=0, kernel_size=1, bias=False),
                         nn.AdaptiveAvgPool3d(1),
                         )


def classifier(output_dim, cls_num):
    return nn.Sequential(nn.Conv3d(output_dim, output_dim // 2, padding=0, kernel_size=1),
                         nn.ReLU(),
                         nn.Conv3d(output_dim // 2, cls_num, padding=0, kernel_size=1))


def ort_project_loss(x, y, label=None):
    flatten = nn.Flatten(1)
    distinct_label = set(label.tolist())
    loss = None
    for lbl in distinct_label:
        pb = torch.nonzero(label == int(lbl))
        x1 = torch.index_select(x, dim=0, index=pb[:, 0])
        y1 = torch.index_select(y, dim=0, index=pb[:, 0])
        x1, y1 = flatten(x1), flatten(y1)
        x1, y1 = F.normalize(x1, p=2, dim=1), F.normalize(y1, p=2, dim=1)
        device = (torch.device('cuda:0') if x1.is_cuda else torch.device('cpu'))
        eye = torch.eye(x1.shape[0]).to(device)
        dot_prod = torch.matmul(x1, y1.T)
        ort_loss = torch.abs(dot_prod - eye).mean()
        loss = ort_loss if loss is None else loss + ort_loss
    return loss


def ort_project_loss2(x, y, label=None):
    flatten = nn.Flatten(1)
    x, y = flatten(x), flatten(y)
    x, y = F.normalize(x, p=2, dim=1), F.normalize(y, p=2, dim=1)
    neg_dot_prod = torch.abs(torch.matmul(x, y.T)).mean()
    pos_dot_prod = (torch.abs(torch.matmul(x, x.T)).mean() + torch.abs(torch.matmul(y, y.T)).mean()) * 0.5
    loss = (1.0 - pos_dot_prod) + neg_dot_prod
    return loss


def ort_project_loss3(x, y, label=None):
    flatten = nn.Flatten(1)
    x, y = flatten(x), flatten(y)
    x, y = F.normalize(x, p=2, dim=1), F.normalize(y, p=2, dim=1)

    labels = label[:, None]
    device = 'cuda:0'
    mask = torch.eq(labels, labels.t()).bool().to(device)
    eye = torch.eye(mask.shape[0], mask.shape[1]).bool().to(device)

    mask_pos = mask.masked_fill(eye, 0).float()
    mask_neg = (~mask).float()

    neg_dot_prod = torch.abs(torch.matmul(x, y.T)).mean()
    pos_dot_prod = (torch.abs(mask_pos * torch.matmul(x, x.T)).mean() + torch.abs(
        mask_pos * torch.matmul(y, y.T)).mean()) * 0.5
    loss = (1.0 - pos_dot_prod) + neg_dot_prod
    return loss


class OrthogonalProjectionLoss(nn.Module):
    def __init__(self, gamma=0.5):
        super(OrthogonalProjectionLoss, self).__init__()
        self.gamma = gamma
        self.flatten = nn.Flatten(1)

    def forward(self, x, labels=None):
        device = (torch.device('cuda:0') if x.is_cuda else torch.device('cpu'))
        x = self.flatten(x)
        features = F.normalize(x, p=2, dim=1)

        labels = labels[:, None]

        mask = torch.eq(labels, labels.t()).bool().to(device)
        eye = torch.eye(mask.shape[0], mask.shape[1]).bool().to(device)

        mask_pos = mask.masked_fill(eye, 0).float()
        mask_neg = (~mask).float()
        dot_prod = torch.matmul(features, features.t())

        pos_pairs_mean = (mask_pos * dot_prod).sum() / (mask_pos.sum() + 1e-6)
        neg_pairs_mean = torch.abs(mask_neg * dot_prod).sum() / (mask_neg.sum() + 1e-6)
        loss = (1.0 - pos_pairs_mean) + self.gamma * neg_pairs_mean

        return loss


class Teacher(nn.Module):
    def __init__(self, backbone_m, backbone_p, output_dim, cls_num, opt=None):
        super(Teacher, self).__init__()
        self.opt = opt
        self.backbone_m = backbone_m
        self.backbone_p = backbone_p
        self.om_proj = nn.Sequential(nn.Conv3d(output_dim, output_dim, padding=0, kernel_size=1, bias=False),
                                     nn.ReLU())
        self.op_proj = nn.Sequential(nn.Conv3d(output_dim, output_dim, padding=0, kernel_size=1, bias=False),
                                     nn.ReLU())
        self.Fi = fusion_fi(output_dim)
        self.classifier = classifier(output_dim, cls_num)
        self.ort_loss = OrthogonalProjectionLoss()

    def forward(self, m, p, l=None, mmse=None):
        fea = None
        m, p = self.backbone_m(m), self.backbone_p(p)
        if isinstance(m, tuple):
            fea = [m, p] if self.opt.fea_kd_mode == 2 or self.opt.fea_kd_mode == -1 else m
            m, p = m[-1], p[-1]
        m, p = self.om_proj(m), self.op_proj(p)
        mp = torch.cat([m, p], 1)
        fi = self.Fi(mp)
        if l is not None:
            if self.opt.orthogonal_type_T == 1:
                ort_loss = ort_project_loss(m, p, l) + self.ort_loss(fi, l)
            elif self.opt.orthogonal_type_T == 2:
                ort_loss = ort_project_loss2(m, p, l) + self.ort_loss(fi, l)
            elif self.opt.orthogonal_type_T == 3:
                ort_loss = ort_project_loss3(m, p, l) + self.ort_loss(fi, l)
        else:
            ort_loss = torch.tensor(0.)
        re = self.classifier(fi)
        xx = re.view(re.size(0), -1)
        log_out = F.log_softmax(xx, dim=1)
        p_out = F.softmax(xx, dim=1)
        ret = {
            'logit': xx,
            'y': log_out,
            'p': p_out,
            'loss': ort_loss,
            'mp': [m, p],
            'fea': fea,
        }
        return ret


class Student(nn.Module):
    def __init__(self, backbone, output_dim, cls_num, opt=None):
        super(Student, self).__init__()
        self.opt = opt
        self.backbone = backbone
        self.om_proj = nn.Sequential(nn.Conv3d(output_dim, output_dim, padding=1, kernel_size=3, bias=False),
                                     nn.ReLU(),
                                     nn.Conv3d(output_dim, output_dim, padding=0, kernel_size=1, bias=False))
        self.op_proj = nn.Sequential(nn.Conv3d(output_dim, output_dim, padding=1, kernel_size=3, bias=False),
                                     nn.ReLU(),
                                     nn.Conv3d(output_dim, output_dim, padding=0, kernel_size=1, bias=False))
        self.fim = nn.Sequential(nn.Conv3d(output_dim, output_dim, padding=0, kernel_size=1, bias=False),
                                 nn.ReLU(),
                                 nn.Conv3d(output_dim, output_dim, padding=0, kernel_size=1, bias=False), )
        self.fip = nn.Sequential(nn.Conv3d(output_dim, output_dim, padding=0, kernel_size=1, bias=False),
                                 nn.ReLU(),
                                 nn.Conv3d(output_dim, output_dim, padding=0, kernel_size=1, bias=False), )
        self.Fi = fusion_fi(output_dim)
        self.classifier = classifier(output_dim, cls_num)
        self.ort_loss = OrthogonalProjectionLoss()

    def forward(self, m, l=None):
        fea = None
        m = self.backbone(m)
        if isinstance(m, tuple):
            fea, m = m, m[-1]
        m, p = self.om_proj(m), self.op_proj(m)
        m, p = self.fim(m), self.fip(p)
        mp = torch.cat([m, p], 1)
        fi = self.Fi(mp)
        ort_loss = self.ort_loss(fi, l) if l is not None else torch.tensor(0.)
        re = self.classifier(fi)
        xx = re.view(re.size(0), -1)
        log_out = F.log_softmax(xx, dim=1)
        p_out = F.softmax(xx, dim=1)
        ret = {
            'logit': xx,
            'y': log_out,
            'p': p_out,
            'loss': ort_loss,
            'mp': [m, p],
            'fea': fea
        }
        return ret
