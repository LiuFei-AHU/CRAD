
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryAccuracy
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.metrics import f1_score

from github.CRAD.dataset import NDataSet
import github.CRAD.configs as config
from github.CRAD.models.dbn import DBN
from github.CRAD.models.densenet import densenet121
from github.CRAD.models.mymodel import Teacher, Student, LayerScores
from github.CRAD.models.resnet import resnet18, resnet50
from github.CRAD.models.resnext import resnext50
from github.CRAD.models.shuffleNet import ShuffleNet
from github.CRAD.models.vgg import vgg16
from github.CRAD.utils import AverageMeter, DistillationOrthogonalLoss

model_names = ["resnet18", "vgg16", "resnet50", "DBN", 'densenet121', 'shuffleNet', 'resnext50']
model_outdim = [512, 512, 512, 512, 1024, 512, 512]
layer_dim = [(64, 128, 256, 512), (64, 128, 256, 512), (64, 128, 256, 512),
             (64, 128, 256, 512), (128, 256, 512, 1024), (64, 128, 256, 512), (64, 128, 256, 512)]
models = [resnet18, vgg16, resnet50, DBN, densenet121, ShuffleNet, resnext50]


def init(opt):
    # set device
    if torch.cuda.is_available():
        device = torch.device(opt.device)
        torch.cuda.set_device(device)
        print('Running on the GPU')
    else:
        device = torch.device('cpu')
        # torch.cuda.set_device(torch.device('cpu'))
        print('Running on the CPU')

    opt.device = device


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def weight_func(func, x, y, weight):
    n = x.shape[0]
    ret = 0.
    for i in range(n):
        ret += func(x[i], y[i]) * weight[i]
    return ret / n


def cal_weight(out_t, out_s, label):
    weight = abs(out_t['p'].max(1)[0] - label) / 5
    weight[weight > 1] = 1
    weight = 1 - weight
    return weight


def cal_weight_soft(out_t, out_s, label):
    out_t1, out_s1 = out_t['p'].clone().detach(), out_s['p'].clone().detach()
    d_t = abs(out_t1.max(1)[0] - label) / len(label)
    d_s = abs(out_s1.max(1)[0] - label) / len(label)

    if d_s.mean() > d_t.mean():
        return cal_weight(out_t, out_s, label)
    else:
        psi = (d_s / (d_t + d_s + 1e-12)) ** 2

        weight = d_t
        weight[weight > 1] = 1
        weight = 1 - weight

        return weight * psi


def at_loss(f_s, f_t, weight=0):
    out = ((at(f_s) - at(f_t)).pow(2)).mean()
    return out


def at(f, ab=0):
    if ab == 0:
        xx = f.pow(2).mean(1).view(f.size(0), -1)
    elif ab == 1:
        xx = f.mean(1).pow(2).view(f.size(0), -1)
    elif ab == 2:
        xx = f.mean(1).view(f.size(0), -1)
    elif ab == 3:
        xx = f.view(f.size(0), -1)
    return xx


def cos_loss(x, y):
    flat = torch.nn.Flatten(2)
    x, y = flat(x), flat(y)
    return torch.mean(1 - torch.abs(torch.nn.functional.cosine_similarity(x, y)))


def create_teacher_model(module_path, opt):
    model = Teacher(models[model_names.index(opt.arch)](mode=opt.mode, no_cuda=False),
                    models[model_names.index(opt.arch)](mode=opt.mode, no_cuda=False),
                    output_dim=model_outdim[model_names.index(opt.arch)], cls_num=opt.task.num_class, opt=opt)
    layer_refine = LayerScores(channels=layer_dim[model_names.index(opt.arch)], opt=opt)
    model.to(opt.device)
    layer_refine.to(opt.device)

    # load teacher model
    params_file_path = os.path.join(module_path, "checkpoints", opt.t_model_name + '.pdparams')
    if os.path.exists(params_file_path):
        model.load_state_dict(torch.load(params_file_path, map_location=opt.device), strict=False)
        print('model loaded!')

    # load layer refine
    params_file_path = os.path.join(module_path, "checkpoints", opt.t_model_name + '_refine.pdparams')
    if os.path.exists(params_file_path):
        layer_refine.load_state_dict(torch.load(params_file_path, map_location=opt.device), strict=False)
        print('model refine loaded!')

    return model, layer_refine


def create_student_model(module_path_s, opt):
    # load student
    model_s = Student(models[model_names.index(opt.arch)](mode=opt.mode, no_cuda=False),
                      output_dim=model_outdim[model_names.index(opt.arch)], cls_num=2, opt=opt)
    model_s.to(opt.device)

    params_file_path = os.path.join(module_path_s, "checkpoints", opt.s_model_name + '.pdparams')
    if os.path.exists(params_file_path):
        model_s.load_state_dict(torch.load(params_file_path, map_location=opt.device))
        print('student model loaded!')

    return model_s


def set_opt(opt, arg):
    opt.arch = arg.arch
    opt.t_model_name = arg.teacher_name
    opt.s_model_name = arg.student_name
    opt.task = arg.task
    opt.layerscore = arg.layerscore
    opt.weight_mode = arg.weight_mode
    opt.fea_kd_mode = arg.fea_kd_mode
    opt.use_kd = arg.use_kd


def train_teacher_stas(arg=None):
    # load configs
    opt = config.opt
    init(opt)

    set_opt(opt, arg)

    module_path = str(os.path.join(opt.rslt_path, opt.arch, opt.t_mudule))
    os.makedirs(module_path, exist_ok=True)
    opt.checkpoint_path = os.path.join(module_path, "checkpoints")
    os.makedirs(opt.checkpoint_path, mode=0o777, exist_ok=True)

    # create model and load params
    model, layer_refine = create_teacher_model(module_path, opt)

    train_list = torch.nn.ModuleList()
    train_list.append(model)
    train_list.append(layer_refine)

    # load data
    data_loader = load_data(opt.dataset)
    test_loader = load_data(opt.validation_dataset)

    optimizer = torch.optim.Adam(train_list.parameters(), lr=opt.lr)
    scheduler = None
    early_stop = None
    subfix = ""

    best_acc = 0.5
    min_loss = 100.0
    train_loss = []
    eval_loss = []
    train_acc = []
    eval_acc = []

    print('Start train ..')
    for epoch_id in range(opt.epoch):
        model.train()
        losses = AverageMeter('Loss', ':.4e')
        loss_acc = AverageMeter('Acc', ':6.2f')
        for batch_id, data in enumerate(data_loader):
            optimizer.zero_grad()
            image, label, image2 = data[0:3]
            image, image2, label = image.to(opt.device), image2.to(opt.device), label.to(opt.device)
            mmse = data[3].to(opt.device)
            predict = model(image, image2, mmse=mmse, l=label)
            ort_loss = predict['loss']

            # default setting that includes all layers
            if "m_layers" not in opt.keys():
                opt.m_layers = [0, 1, 2, 3]

            if opt.fea_kd_mode == -1:
                layer_pres = layer_refine(predict['fea'][0] + predict['fea'][1], mmse)
                sts_loss = sum(
                    [torch.nn.functional.smooth_l1_loss(layer_pres[i][1], mmse, reduce=True, reduction='mean') for i in
                     range(len(layer_pres)) if i in opt.m_layers]) / len(layer_pres)
            elif opt.fea_kd_mode == 1:
                layer_pres = layer_refine(predict['fea'], mmse)
                sts_loss = sum(
                    [torch.nn.functional.smooth_l1_loss(layer_pres[i][1], mmse, reduce=True, reduction='mean') for i in
                     range(len(layer_pres)) if i in opt.m_layers])
            else:
                layer_pres = layer_refine(predict['fea'][0], mmse)
                sts_loss = sum(
                    [torch.nn.functional.smooth_l1_loss(layer_pres[i][1], mmse, reduce=True, reduction='mean') for i in
                     range(len(layer_pres)) if i in opt.m_layers])
                layer_pres = layer_refine(predict['fea'][1], mmse)
                sts_loss += sum(
                    [torch.nn.functional.smooth_l1_loss(layer_pres[i][1], mmse, reduce=True, reduction='mean') for i in
                     range(len(layer_pres)) if i in opt.m_layers])
            cls_loss = F.nll_loss(predict['y'], label)
            loss = opt.alpha * cls_loss + opt.gma * sts_loss + opt.beta * ort_loss
            acc = accuracy(predict['y'], label)
            losses.update(loss.item(), image.size(0))
            loss_acc.update(acc.item(), image.size(0))
            loss.backward()
            optimizer.step()
        epoch_loss = losses.avg
        epoch_acc = loss_acc.avg
        train_loss.append(epoch_loss)
        train_acc.append(epoch_acc)

        # testing
        model.eval()
        losses.reset()
        loss_acc.reset()
        with torch.no_grad():
            for batch_id, data in enumerate(test_loader):
                x_data = data[0].to(opt.device)
                x_data2 = data[2].to(opt.device)
                y_data = data[1].to(opt.device)
                mmse = data[3].to(opt.device) if len(data) >= 4 else None
                predicts = model(x_data, x_data2, mmse=mmse, l=y_data)
                predicts = predicts['y']
                loss = F.nll_loss(predicts, y_data)
                acc = accuracy(predicts, y_data)
                losses.update(loss.item(), x_data.size(0))
                loss_acc.update(acc.item(), x_data.size(0))
        val_acc = loss_acc.avg
        val_los = losses.avg
        eval_loss.append(val_los)
        eval_acc.append(val_acc)

        print("epoch: {}, acc is {} ,loss is: {} val acc is {} , val loss is {}"
              .format(epoch_id, epoch_acc, epoch_loss, val_acc, val_los))

        if val_acc > best_acc or (val_acc == best_acc and val_los < min_loss):
            min_loss = val_los if val_los < min_loss else min_loss
            best_acc = val_acc
            best_save_path = opt.checkpoint_path + f'/[T]Task-{opt.task.Task_no}{subfix}_epoch_{epoch_id}_{val_acc:.4f}_{val_los:.4f}'
            torch.save(model.state_dict(), best_save_path + '.pdparams')
            torch.save(layer_refine.state_dict(), best_save_path + '_refine.pdparams')

        if scheduler is not None:
            scheduler.step()

        if early_stop is not None and early_stop(val_acc):
            break

    return train_loss, eval_loss, train_acc, eval_acc


def train_student_stas(arg):
    opt = config.opt
    init(opt)

    set_opt(opt, arg)

    module_path_t = str(os.path.join(opt.rslt_path, opt.arch, opt.t_mudule))
    module_path_s = str(os.path.join(opt.rslt_path, opt.arch, opt.s_mudule))
    os.makedirs(module_path_s, exist_ok=True)
    opt.checkpoint_path = os.path.join(module_path_s, "checkpoints")
    os.makedirs(opt.checkpoint_path, mode=0o777, exist_ok=True)

    # create model and load params
    model_t, layer_refine = create_teacher_model(module_path_t, opt)
    model_s = create_student_model(module_path_s, opt)
    model_t.eval()
    layer_refine.eval()

    train_list = torch.nn.ModuleList()
    train_list.append(model_s)

    # load data
    data_loader = load_data(opt.dataset)
    test_loader = load_data(opt.validation_dataset)

    optimizer = torch.optim.Adam(train_list.parameters(), lr=opt.lr)
    scheduler = None
    early_stop = None
    subfix = ""

    softmax = torch.nn.Softmax()
    dis_kl = DistillationOrthogonalLoss()
    best_acc = 0.8
    min_loss = 100.0
    train_loss = []
    eval_loss = []
    train_acc = []
    eval_acc = []

    print('start train ..')
    for epoch_id in range(opt.epoch):
        model_s.train()

        losses = AverageMeter('Loss', ':.4e')
        loss_acc = AverageMeter('Acc', ':6.2f')
        for batch_id, data in enumerate(data_loader):
            optimizer.zero_grad()

            image, label, image2 = data[0:3]
            image, image2, label = image.to(opt.device), image2.to(opt.device), label.to(opt.device)
            mmse = data[3].to(opt.device) if len(data) >= 4 else None

            out_s = model_s(image, l=label)
            out_t = model_t(image, image2, mmse=mmse, l=label)

            if str(opt.weight_mode) == "2":
                weight = cal_weight_soft(out_t, out_s, label)
            else:
                weight = cal_weight(out_t, out_s, label)

            ort_loss = out_s['loss']
            loss_kd = weight_func(torch.nn.KLDivLoss(log_target=True), out_s['y'], out_t['y'].detach(), weight)
            fi_loss = (weight_func(dis_kl, out_s['mp'][0], out_t['mp'][0].detach(), weight) +
                       weight_func(dis_kl, out_s['mp'][1], out_t['mp'][1].detach(), weight))

            layer_pres_t, layer_pres_s = layer_refine(out_t['fea'], mmse), layer_refine(out_s['fea'])
            if "m_layers" not in opt.keys():
                opt.m_layers = [0, 1, 2, 3]

            if opt.fea_kd_mode == 1:
                refine_loss = sum(
                    [weight_func(torch.nn.SmoothL1Loss(reduce=True, reduction='mean'), layer_pres_s[i][0],
                                 layer_pres_t[i][0].detach(), weight)
                     for i in range(len(layer_pres_t)) if i in opt.m_layers])
            else:
                refine_loss = sum(
                    [weight_func(torch.nn.SmoothL1Loss(reduce=True, reduction='mean'), layer_pres_s[i][0],
                                 layer_pres_t[i][0].detach(), weight)
                     for i in range(len(layer_pres_t)) if i in opt.m_layers])
                refine_loss += sum(
                    [weight_func(torch.nn.SmoothL1Loss(reduce=True, reduction='mean'), layer_pres_s[i][1],
                                 layer_pres_t[i][1].detach(), weight)
                     for i in range(len(layer_pres_t)) if i in opt.m_layers])

            cls_loss = F.nll_loss(out_s['y'], label)
            if opt.use_kd:
                loss = opt.alpha * cls_loss + opt.beta * ort_loss + opt.gma * loss_kd + opt.theta * fi_loss + opt.zeta * refine_loss
            else:
                loss = cls_loss

            acc = accuracy(out_s['y'], label)
            losses.update(loss.item(), image.size(0))
            loss_acc.update(acc.item(), image.size(0))
            loss.backward()
            optimizer.step()

        epoch_loss = losses.avg
        epoch_acc = loss_acc.avg
        train_loss.append(epoch_loss)
        train_acc.append(epoch_acc)

        # testing
        model_s.eval()
        losses.reset()
        loss_acc.reset()
        preds, target = [], []
        with torch.no_grad():
            for batch_id, data in enumerate(test_loader):
                x_data = data[0].to(opt.device)
                y_data = data[1].to(opt.device)
                predicts = model_s(x_data)
                predicts = predicts['y']
                loss = F.nll_loss(predicts, y_data)
                acc = accuracy(predicts, y_data)
                losses.update(loss.item(), x_data.size(0))
                loss_acc.update(acc.item(), x_data.size(0))
                pred = softmax(predicts)
                preds.extend(pred.cpu().detach().max(1)[1])
                target.extend(y_data.cpu().detach())

        val_acc = loss_acc.avg
        val_los = losses.avg
        eval_loss.append(val_los)
        eval_acc.append(val_acc)

        print("epoch: {}, acc is {} ,loss is: {} val acc is {} , val loss is {}".
              format(epoch_id, epoch_acc, epoch_loss, val_acc, val_los))

        if val_acc > best_acc or (val_acc == best_acc and val_los < min_loss):
            min_loss = val_los if val_los < min_loss else min_loss
            best_acc = val_acc
            best_save_path = opt.checkpoint_path + f'/[S]Task-{opt.task.Task_no}{subfix}_epoch_{epoch_id}_{val_acc:.4f}_{val_los:.4f}'
            torch.save(model_s.state_dict(), best_save_path + '.pdparams')

        if scheduler is not None:
            scheduler.step()

        if early_stop is not None and early_stop(val_acc):
            break

    return train_loss, eval_loss, train_acc, eval_acc


def test_teacher(arg):
    # load configs
    opt = config.opt
    init(opt)

    set_opt(opt, arg)

    module_path = str(os.path.join(opt.rslt_path, opt.arch, opt.t_mudule))

    # create model and load params
    model, _ = create_teacher_model(module_path, opt)
    # load data
    data_loader = load_data(opt.test.dataset)

    print("Start testing...")
    preds, target = [], []
    softmax = torch.nn.Softmax()
    model.eval()
    losses = AverageMeter('Loss', ':.4e')
    with torch.no_grad():
        for batch_id, data in enumerate(data_loader):
            x_data = data[0].to(opt.device)
            x_data1 = data[2].to(opt.device)
            y_data = data[1].to(opt.device)
            pred = model(x_data, x_data1, mmse=None)

            loss = F.nll_loss(pred['y'], y_data)
            losses.update(loss.item(), x_data.size(0))
            pred = softmax(pred['y'])
            preds.extend(pred.cpu().detach().max(1)[1])
            target.extend(y_data.cpu().detach())
    metric = BinaryAccuracy()
    acc = metric(torch.as_tensor(preds), torch.as_tensor(target))
    f1 = f1_score(target, preds)
    auc = round(roc_auc_score(target, preds), 4)
    cm = confusion_matrix(target, preds)
    sen = round(cm[1, 1] / float(cm[1, 1] + cm[1, 0]), 4)
    spe = round(cm[0, 0] / float(cm[0, 0] + cm[0, 1]), 4)
    print("acc:{:.4f} auc:{:.4f} sen:{:.4f} spe:{:.4f}".format(acc, auc, sen, spe))
    return preds, target, (acc.item(), auc, sen, spe, f1)


def test_student(arg):
    opt = config.opt
    init(opt)

    set_opt(opt, arg)

    module_path_s = str(os.path.join(opt.rslt_path, opt.arch, opt.s_mudule))
    opt.checkpoint_path = os.path.join(module_path_s, "checkpoints")

    # create model and load params
    model_s = create_student_model(module_path_s, opt)
    model_s.eval()

    # load data
    data_loader = load_data(opt.test.dataset)

    preds, target, auc_preds = [], [], []
    softmax = torch.nn.Softmax()

    with torch.no_grad():
        for batch_id, data in enumerate(data_loader):
            x_data = data[0].to(opt.device)
            y_data = data[1].to(opt.device)
            pred = softmax(model_s(x_data)['y'])
            preds.extend(pred.cpu().detach().max(1)[1])
            auc_preds.extend(pred)
            target.extend(y_data.cpu().detach())
    metric = BinaryAccuracy()
    acc = metric(torch.as_tensor(preds), torch.as_tensor(target))
    auc = round(roc_auc_score(target, preds), 4)
    cm = confusion_matrix(target, preds)
    sen = round(cm[1, 1] / float(cm[1, 1] + cm[1, 0]), 4)
    spe = round(cm[0, 0] / float(cm[0, 0] + cm[0, 1]), 4)
    f1 = f1_score(target, preds, average='binary')
    print("acc:{:.4f} auc:{:.4f} sen:{:.4f} spe:{:.4f}".format(acc, auc, sen, spe))
    return preds, target, (acc.item(), auc, sen, spe, f1)


def load_data(dataset, batch_size=8, training=True, norm=False):
    assert dataset is not None, "dataset MUST NOT BE NONE!"
    data = np.load(dataset)
    mris, labels, pets, stas = data['mris'], data['labels'], data['pets'], data['stas']
    print("lable 1:", len(labels[labels == 1]), "label 0:", len(labels[labels == 0]))

    ds = NDataSet(mris, labels, pets, stas, aug=None, norm=norm)
    data_loader = DataLoader(ds, batch_size=batch_size,  shuffle=True if training else False)
    return data_loader
