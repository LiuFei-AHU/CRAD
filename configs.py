from easydict import EasyDict


class Task:
    task1 = {"Task_no": 1, "name": "AD-NC", "num_class": 2}
    task2 = {"Task_no": 2, "name": "pMCI-sMCI", "num_class": 2}


opt = EasyDict()

env_name = ""

# Basic
opt.arch = 'vgg16'  # default encoder, u can use other encoder
opt.mode = 1
opt.opt_level = 'O1'
opt.ab = 0
opt.T = 1
opt.alpha = 1
opt.beta = 0.8
opt.gma = 0.8
opt.theta = 0.8
opt.zeta = 1.0
opt.scale = 5
opt.lam = 0.5

# ---- Train configs ----
opt.device = 'cuda:0'
opt.rslt_path = 'out'
opt.refine = False
opt.epoch = 150
opt.endure_epochs = 10
opt.batch_size = 8
opt.last_best_acc = 0.94

opt.dataset = './data/demo.npz'
opt.validation_dataset = './data/demo.npz'
opt.lr = 1e-4
opt.aug = False
opt.norm = False

opt.print_scores = True
opt.print_test = False
opt.fea_kd_mode = 1

opt.shuffle = False
opt.m_layers = [0, 1, 2, 3]

# teacher & student
opt.t_mudule = 'teacher'
opt.s_mudule = 'student'
opt.s_contrast = True
opt.t_contrast = True
opt.orthogonal_type_T = 1

opt.t_model_name = 'pretrained teacher model'
opt.s_model_name = 'pretrained student model'

# ---- Test configs ----
opt.test = EasyDict()
opt.test.print_metric = True
opt.test.dataset = './data/demo.npz'
opt.test.batch_size = 8

opt.test.t_model_name = ''
opt.test.s_model_name = ''
