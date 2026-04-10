import argparse

from github.CRAD.configs import Task
from github.CRAD.train_test import train_teacher_stas as train_teacher, train_student_stas as train_student, test_teacher, test_student


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, help='task name', default="task1")
    parser.add_argument('--run_mode', type=str, help='running mode: train, test', default="train")
    parser.add_argument('--use_kd', type=str2bool, help='include kd loss?', default="y")
    parser.add_argument('--run_teacher', type=str2bool, help='run teacher or student', default="y")
    parser.add_argument('--teacher_name', type=str, help='name of trained teacher model', default="")
    parser.add_argument('--student_name', type=str, help='name of trained student model', default="")
    # when train teacher model, fea_kd_mode in [-1, 1, 2], means how to refine intermediate features,
    # when train student model, fea_kd_mode in [1, 2], means align intermediate features and attention scores
    parser.add_argument('--fea_kd_mode', type=int, help='', default=1)
    parser.add_argument('--arch', type=str, help='CNN backbone', default="vgg16")
    parser.add_argument('--layerscore', type=str, help='layer attention name', default="PFAC_wn")
    parser.add_argument('--weight_mode', type=str, help='calculate the gating weights for sdu: 2 is soft weight', default="2")

    args = parser.parse_args()
    if args.task == "task1":
        args.task = Task.task1
    else:
        args.task = Task.task2

    return args


def str2bool(v):
    """
    true/false, yes/no, y/n, 1/0
    """
    if isinstance(v, bool):
        return v
    v_lower = v.strip().lower()
    if v_lower in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v_lower in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError(f"invalid value: '{v}'")


def train(arg):
    if arg.run_teacher:
        train_teacher(arg)
    else:
        train_student(arg)


def test(arg):
    if arg.run_teacher:
        test_teacher(arg)
    else:
        test_student(arg)


if __name__ == "__main__":
    args = parse_args()

    if args.run_mode == "train":
        train(args)
    if args.run_mode == "test":
        test(args)