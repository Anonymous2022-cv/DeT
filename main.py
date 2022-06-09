from torch.autograd import Variable
from torch.utils.data import DataLoader
from data_utils import TrainsetLoader
from models.DeF import DeF
import torch.backends.cudnn as cudnn
import argparse
import torch
import numpy as np
import torch.nn.functional as F
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from matplotlib import pyplot as plt

train_data = []
with open("/home/lhd/project/real-time-deinterlacing(1)/data/train/vimeo_septuplet/sep_trainlist.txt",
          "r") as f:
    for line in f.readlines():
        line = line.strip('\n')
        train_data.append(line)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--degradation", type=str, default='BI')
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument('--gpu_mode', type=bool, default=True)
    parser.add_argument('--patch_size', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--n_iters', type=int, default=50000, help='number of iterations to train')
    parser.add_argument('--trainset_dir', type=str, default='data/train')
    parser.add_argument('--train_data', type=str, default=train_data)
    return parser.parse_args()


def main(cfg):
    # model
    net = DeF()
    if cfg.gpu_mode:
        net.cuda()
    cudnn.benchmark = True

    # dataloader
    train_set = TrainsetLoader(cfg)
    train_loader = DataLoader(train_set, num_workers=4, batch_size=cfg.batch_size, shuffle=True)

    # train
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
    milestones = [15000, 40000]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    criterion = torch.nn.MSELoss()
    loss_list = []

    loss1 = []
    flag = []
    for idx_iter, (IR0_even, IR1_odd, IR2_even, IR1) in enumerate(train_loader):
        scheduler.step()


        IR0_even, IR1_odd, IR2_even, IR1 = Variable(IR0_even), Variable(IR1_odd), Variable(IR2_even), Variable(IR1)
        if cfg.gpu_mode:
            IR0_even = IR0_even.cuda()
            IR1_odd = IR1_odd.cuda()
            IR2_even = IR2_even.cuda()
            IR1 = IR1.cuda()
        IR0_even = IR0_even.unsqueeze(1)
        IR1_odd = IR1_odd.unsqueeze(1)
        IR2_even = IR2_even.unsqueeze(1)
        input = torch.cat((IR0_even, IR1_odd, IR2_even), dim=1)

        rec_frame = net(input)

        # loss
        loss = criterion(rec_frame, IR1)
        loss_list.append(loss.data.cpu())

        # backwards
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # save checkpoint
        if idx_iter % 1000 == 0:
            print('Iteration---%6d,   loss---%f' % (idx_iter + 1, np.array(loss_list).mean()))

            flag.append(idx_iter + 1)
            loss1.append(np.array(loss_list).mean())

            save_path = 'log/set_net_pred_even' + '_x' + str(cfg.scale)
            save_name = 'loss_' + str(np.array(loss_list).mean()) + '_iter' + str(idx_iter) + '.pth'
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            torch.save(net.state_dict(), save_path + '/' + save_name)
            loss_list = []

    plt.title("Loss")
    plt.xlabel("x idx_iter")
    plt.ylabel("y loss")
    plt.plot(flag, loss1)
    plt.show()

if __name__ == '__main__':
    cfg = parse_args()
    main(cfg)
