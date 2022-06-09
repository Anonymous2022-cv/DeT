import time

from torch.autograd import Variable
from torch.utils.data import DataLoader
from data_utils import TestsetLoader, ycbcr2rgb
from models.DeF import DeF
from torchvision.transforms import ToPILImage
import numpy as np
import os
import argparse
import torch
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--degradation", type=str, default='BD')
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument('--gpu_mode', type=bool, default=True)
    parser.add_argument('--testset_dir', type=str, default='/test/data')
    parser.add_argument('--chop_forward', type=bool, default=False)
    return parser.parse_args()


def main(cfg):
    # model
    net = DeF()
    ckpt = torch.load('log/model/DeF.pth') # Iteration--- 49001,   loss---0.000109
    net.load_state_dict(ckpt)
    if cfg.gpu_mode:
        net.cuda()

    with torch.no_grad():
        video_list = os.listdir(cfg.testset_dir)

        for idx_video in range(len(video_list)):
            video_name = video_list[idx_video]

            # dataloader
            test_set = TestsetLoader(cfg, video_name)
            test_loader = DataLoader(test_set, num_workers=1, batch_size=1, shuffle=False)

            for idx_iter, (IR0_even, IR1_odd, IR2_even) in enumerate(test_loader):
                # data
                IR0_even = Variable(IR0_even)
                IR1_odd = Variable(IR1_odd)
                IR2_even = Variable(IR2_even)
                s_time = time.time()
                # LR_y_cube = LR_y_cube.view(b, -1, 1, h_lr, w_lr)

                if cfg.gpu_mode:
                    IR0_even = IR0_even.cuda()
                    IR1_odd = IR1_odd.cuda()
                    IR2_even = IR2_even.cuda()
                    IR0_even = IR0_even.unsqueeze(1)
                    IR1_odd = IR1_odd.unsqueeze(1)
                    IR2_even = IR2_even.unsqueeze(1)
                    input = torch.cat((IR0_even, IR1_odd, IR2_even), dim=1)

                    re_frame = net(input)

                re_frame = np.array(re_frame.data.cpu())

                re_frame = re_frame.squeeze(0).transpose(1, 2, 0)
                re_frame = re_frame * 255.0
                re_frame = np.clip(re_frame, 0, 255)
                re_frame = ToPILImage()(np.round(re_frame).astype(np.uint8))
                re_frame.save('/home/lhd/project/lab(1)-without-Desnet/data/results/real_interlace/' + video_name + '/re_' + str(idx_iter+1) + '.png')
                print('/home/lhd/project/lab(1)-without-Desnet/data/results/real_interlace/' + video_name + '/re_' + str(idx_iter+1) + '.png')

                print('time: {} sec'.format(time.time() - s_time))
                # if not os.path.exists('results/Vid4'):
                #     os.mkdir('results/Vid4')
                # if not os.path.exists('results/Vid4/' + cfg.degradation + '_x' + str(cfg.scale)):
                #     os.mkdir('results/Vid4/' + cfg.degradation + '_x' + str(cfg.scale))
                # if not os.path.exists('results/Vid4/' + cfg.degradation + '_x' + str(cfg.scale) + '/' + video_name):
                #     os.mkdir('results/Vid4/' + cfg.degradation + '_x' + str(cfg.scale) + '/' + video_name)
                # re_frame.save('results/Vid4/' + cfg.degradation + '_x' + str(cfg.scale) + '/' + video_name + '/sr_' + str(idx_iter+2).rjust(2,'0') + '.png')


if __name__ == '__main__':
    cfg = parse_args()
    s_time = time.time()
    main(cfg)
    print('time: {} sec'.format(time.time() - s_time))