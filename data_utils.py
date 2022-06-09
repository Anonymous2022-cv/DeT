from PIL import Image
from torch.utils.data.dataset import Dataset
import numpy as np
import os
import torch
import random
import cv2


class TrainsetLoader(Dataset):
    def __init__(self, cfg):
        super(TrainsetLoader).__init__()
        self.trainset_dir = cfg.trainset_dir
        self.scale = cfg.scale
        self.patch_size = cfg.patch_size
        self.n_iters = cfg.n_iters * cfg.batch_size
        # self.video_list = os.listdir(cfg.trainset_dir)
        self.train_data = cfg.train_data
        # self.degradation = cfg.degradation

    def __getitem__(self, idx):
        idx = random.randint(0, len(self.train_data) - 1)
        idx_frame = random.randint(1, 5)  # #frames of training videos is 31, 31-3=28
        ir_dir = '/home/lhd/project/real-time-deinterlacing(1)/' + self.trainset_dir + '/' + 'vimeo_septuplet/' + 'sequences/' + \
                 self.train_data[idx]

        # read HR & LR frames
        IR0 = Image.open(ir_dir + '/im' + str(idx_frame) + '.png')
        IR1 = Image.open(ir_dir + '/im' + str(idx_frame + 1) + '.png')
        IR2 = Image.open(ir_dir + '/im' + str(idx_frame + 2) + '.png')

        IR0 = np.array(IR0, dtype=np.float32) / 255.0
        IR1 = np.array(IR1, dtype=np.float32) / 255.0
        IR2 = np.array(IR2, dtype=np.float32) / 255.0

        # 第一帧为偶数行，第二帧为奇数行，第三帧为偶数行
        IR0_even = init_frame(generate(IR0, flag=1), flag=0)
        IR1_odd = init_frame(generate(IR1, flag=0), flag=1)
        IR2_even = init_frame(generate(IR2, flag=1), flag=0)


        # crop patchs randomly
        IR0_even, IR1_odd, IR2_even, IR1 = random_crop(IR0_even, IR1_odd, IR2_even, IR1, self.patch_size, self.scale)


        # data augmentation
        IR0_even, IR1_odd, IR2_even, IR1 = augmentation()(IR0_even, IR1_odd, IR2_even, IR1)
        return toTensor(IR0_even), toTensor(IR1_odd), toTensor(IR2_even), toTensor(IR1)

    def __len__(self):
        return self.n_iters


class TestsetLoader(Dataset):
    def __init__(self, cfg, video_name):
        super(TestsetLoader).__init__()
        self.dataset_dir = cfg.testset_dir + '/' + video_name
        self.degradation = cfg.degradation
        self.scale = cfg.scale
        self.video_name = video_name
        self.frame_list = os.listdir(cfg.testset_dir + '/' + video_name)

    def __getitem__(self, idx):
        dir = self.dataset_dir
        LR0 = Image.open(dir + '/' + str(idx) + '.png')
        LR1 = Image.open(dir + '/' + str(idx + 1) + '.png')
        LR2 = Image.open(dir + '/' + str(idx + 2) + '.png')
        W, H = LR1.size

        # H and W should be divisible by 2
        W = int(W // 2) * 2
        H = int(H // 2) * 2
        LR0 = LR0.crop([0, 0, W, H])
        LR1 = LR1.crop([0, 0, W, H])
        LR2 = LR2.crop([0, 0, W, H])


        LR0 = np.array(LR0, dtype=np.float32) / 255.0
        LR1 = np.array(LR1, dtype=np.float32) / 255.0
        LR2 = np.array(LR2, dtype=np.float32) / 255.0

        # LR0 = LR0[:-2, :,:]
        # LR1 = LR1[:-2, :,:]
        # LR2 = LR2[:-2, :,:]

        IR0_even = init_frame(generate(LR0, flag=1), flag=0)
        IR1_odd = init_frame(generate(LR1, flag=0), flag=1)
        IR2_even = init_frame(generate(LR2, flag=1), flag=0)


        return toTensor(IR0_even), toTensor(IR1_odd), toTensor(IR2_even)

    def __len__(self):
        return len(self.frame_list) - 2


class augmentation(object):
    def __call__(self, IR0_even, IR1_odd, IR2_even, IR1):
        if random.random() < 0.5:
            IR0_even = IR0_even[:, ::-1, :]
            IR1_odd = IR1_odd[:, ::-1, :]
            IR2_even = IR2_even[:, ::-1, :]
            IR1 = IR1[:, ::-1, :]
        if random.random() < 0.5:
            IR0_even = IR0_even[::-1, :, :]
            IR1_odd = IR1_odd[::-1, :, :]
            IR2_even = IR2_even[::-1, :, :]
            IR1 = IR1[::-1, :, :]
        if random.random() < 0.5:
            IR0_even = IR0_even.transpose(1, 0, 2)
            IR1_odd = IR1_odd.transpose(1, 0, 2)
            IR2_even = IR2_even.transpose(1, 0, 2)
            IR1 = IR1.transpose(1, 0, 2)
        return np.ascontiguousarray(IR0_even), np.ascontiguousarray(IR1_odd), np.ascontiguousarray(
            IR2_even), np.ascontiguousarray(IR1)


def random_crop(IR0_even, IR1_odd, IR2_even, IR1, patch_size_lr, scale):
    h_hr, w_hr, _ = IR0_even.shape
    h_lr = h_hr
    w_lr = w_hr
    idx_h = random.randint(10, h_lr - patch_size_lr - 10)
    idx_w = random.randint(10, w_lr - patch_size_lr - 10)

    h_start_hr = (idx_h - 1) * scale
    h_end_hr = (idx_h - 1 + patch_size_lr) * scale
    w_start_hr = (idx_w - 1) * scale
    w_end_hr = (idx_w - 1 + patch_size_lr) * scale

    h_start_lr = idx_h - 1
    h_end_lr = idx_h - 1 + patch_size_lr
    w_start_lr = idx_w - 1
    w_end_lr = idx_w - 1 + patch_size_lr

    IR0_even = IR0_even[h_start_lr:h_end_lr, w_start_lr:w_end_lr]
    IR1_odd = IR1_odd[h_start_lr:h_end_lr, w_start_lr:w_end_lr]
    IR2_even = IR2_even[h_start_lr:h_end_lr, w_start_lr:w_end_lr]
    IR1 = IR1[h_start_lr:h_end_lr, w_start_lr:w_end_lr]
    return IR0_even, IR1_odd, IR2_even, IR1


def generate(frame, flag):
    '''
        flag = 0 奇数行
        flag = 1 偶数行
    '''
    zero_frame = np.zeros((frame.shape), dtype=np.float32)
    for i in range(flag, frame.shape[0], 2):
        zero_frame[i][:][:] = frame[i][:][:]
    return zero_frame


def init_frame(frame, flag):
    '''
        flag = 0 偶数行
        flag = 1 奇数行
    '''
    if flag:
        for i in range(1, frame.shape[0] - 1, 2):
            frame[i][:][:] = (frame[i - 1][:][:] + frame[i + 1][:][:]) / 2
        if frame.shape[0] % 2 == 0:
            frame[frame.shape[0] - 1][:][:] = frame[frame.shape[0] - 2][:][:]
    else:
        frame[0][:][:] = frame[1][:][:]
        for i in range(2, frame.shape[0] - 1, 2):
            frame[i][:][:] = (frame[i - 1][:][:] + frame[i + 1][:][:]) / 2
        if frame.shape[0] % 2 == 1:
            frame[frame.shape[0] - 1][:][:] = frame[frame.shape[0] - 2][:][:]
    return frame


def toTensor(img):
    img = torch.from_numpy(img.transpose((2, 0, 1)))
    return img


def rgb2ycbcr(img_rgb):
    ## the range of img_rgb should be (0, 1)
    img_y = 0.257 * img_rgb[:, :, 0] + 0.504 * img_rgb[:, :, 1] + 0.098 * img_rgb[:, :, 2] + 16 / 255.0
    img_cb = -0.148 * img_rgb[:, :, 0] - 0.291 * img_rgb[:, :, 1] + 0.439 * img_rgb[:, :, 2] + 128 / 255.0
    img_cr = 0.439 * img_rgb[:, :, 0] - 0.368 * img_rgb[:, :, 1] - 0.071 * img_rgb[:, :, 2] + 128 / 255.0
    return img_y, img_cb, img_cr


def ycbcr2rgb(img_ycbcr):
    ## the range of img_ycbcr should be (0, 1)
    img_r = 1.164 * (img_ycbcr[:, :, 0] - 16 / 255.0) + 1.596 * (img_ycbcr[:, :, 2] - 128 / 255.0)
    img_g = 1.164 * (img_ycbcr[:, :, 0] - 16 / 255.0) - 0.392 * (img_ycbcr[:, :, 1] - 128 / 255.0) - 0.813 * (
            img_ycbcr[:, :, 2] - 128 / 255.0)
    img_b = 1.164 * (img_ycbcr[:, :, 0] - 16 / 255.0) + 2.017 * (img_ycbcr[:, :, 1] - 128 / 255.0)
    img_r = img_r[:, :, np.newaxis]
    img_g = img_g[:, :, np.newaxis]
    img_b = img_b[:, :, np.newaxis]
    img_rgb = np.concatenate((img_r, img_g, img_b), 2)
    return img_rgb


def rgb2y(img_rgb):
    ## the range of img_rgb should be (0, 1)
    image_y = 0.257 * img_rgb[:, :, 0] + 0.504 * img_rgb[:, :, 1] + 0.098 * img_rgb[:, :, 2] + 16 / 255.0
    return image_y


def L1_regularization(image):
    b, _, h, w = image.size()
    reg_x_1 = image[:, :, 0:h - 1, 0:w - 1] - image[:, :, 1:, 0:w - 1]
    reg_y_1 = image[:, :, 0:h - 1, 0:w - 1] - image[:, :, 0:h - 1, 1:]
    reg_L1 = torch.abs(reg_x_1) + torch.abs(reg_y_1)
    return torch.sum(reg_L1) / (b * (h - 1) * (w - 1))


def optical_flow(img1_path, img2_path):
    # prev_img = cv2.imread(img1_path, cv2.IMREAD_COLOR)
    # curr_img = cv2.imread(img2_path, cv2.IMREAD_COLOR)
    prev_img = img1_path
    curr_img = img2_path

    prev_gray = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)

    scale = 0.5
    levels = 1
    winsize = 15
    iterations = 20
    poly_n = 5
    poly_sigma = 1.1
    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray,
                                        flow=None, pyr_scale=scale, levels=levels, iterations=iterations,
                                        winsize=winsize, poly_n=poly_n, poly_sigma=poly_sigma, flags=0)

    return flow
