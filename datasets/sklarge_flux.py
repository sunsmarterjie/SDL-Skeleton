import numpy as np
import cv2
import random
import pandas as pd
import skimage.io as io
import torch
import os
from torch.utils.data import Dataset, DataLoader


class DataLayer(Dataset):

    def __init__(self, data_dir):
        # data layer config
        self.data_dir = data_dir
        self.mean = np.array([103.939, 116.779, 123.68])

        # read filename list for each dataset here
        self.fnLst = open(self.data_dir + 'SKLARGE/train_pair_255_s_all.lst').readlines()
        # randomization: seed and pick

    def __len__(self):
        return len(self.fnLst)

    def __getitem__(self, idx):
        # load image, flux and dilmask

        self.image, self.flux, self.dilmask = self.loadsklarge(self.fnLst[idx].split()[0],
                                                               self.fnLst[idx].split()[1])
        return self.image, self.flux, self.dilmask

    def loadsklarge(self, imgidx, gtidx):
        # load image and skeleton
        image = cv2.imread('{}/SKLARGE/{}'.format(self.data_dir, imgidx), 1)
        skeleton = cv2.imread('{}/SKLARGE/{}'.format(self.data_dir, gtidx), 0)
        skeleton = (skeleton > 0).astype(np.uint8)
        image = image.astype(np.float32)
        image -= self.mean
        image = image.transpose((2, 0, 1))

        # compute flux and dilmask
        kernel = np.ones((15, 15), np.uint8)
        dilmask = cv2.dilate(skeleton, kernel)
        rev = 1 - skeleton
        height = rev.shape[0]
        width = rev.shape[1]
        rev = (rev > 0).astype(np.uint8)
        dst, labels = cv2.distanceTransformWithLabels(rev, cv2.DIST_L2, cv2.DIST_MASK_PRECISE,
                                                      labelType=cv2.DIST_LABEL_PIXEL)

        index = np.copy(labels)
        index[rev > 0] = 0
        place = np.argwhere(index > 0)

        nearCord = place[labels - 1, :]
        x = nearCord[:, :, 0]
        y = nearCord[:, :, 1]
        nearPixel = np.zeros((2, height, width))
        nearPixel[0, :, :] = x
        nearPixel[1, :, :] = y
        grid = np.indices(rev.shape)
        grid = grid.astype(float)
        diff = grid - nearPixel

        dist = np.sqrt(np.sum(diff ** 2, axis=0))

        direction = np.zeros((2, height, width), dtype=np.float32)
        direction[0, rev > 0] = np.divide(diff[0, rev > 0], dist[rev > 0])
        direction[1, rev > 0] = np.divide(diff[1, rev > 0], dist[rev > 0])

        direction[0] = direction[0] * (dilmask > 0)
        direction[1] = direction[1] * (dilmask > 0)

        flux = -1 * np.stack((direction[0], direction[1]))

        dilmask = (dilmask > 0).astype(np.float32)
        dilmask = dilmask[np.newaxis, ...]

        return image, flux, dilmask


class TestDataLayer(Dataset):
    def __init__(self, rootDir, transform=None):
        self.rootDir = rootDir
        self.transform = transform
        self.frame = sorted(os.listdir(self.rootDir))
        print(type(self.frame))

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        fname = self.frame[idx]
        inputName = os.path.join(self.rootDir, fname)

        inputImage = io.imread(inputName)[:, :, ::-1]
        inputImage = inputImage.astype(np.float32)
        inputImage -= np.array([104.00699, 116.66877, 122.67892])
        inputImage = inputImage.transpose((2, 0, 1))
        inputImage = torch.Tensor(inputImage)
        return inputImage, fname

