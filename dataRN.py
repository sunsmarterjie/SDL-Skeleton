import os
import numpy as np
import pandas as pd
import skimage.io as io
import matplotlib.pyplot as plt
import cv2


def thinImage(src, maxIterations=-1):
    assert len(src.shape) == 2, 'please binarify pictures'
    img_height, img_width = src.shape
    dst = src.copy()
    count = 0
    while True:
        count += 1
        if maxIterations != -1 and count > maxIterations:
            break
        mFlag = []
        for i in range(img_height):
            for j in range(img_width):
                p1 = dst[i, j]
                if p1 != 1:
                    continue
                p4 = 0 if j == img_width - 1 else dst[i, j + 1]
                p8 = 0 if j == 0 else dst[i, j - 1]
                p2 = 0 if i == 0 else dst[i - 1, j]
                p3 = 0 if i == 0 or j == img_width - 1 else dst[i - 1, j + 1]
                p9 = 0 if i == 0 or j == 0 else dst[i - 1, j - 1]
                p6 = 0 if i == img_height - 1 else dst[i + 1, j]
                p5 = 0 if i == img_height - 1 or j == img_width - 1 else dst[i + 1, j + 1]
                p7 = 0 if i == img_height - 1 or j == 0 else dst[i + 1, j - 1]
                if p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9 >= 2 and p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9 <= 6:
                    ap = 0
                    if p2 == 0 and p3 == 1:
                        ap += 1
                    if p3 == 0 and p4 == 1:
                        ap += 1
                    if p4 == 0 and p5 == 1:
                        ap += 1
                    if p5 == 0 and p6 == 1:
                        ap += 1
                    if p6 == 0 and p7 == 1:
                        ap += 1
                    if p7 == 0 and p8 == 1:
                        ap += 1
                    if p8 == 0 and p9 == 1:
                        ap += 1
                    if p9 == 0 and p2 == 1:
                        ap += 1
                    if ap == 1 and p2 * p4 * p6 == 0 and p4 * p6 * p8 == 0:
                        mFlag.append([i, j])
        for flag in mFlag:
            dst[flag[0], flag[1]] = 0
        if len(mFlag) == 0:
            break
        else:
            mFlag.clear()
        for i in range(img_height):
            for j in range(img_width):
                p1 = dst[i, j]
                if p1 != 1:
                    continue
                p4 = 0 if j == img_width - 1 else dst[i, j + 1]
                p8 = 0 if j == 0 else dst[i, j - 1]
                p2 = 0 if i == 0 else dst[i - 1, j]
                p3 = 0 if i == 0 or j == img_width - 1 else dst[i - 1, j + 1]
                p9 = 0 if i == 0 or j == 0 else dst[i - 1, j - 1]
                p6 = 0 if i == img_height - 1 else dst[i + 1, j]
                p5 = 0 if i == img_height - 1 or j == img_width - 1 else dst[i + 1, j + 1]
                p7 = 0 if i == img_height - 1 or j == 0 else dst[i + 1, j - 1]
                if p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9 >= 2 and p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9 <= 6:
                    ap = 0
                    if p2 == 0 and p3 == 1:
                        ap += 1
                    if p3 == 0 and p4 == 1:
                        ap += 1
                    if p4 == 0 and p5 == 1:
                        ap += 1
                    if p5 == 0 and p6 == 1:
                        ap += 1
                    if p6 == 0 and p7 == 1:
                        ap += 1
                    if p7 == 0 and p8 == 1:
                        ap += 1
                    if p8 == 0 and p9 == 1:
                        ap += 1
                    if p9 == 0 and p2 == 1:
                        ap += 1
                    if ap == 1 and p2 * p4 * p8 == 0 and p2 * p6 * p8 == 0:
                        mFlag.append([i, j])
        for flag in mFlag:
            dst[flag[0], flag[1]] = 0
        if len(mFlag) == 0:
            break
        else:
            mFlag.clear()
    return dst

rootDir = './SKLARGE_RN'
fileNames = './SKLARGE_RN/aug_data/train_pair.lst'
frame = pd.read_csv(fileNames, dtype=str, delimiter=' ', header=None)
f = open(os.path.join(rootDir, 'train_pairRN60_255_s_all.lst'), 'w')
for idx in range(len(frame)):
    inputName = os.path.join(rootDir, frame.iloc[idx, 0])
    targetName = os.path.join(rootDir, frame.iloc[idx, 1])
    inputImage = io.imread(inputName)
    targetImage = io.imread(targetName)
    targetImage[targetImage > 0] = 255

    K = 60000.0
    # K = 180000.0
    H, W = targetImage.shape[0], targetImage.shape[1]
    sy = np.sqrt(K * H / W) / float(H)
    sx = np.sqrt(K * W / H) / float(W)
    inputImage_RN = cv2.resize(inputImage, None, None, fx=sx, fy=sy, interpolation=cv2.INTER_LINEAR)
    targetImage_RN = cv2.resize(targetImage, None, None, fx=sx, fy=sy, interpolation=cv2.INTER_LINEAR)
    targetImage_RN[targetImage_RN > 0] = 255
    aa, bb = np.nonzero(targetImage_RN)
    cc = targetImage_RN[aa, bb]
    if (cc.min() != 255 or cc.max() != 255):
        print('min max error!')
        print('RN error')
        break

    targetImage_BN = targetImage_RN.copy()
    targetImage_BN[targetImage_BN == 255] = 1
    targetImage_TN = thinImage(targetImage_BN)
    targetImage_TN[targetImage_TN == 1] = 255
    aa, bb = np.nonzero(targetImage_TN)
    cc = targetImage_TN[aa, bb]
    if (cc.min() != 255 or cc.max() != 255):
        print('TN error')
        break

    inputDir_TN = frame.iloc[idx, 0].replace('scale', 'scaleRN60_')
    targetDir_TN = frame.iloc[idx, 1].replace('scale', 'scaleRN60_')
    inputName_TN = os.path.join(rootDir, inputDir_TN)
    targetName_TN = os.path.join(rootDir, targetDir_TN)
    inputsDir_TN = inputName_TN[:inputName_TN.find(inputDir_TN.split('/')[-1])]
    if not os.path.exists(inputsDir_TN):
        os.makedirs(inputsDir_TN)
    targetsDir_TN = targetName_TN[:targetName_TN.find(targetDir_TN.split('/')[-1])]
    if not os.path.exists(targetsDir_TN):
        os.makedirs(targetsDir_TN)
    io.imsave(inputName_TN, inputImage_RN)
    io.imsave(targetName_TN, targetImage_TN)
    f.write(inputDir_TN + ' ' + targetDir_TN + '\n')
    if idx % 12 == 0:
        print('idx:{} {} {}'.format(idx, inputDir_TN, targetDir_TN))

f.close()
