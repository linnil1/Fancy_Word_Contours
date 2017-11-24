import numpy as np
import cv2

def imgToPoints(img, n, m, dev, mask=[]):
    # get index randomly
    dis_img = cv2.distanceTransform(img, cv2.DIST_L2, 5)
    ind = np.where(dis_img)
    weight = dis_img[ind] / np.sum(dis_img)
    cho = np.random.choice(np.arange(len(weight)), size=n, p=weight)
    samples = np.vstack(ind).T[cho].repeat(m, axis=0)
    
    # label
    labels = []
    if len(mask):
        labels = mask[samples[:, 0], samples[:, 1]]

    # set random points center from those indexs
    dist = np.int8(np.random.normal(0, dev, [n * m, 2]))
    samples += dist
    samples[samples[:, 0] >= img.shape[0], 0] = img.shape[0] - 1
    samples[samples[:, 1] >= img.shape[1], 1] = img.shape[1] - 1
    samples[samples[:, 0] < 0, 0] = 0
    samples[samples[:, 1] < 0, 1] = 0

    # background = label 0
    zeros = np.vstack(np.where(img == 0)).T
    zeros_cho = zeros[np.random.choice(np.arange(zeros.shape[0]), size=n), :]
    samples = np.vstack([samples, zeros_cho])
    labels = np.hstack([labels, np.zeros(zeros_cho.shape[0], dtype=np.uint8)])

    return samples, labels
