import numpy as np
import matplotlib.pyplot as plt
import cv2

def imgToPoints(img, n, m, dev, mask=[]):
    # get index randomly
    dis_img = cv2.distanceTransform(img, cv2.DIST_L2, 5)
    plt.imshow(dis_img)
    plt.show()
    ind = np.where(dis_img)
    weight = dis_img[ind] / np.sum(dis_img)
    cho = np.random.choice(np.arange(len(weight)), size=n, p=weight)
    samples = np.vstack(ind).T[cho].repeat(m, axis=0)
    
    # label
    labels = []
    if len(mask):
        labels = mask[samples[:, 0], samples[:, 1]]
        print(labels)

    # set random points center from those indexs
    dist = np.int8(np.random.normal(0, dev, [n * m, 2]))
    samples += dist
    samples[samples[:, 0] >= img.shape[0], 0] = img.shape[0] - 1
    samples[samples[:, 1] >= img.shape[1], 1] = img.shape[1] - 1
    samples[samples[:, 0] < 0, 0] = 0
    samples[samples[:, 1] < 0, 1] = 0

    return samples, labels

img = cv2.imread("image.jpg", 0)
n = 2000
m = 5
dev = 5
ret, th_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
mask = cv2.connectedComponents(th_img)[1]
samples, labels = imgToPoints(th_img, n, m, dev, mask)

new_img = np.zeros(img.shape[:2], dtype=np.uint8)
new_img[samples[:, 0], samples[:, 1]] = labels
plt.figure()
plt.imshow(new_img)
plt.show() 
