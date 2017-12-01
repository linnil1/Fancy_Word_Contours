from MLAnalysis import decision_map
from DataToPoints import imgToPoints
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from skimage import measure
import numpy as np
import cv2
import os
import argparse

def getCmap(color):
    two = np.repeat(np.array([[color]]), 256, axis=0)
    three = np.c_[two,np.arange(256), np.arange(256)]
    rgb = cv2.cvtColor(np.uint8([three]), cv2.COLOR_HSV2RGB)[0]
    return ListedColormap(rgb / 255), rgb[255] / 255
    # mpl.colorbar.ColorbarBase(plt.gca(),cmap=cmap)

def checkPositive(value):
    v = int(value)
    if v <= 0:
         raise argparse.ArgumentTypeError("%s is an invalid positive int value" % value)
    return v

# parse data
parser = argparse.ArgumentParser(description='Using SVM to redraw you image.')
parser.add_argument('path', type=str, help="Path of image",
                    default="image.jpg")
parser.add_argument('-n', type=checkPositive, default=2000,
                    help="Number of points in image")
parser.add_argument('-m', type=checkPositive, default=5,
                    help="How many points that beside real points we choose from image")
parser.add_argument('-d', '--dev', type=checkPositive, default=5, 
                    help="The deviation of points that beside real points we choose from image")
parser.add_argument('-r', '--reso', type=checkPositive, default=5,
                    help="Resolution of image (size = real_size / reso)")
parser.add_argument('-l', '--level', type=checkPositive, default=5,
                    help="How many contour level you want to plot")
args = parser.parse_args()
print("Add points to image")

# image data
name = args.path
reso = args.reso
num_cont = args.level
img = cv2.imread(name, 0)
dpi = 100
want_save = True # debug use

# main process
ret, th_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
mask = cv2.connectedComponents(th_img)[1]
samples, labels = imgToPoints(th_img, args.n, args.m, args.dev, mask)
print("Training")
label_map, value_map, xy = decision_map(th_img, samples, labels, reso=reso, name=name)

# orignial
plt.subplot(221)
plt.imshow(th_img)

# draw value
plt.subplot(222)
# blank = np.zeros([*img.shape, 3], dtype=np.float)
# plt.contour(xy[1].T, xy[0].T, value_map.reshape(xy[0].shape).T, linewidth=2, alpha=0.5)
# plt.imshow(blank)
plt.imshow(value_map.reshape(xy[0].shape).T)

# draw label
plt.subplot(223)
plt.imshow(label_map.reshape(xy[0].shape).T)

# draw map
plt.subplot(224)

if want_save:
    plt.figure(figsize=(img.shape[1] / dpi, img.shape[0] / dpi), dpi=dpi, frameon=False)
    plt.axis('off')
blank = np.zeros([*img.shape, 3], dtype=np.float)
blank[:,:,0] = 0.1
num_labels = np.sort(np.unique(labels))

# find and draw contour
for n in num_labels:
    cmap, color= getCmap(180 * n / len(num_labels))
    tmp_value = np.ma.array(value_map, mask=(label_map!=n), fill_value=0).filled()
    tmp_value = tmp_value.reshape(xy[0].shape)

    label_value = np.linspace(np.min(tmp_value), 1, num_cont)
    color_value = cmap.colors[np.uint(np.linspace(0, 255, num_cont))]
    for i in range(num_cont):
        contours = measure.find_contours(tmp_value, label_value[i])
        for contour in contours:
            contour = np.uint(contour * reso) 
            plt.plot(contour[:, 0], contour[:, 1], color=color_value[i])
            if i in [0, (num_cont // 2), num_cont - 1]:
                cv2.drawContours(blank, [contour[:,np.newaxis,:]], -1, color_value[i], -1)

# draw points
for n in num_labels:
    if not n:
        continue
    cmap, color= getCmap(180 * n / len(num_labels))
    filt_xy = samples[labels == n, :]
    blank[np.uint(filt_xy[::reso, 0]), np.uint(filt_xy[::reso, 1])] = color
plt.imshow(blank)

# save and plot it
if want_save:
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(name + "_result.jpg")
    print("save it at " + name + "_result.jpg")
plt.show() 
print("You can clean it by `rm " + name + "_*`")
