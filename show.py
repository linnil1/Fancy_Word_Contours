from MLAnalysis import decision_map
from DataToPoints import imgToPoints
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from skimage import measure
import numpy as np
import cv2
import os

def getCmap(color):
    two = np.repeat(np.array([[color]]), 256, axis=0)
    three = np.c_[two,np.arange(256), np.arange(256)]
    rgb = cv2.cvtColor(np.uint8([three]), cv2.COLOR_HSV2RGB)[0]
    return ListedColormap(rgb / 255), rgb[255] / 255
    # mpl.colorbar.ColorbarBase(plt.gca(),cmap=cmap)


# image data
name = "image.jpg"
img = cv2.imread(name, 0)
n = 2000
m = 5
dev = 5
reso = 5
num_cont = 10
want_save = True
ret, th_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
mask = cv2.connectedComponents(th_img)[1]
samples, labels = imgToPoints(th_img, n, m, dev, mask)
label_map, value_map, xy = decision_map(th_img, samples, labels, reso=reso, name=name)

# orignial
plt.subplot(221)
plt.imshow(img)

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
    plt.figure()
    plt.axis('off')
blank = np.zeros([*img.shape, 3], dtype=np.float)
blank[:,:,0] = 0.1
num_labels = np.sort(np.unique(labels))

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

if want_save:
    plt.savefig(name + "_result.jpg", bbox_inches='tight')
plt.show() 
