import numpy as np
from sklearn import svm
from sklearn.externals import joblib
import os

def decision_map(th_img, samples, labels, reso=1, name="tmp"):
    # ser learning parameter
    if not os.path.isfile(name + "_ml.pkl"):
        models = svm.SVC(kernel="rbf", gamma=0.05, C=0.25)
        clf = models.fit(samples, labels)
        joblib.dump(clf, name + "_ml.pkl")
    else:
        clf = joblib.load(name + "_ml.pkl")

    # perdict map
    xx, yy = np.meshgrid(np.arange(0, th_img.shape[0], reso), np.arange(0, th_img.shape[1], reso))

    if not os.path.isfile(name + "_value_map.npy"):
        dec = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        label_map = np.argmax(dec, axis=1)
        value_map = dec[np.arange(len(label_map)), label_map]
        value_map = value_map - np.min(value_map)
        value_map = value_map / np.max(value_map)
        np.save(name + "_value_map", value_map)
        np.save(name + "_label_map", label_map)
    else:
        value_map = np.load(name + "_value_map.npy")
        label_map = np.load(name + "_label_map.npy")

    return label_map, value_map, [xx, yy]
