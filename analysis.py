import numpy as np
import cv2
import os
from matplotlib import pyplot as plt


def effective_lifetime(lifetime, yield_value, num):
    lifetime_list = list()
    # lifetime[yield_value <= 0.1 * np.amax(yield_value)] = 0
    tmp = lifetime * 255
    tmp[tmp > 255] = 255
    gray = np.uint8(tmp)
    ret, thresh = cv2.threshold(gray, 0, 10, cv2.THRESH_OTSU)
    ret, markers, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=4)
    # plt.figure()
    # hold = plt.imshow(yield_value)
    # plt.colorbar(hold)
    # plt.show()
    plt.figure()
    hold = plt.imshow(lifetime)
    plt.colorbar(hold)
    for i in range(1, len(centroids)):
        tmp_yield = yield_value[markers == i]
        tmp_life = lifetime[markers == i]
        tmp_yield[tmp_life <= 0.2 * np.amax(tmp_life)] = 0
        tmp = tmp_life * tmp_yield / np.sum(tmp_yield)  # * yield_value[markers == i] / sum(yield_value[markers == i])
        # tmp = tmp[tmp >= 0.1 * np.max(tmp)]
        lifetime_list.append(np.sum(tmp))
        if np.sum(markers == i) <= 2:
            continue
        plt.text(centroids[i, 0] + 5, centroids[i, 1] + 5, '%.2f' % lifetime_list[i - 1], ha='center', va='bottom',
                 fontsize=12, color="r")
    plt.savefig(os.getcwd() + '/results/' + str(num) + 'lifetime.jpg')
    plt.show()
    return lifetime_list


def yita_recon_quality(yita, num):
    value_list = list()
    # yita = yita
    tmp = yita / (np.amax(yita) + 0.0001) * 255
    tmp[tmp > 255] = 255
    gray = np.uint8(tmp)
    # ret, thresh = cv2.threshold(gray, 0, 2, cv2.THRESH_OTSU)
    thresh = gray * 0
    thresh[gray >= 50] = 1
    ret, markers, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)
    plt.figure()
    hold = plt.imshow(yita)
    plt.colorbar(hold)
    for i in range(1, len(centroids)):
        tmp = yita[markers == i]
        tmp = tmp[tmp >= 0.1 * np.max(tmp)]
        if np.sum(markers == i) <= 2:
            tmp = 0
        value_list.append(np.sum(tmp))
    if len(value_list) >= 1:
        value_list = value_list / np.amax(value_list)
    for i in range(1, len(centroids)):
        if np.sum(markers == i) <= 2:
            continue
        plt.text(centroids[i, 0] + 5, centroids[i, 1] + 5, '%.2f' % value_list[i - 1] + '\n[%.1f,' % centroids[i, 0]
                 + '%.1f]' % centroids[i, 1],
                 ha='center', va='bottom', fontsize=12, color="w")
    plt.text(30, -15, num, ha='center', va='bottom', fontsize=15, color="k")
    plt.savefig(os.getcwd() + '/results/' + str(num) + 'yield.jpg')
    plt.show()
    return value_list
