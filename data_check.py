import scipy.io as sio
import numpy as np
import os
import matplotlib.pyplot as plt

counter = 1
for counter in range(20):
    dataset_idxs = np.random.permutation(500000) + 1
    matfilename = '/data/cjj/data/generated_sets8/' + str(int(dataset_idxs[counter])) + '.mat'
    f = sio.loadmat(matfilename)
    # TPSF_ARRAY = f['TPSF_ARRAY'][()]
    tpsftrue = f['tpsftrue'][()]
    tpsftimeless = f['tpsftimeless'][()].reshape(45)
    tpsf_logtail = f['tpsf_logtail'][()].reshape(54)
    weight_logtail = f['weight_logtail'][()].reshape(54)
    # distribution_true_slice = f['distribution_true'][()]
    # lifetime_true_slice = f['lifetime_true'][()]
    # direct store to improve training efficiency
    distribution_true = f['distribution_true'][()]
    lifetime_true = f['lifetime_true'][()]
    TPSF_ARRAY_batch = f['TPSF_ARRAY_batch'][()]

    img_lifetime = lifetime_true[2]
    plt.imshow(img_lifetime)
    plt.show()