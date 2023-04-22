import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import database

datapath_train = 'data/real_data/'
n_samples_train = 50000
database = database.database(datapath_train, n_samples_train)
TPSF_ARRAY_batch, distribution_true_batch = database.get_special_batch(20)
for i in range(len(distribution_true_batch)):
    plt.figure()
    plt.imshow(distribution_true_batch[i, 0])
    plt.show()