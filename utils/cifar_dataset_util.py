
# coding: utf-8

# ## Download CIFAR-10 dataset from url

# In[25]:

# helper methods to download CIFAR-10 provided
# by toronto.edu dataset & decompress it 
# --------------------------------------------

import urllib
import os
import tarfile
import time_util
import cPickle as pickle
import numpy as np

# download CIFAR-10 file as tar.gz file
# param: url url of the dataset
# param: file_name 
def download_cifar10(url, file_name):
    print "star downloading CIFAR-10..tar.gz"
    urllib.urlretrieve(url, file_name)

# Get the downloaded file and decompress it
# param: path_of_file is the path of tar.gz file
def decompress_save_cifar10(path_of_file, path_of_dataset_repo):
    print "star decompressing CIFAR-10..tar.gz"
    tar_gz_cifar = tarfile.open(path_of_file)
    tar_gz_cifar.extractall(path_of_dataset_repo)
    tar_gz_cifar.close() 


# In[9]:
"""
# path & name of the file
url = "http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
file_name = "cifar-10-python.tar.gz"
path_of_dataset_repo = os.path.join("../", "datasets")
path_of_file = os.path.join("../", "datasets", file_name)


# In[ ]:

# download..
_time = time_util.time_function(download_cifar10, url, path_of_file)
print 'download_cifar10 took %f seconds' % _time


# In[10]:

# decompress..
_time = time_util.time_function(decompress_save_cifar10, path_of_file, path_of_dataset_repo)
print 'decompress_save_cifar10 took %f seconds' % _time
"""

# In[33]:

# load one batch file of CIFAR by name
def load_cifar10_batch(batch_name):
    with open(batch_name, 'rb') as f:
        datadictionary = pickle.load(f)
        images = datadictionary["data"]
        labels = datadictionary["labels"]
        
        images = images.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
        labels = np.array(labels)
        return images, labels


# In[45]:

# load the entire CIFAR-10 dataset
def load_all_cifar10(path_name):
    imgs = []
    lbls = []
    for f in [x for x in os.listdir(path_name) if x.startswith("data_batch_")]:
        X, Y = load_cifar10_batch(os.path.join(path_name, f))
        imgs.append(X)
        lbls.append(Y)
    Xtr = np.concatenate(imgs)
    Ytr = np.concatenate(lbls)
    del X, Y
    Xte, Yte = load_cifar10_batch(os.path.join(path_name, 'test_batch'))
    
    return Xtr, Ytr, Xte, Yte

