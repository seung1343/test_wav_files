# -*- coding: utf-8 -*-
'''
By kyubyong park. kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/vq-vae
'''

from __future__ import print_function

from hparams import Hyperparams as hp
from utils import get_wav
import os
import glob
import numpy as np


def f(fpath):
    w, q = get_wav(fpath)
    fname = os.path.basename(fpath).replace('wav', 'npy')
    np.save(hp.transformed_data_wav+"/{}".format(fname), w)
    np.save(hp.transformed_data_qts+"/{}".format(fname), q)

fpaths = glob.glob(hp.data)
#print(fpaths)
total_files = len(fpaths)
completedfile=0
for fpath in fpaths:
    f(fpath)
    completedfile+=1
    print(completedfile/total_files)