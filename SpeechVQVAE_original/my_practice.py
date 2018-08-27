import utils
import os
from hparams import Hyperparams as hp
from utils import mu_law_decode
import data_load
import networks
import tensorflow as tf
import modules
import glob
from modules import conv1d
import numpy as np
import time
from train import Graph
from scipy.io.wavfile import write
'''
paths=glob.glob(hp.transformed_data_qts+'/*.npy')
paths2=glob.glob(hp.transformed_data_wav+'/*.npy')
'''
'''
for i in range(0,10):
 x=np.load(paths[i])
 y=np.load(paths2[i])
 #print(x.shape)
 #print(y.shape)
 x=tf.convert_to_tensor(x)
 print(x.shape)
 x=tf.pad(x,([0,hp.T],[0,0]),mode="CONSTANT")[:hp.T, :]
 print(x.shape)
'''
'''
qt,wav,speaker,batch_num=data_load.get_batch()
print(qt.shape)
qt2=tf.to_float(qt)
b=modules.conv1d(qt2)
print(b)
a=networks.encoder(qt2)
print(a)
z_q=networks.vq(a)
print(z_q)
B,t,D=z_q.get_shape().as_list()
print(B,t,D)
pad_val=int((hp.T-t)/2)
x=tf.pad(z_q,[[0,0],[pad_val,pad_val],[0,0]])
x=tf.layers.conv1d(x,filters=hp.D,kernel_size=hp.winsize,dilation_rate=64)
print(x)
'''
print