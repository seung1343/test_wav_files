# -*- coding: utf-8 -*-
'''
By kyubyong park. kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/vq-vae
'''

from hparams import Hyperparams as hp
import tensorflow as tf
import os
import glob
import numpy as np

def speaker2id(speaker):
    func = {speaker:id for id, speaker in enumerate(hp.speakers)}
    return func.get(speaker, None)

def id2speaker(id):
    func = {id:speaker for id, speaker in enumerate(hp.speakers)}
    return func.get(id, None)

def load_data(mode="train"):
    '''Loads data
    Args:
      mode: "train" or "eval".

    Returns:
      files: A list of sound file paths.
      speaker_ids: A list of speaker ids.
    '''
    if mode=="train":
        # wavs = glob.glob(hp.transformed_data_wav+"/*.npy")
        # wavs = glob.glob('vctk/wavs/*.npy')
        qts = glob.glob(hp.transformed_data_qts+"/*.npy")
        speakers = np.array([speaker2id(os.path.basename(qt)[:4]) for qt in qts], np.int32)
        data_num = len(qts)
        # wavs = np.expand_dims(wavs, axis=1)
        qts = np.expand_dims(qts, axis=1)
        speakers = np.expand_dims(speakers, axis=1)
        total = np.concatenate((qts, speakers), axis=1)  # Concatenated
        np.random.shuffle(total)
        test_data = total[int(data_num*0.9):,:]
        train_data = total[:int(data_num*0.9),:]
        np.save('/home/shyou/Desktop/test_data',test_data)
        np.save('/home/shyou/Desktop/train_data',train_data)
        return total, data_num
    else: # test. two samples.
        files = ['/data/private/speech/vctk/qts/'+line.split("|")[0].strip() + ".npy" for line in hp.test_data.splitlines()]
        speaker_ids = [int(line.split("|")[1]) for line in hp.test_data.splitlines()]
        return files, speaker_ids

# load_data()
def get_batch():
    """Loads training data and put them in queues"""
    with tf.device('/cpu:0'):
        # Load data
        wavs, qts, speakers = load_data() # list
        # Calc total batch count
        num_batch = len(wavs) // hp.batch_size
        wavs = np.expand_dims(wavs, axis=1)
        qts = np.expand_dims(qts, axis=1)
        speakers = np.expand_dims(speakers,axis=1)
        total = np.concatenate((wavs,qts,speakers),axis=1)#Concatenated
        np.random.shuffle(total)
        # Create Queues
        wav, qt, speaker = tf.train.slice_input_producer([wavs, qts, speakers], shuffle=True)
        # Parse
        wav, = tf.py_func(lambda x: np.load(x, allow_pickle=False), [wav], [tf.float32])  # (None, 1)
        qt, = tf.py_func(lambda x: np.load(x, allow_pickle=False), [qt], [tf.int32])  # (None, 1)
        # Cut off
        qt = tf.pad(qt, ([0, hp.T], [0, 0]), mode="CONSTANT")[:hp.T, :]

        # Add shape information
        wav.set_shape((None,))
        qt.set_shape((hp.T, 1))
        speaker.set_shape(())

        # Batching
        qts, wavs, speakers = tf.train.batch(tensors=[qt, wav, speaker],
                                             batch_size=hp.batch_size,
                                             shapes=([hp.T, 1], [None,], []),
                                             num_threads=32,
                                             dynamic_pad=True)
        return qts, wavs, speakers, num_batch
'''
#Code Test
x, y = load_data()
print(np.shape(x))
z=x[4:][:]
x=x[:4][:]
print(np.shape(x))
print(np.shape(z))
print(np.shape(np.concatenate((x,z))))
qts=[]
speakers=[]
for i in range(0,4):
    qt = np.load(x[i][1])
    qt = tf.pad(qt, ([0, hp.T], [0, 0]), mode="CONSTANT")[:hp.T, :]
    speaker=int(x[i][2])
    qts.append(qt)
    speakers.append(speaker)
qts = tf.convert_to_tensor(qts, dtype=tf.int32)
speakers = tf.convert_to_tensor(speakers, dtype=tf.int32)
print(qts)
print(speakers)
'''
load_data()
