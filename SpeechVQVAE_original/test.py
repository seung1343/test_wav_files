import tensorflow as tf
import os
import numpy as np
from train import Graph
from get_ivector import get_ivec
from data_load import speaker2id
from Wavenet_Decoder import decoder
from hparams import Hyperparams as hp
from tqdm import tqdm
from scipy.io.wavfile import write
from utils import mu_law_decode

g = Graph(mode='test')
sess = tf.Session(config=tf.ConfigProto())
saver = tf.train.Saver()
ckpt = tf.train.latest_checkpoint(hp.logdir)
saver.restore(sess, ckpt)
print("check_point restored")
# ivecs = get_ivec()
test_data_paths = np.load('/home/shyou/Desktop/test_data.npy')
# test_data_path = 'F:/data/private/speech/vctk/qts/p225_001.npy'
# target_speaker_ivec = ivecs[speaker2id(target_speaker)]
receptive_field = hp.dilations[-1]*hp.size
speaker_emb = tf.placeholder(dtype=tf.float32,shape=(hp.batch_size,len(hp.speakers)))
_input = tf.placeholder(dtype=tf.float32,shape=(hp.batch_size, receptive_field,hp.Q))
_z_q = tf.placeholder(dtype=tf.float32,shape=(hp.batch_size, receptive_field, hp.D))
x = decoder(_input,speaker_emb,_z_q,is_training=False)
out = tf.multinomial(tf.squeeze(x,1), num_samples=1,output_dtype=tf.int32)
for j in range(0,len(test_data_paths)):
  test_qt = np.load(test_data_paths[j][0])
  test_speaker = test_data_paths[j][1]
  x = np.pad(test_qt, ([0, hp.T], [0, 0]), mode="constant", constant_values=0)[:hp.T, :]
  x = np.expand_dims(x,0)
  test_speaker = np.expand_dims(test_speaker,0)
# target_speaker_ivec = np.expand_dims(target_speaker_ivec,0)
  z_q, x,_speaker_emb = sess.run([g.z_q, g.encoder_inputs, g.speakers],feed_dict={g.x:x, g.speaker_id:test_speaker}) #shape = (B,T,K)
  output = np.squeeze(x)
  inputs = x[:, :receptive_field, :]  # (B,r,Q)
# decode and get multinomial distribuition


  for i in tqdm(range(hp.T-receptive_field-1),total=hp.T-receptive_field,ncols=70,leave=False,unit='step'):
    new_z_q = z_q[:, i:receptive_field+i, :]
    out_node = sess.run(out,feed_dict={_input: inputs, _z_q: new_z_q,speaker_emb: _speaker_emb})
    ans = np.squeeze(out_node)
    output[receptive_field+i+1] = ans
    out_one_hot = np.zeros((hp.batch_size,1,hp.Q))
    out_one_hot[0,0,ans]=1
    inputs = np.concatenate((inputs[:, 1:, :], out_one_hot), axis=1)

  audio = mu_law_decode(output)
  write(os.path.join(hp.sampledir, 'test{}.wav'.format(j)), hp.sr, audio)
