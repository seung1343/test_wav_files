import tensorflow as tf
from hparams import Hyperparams as hp
from modules import conv1d
from networks import encoder


def decoder(decoder_inputs, speaker_emb, z_q, is_training=True):
    '''
    Wavenet decoder.
    Args:
      decoder_inputs: raw wav form [B, T, 1].
      speaker_emb: [B, len(speaker)]. One-hot. Global condition.
      -->speaker_emb:[B,ivec_size] speaker ivector
      z_q: [B, T', D]. Local condition.
      is_training: tell model whether it is in training mode
    Return:
      output: [B,T-receptive_field+1,Q]
    '''
    with tf.variable_scope("decoder"):
        #multiples = hp.stride**hp.encoder_layers
        receptive_field=hp.dilations[-1]*hp.size
        output_width = decoder_inputs.get_shape().as_list()[1]-receptive_field+1
        # raw wav form(B,T,1) to (B,T,Q)
        #decoder_inputs=tf.reshape(decoder_inputs,[hp.batch_size,hp.T,hp.Q])
        # local condition (B,T',D) to (B,T,Q)
        '''
        B,t,D = z_q.get_shape().as_list()
        z_out = tf.reshape(z_q,[1,-1,D])
        for i in range(0,hp.encoder_layers):
            z_out = tf.concat((z_out,z_out),axis=0)
        z_out = tf.transpose(z_out,perm=[1,0,2])
        z_out = tf.reshape(z_out,[B,multiples*t,D])
        '''
        reuse = None
        if is_training is True:
            z_out = transposed_conv(z_q)
        else:
            reuse = tf.AUTO_REUSE
            z_out = z_q
        #z_out is now (B,T,Q)
        #global conditioning (B,L) to (B,1,Q)
        speaker_emb = tf.expand_dims(speaker_emb,1) #(B,1,L)
        gc = speaker_emb
        outputs=[]
        for index,dilation in enumerate(hp.dilations):
            out,decoder_inputs = dilated_convolution(decoder_inputs,
                                                     z_out,
                                                     gc,
                                                     hp.size,
                                                     dilation,
                                                     output_width,
                                                     index,
                                                     dilation,
                                                     reuse)
            outputs.append(out)
        #postnet
        total=sum(outputs)
        transformed1=tf.nn.relu(total)
        conv1=conv1d(transformed1,hp.Q,scope='transformed1',onebyone=True, reuse=reuse)
        transformed2=tf.nn.relu(conv1)
        conv2=conv1d(transformed2,hp.Q,scope='transformed2',onebyone=True, reuse=reuse)

    return conv2


def dilated_convolution(input,
                        Local_condition,
                        Global_condition,
                        filter_width,
                        dilation_rate,
                        output_width,
                        index,
                        dilation,
                        reuse):
        #dilated convolution
        conv=conv1d(input, hp.Q*2, filter_width, rate=dilation_rate, padding="causal", scope='conv_{}_{}'.format(index, dilation), reuse=reuse)
        local_cond=conv1d(Local_condition,hp.Q*2, 1,rate=1, padding="SAME", scope='local_cond_{}_{}'.format(index, dilation), reuse=reuse)
        local_cond=local_cond[:,hp.size**(index+1)-1:,:]
        global_cond=conv1d(Global_condition, hp.Q*2, 1, rate=1, padding="SAME", scope='global_cond_{}_{}'.format(index, dilation), reuse=reuse)
        conv_filter,conv_gate = tf.split(conv,2,-1)
        local_cond_filter,local_cond_gate = tf.split(local_cond,2,-1)
        global_cond_filter,global_cond_gate = tf.split(global_cond,2,-1)
        conv_filter=conv_filter+local_cond_filter+global_cond_filter #broadcast
        conv_gate=conv_gate+local_cond_gate+global_cond_gate #broadcast
        out=tf.tanh(conv_filter)+tf.sigmoid(conv_gate)
        transformed=conv1d(out,filters=hp.Q, padding="SAME", scope='transformed_{}_{}'.format(index, dilation),onebyone=True, reuse=reuse)
        _,x,__=out.get_shape().as_list()
        skip_cut=x-output_width
        out_skip=tf.slice(out,[0,skip_cut,0],[-1,-1,-1],name='out_skip')
        out_skip=tf.reshape(out_skip,[hp.batch_size,output_width,hp.Q])
        skip_contribution=conv1d(out_skip,filters=hp.Q,padding="SAME",scope='skip_contribution_{}_{}'.format(index, dilation),onebyone=True, reuse=reuse)
        transformed_cut=transformed.get_shape().as_list()[1]
        input_cut=input.get_shape().as_list()[1]-transformed_cut
        input_batch=tf.slice(input,[0,input_cut,0],[-1,-1,-1])
        input_batch=tf.reshape(input_batch,[hp.batch_size,transformed_cut,hp.Q])

        return skip_contribution,input_batch+transformed


def transposed_conv(z_q):
    filter_size = hp.stride
    for i in range(0, hp.encoder_layers):
        z_q = tf.layers.conv1d(z_q,
                               kernel_size=filter_size,
                               filters=hp.D * 2,
                               strides=1,
                               padding='SAME',
                               activation=tf.nn.relu,
                               name='transposed_conv_{}'.format(i),
                               reuse=tf.AUTO_REUSE)
        z_q = tf.reshape(z_q, (hp.batch_size, -1, hp.D))
    return z_q


def generate(inputs, speaker_emb, z_q, ith, is_first=False):
    '''
    :param inputs: (B,T,1)
    :param speaker_emb: i_vector embedding(B,hp.ivec_size)
    :return: wav form
    '''
    receptive_field = hp.dilations[-1] * hp.size
    speaker_emb = tf.expand_dims(speaker_emb, 1)  # (B,1,L)
    gc = speaker_emb
    if is_first:
        z_q = transposed_conv(z_q)
    _z_q = z_q[:, ith:receptive_field+ith, :]
    x = decoder(inputs,gc,_z_q,is_training=False)
    out = tf.multinomial(tf.squeeze(x,1), num_samples=1, output_dtype=tf.int32)  # (B,1)
    output = tf.concat((output,tf.expand_dims(out,1)),axis=1)
    out = tf.one_hot(out,hp.Q,dtype=tf.float32)
    inputs = tf.concat((inputs[:, 1:, :], out),axis=1)
    return output
