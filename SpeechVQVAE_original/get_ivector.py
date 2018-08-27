from hparams import Hyperparams as hp
from data_load import speaker2id


def get_ivec():
    spk_dict = {}
    f = open(hp.ivector_path,'r')
    x = f.readlines()
    for line in x:
        tokens = line.split(' ')
        del tokens[1]
        del tokens[1]
        del tokens[-1]
        speaker = tokens[0]
        del tokens[0]
        vector = []
        for token in tokens:
            vector.append(float(token))
        spk_dict[speaker2id(speaker)] = vector
    return spk_dict