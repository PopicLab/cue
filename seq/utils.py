from bitarray import bitarray
import seq.io as io
import random
import numpy as np

def seq_to_num(seq):
    base2num = {'A': bitarray('00'), 'C': bitarray('01'), 'G': bitarray('10'), 'T': bitarray('11')}
    seq_bits = bitarray()
    seq_bits.encode(base2num, seq[:-2])
    return int(seq_bits.to01(), 2)

def partition_chrs(chr_names, fai_fname, n_chunks):
    if chr_names is None:
        chr_index = io.load_faidx(fai_fname, all=True)
        chr_names = list(chr_index.chr_names())
    random.shuffle(chr_names)
    return np.array_split(np.array(list(chr_names)), n_chunks), chr_names
