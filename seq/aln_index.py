# MIT License
#
# Copyright (c) 2022 Victoria Popic
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


from seq import io
from seq import utils
import numpy as np
import pickle
import os.path
from cachetools import cached, LRUCache
from cachetools.keys import hashkey
from img import constants
from img.constants import SVSignals
from collections import defaultdict
import operator
import logging


class AlnIndex:
    def __init__(self, chr_index, bin_size, chr_name, signal_set_type):
        self.bin_size = bin_size
        self.tid = chr_index.tid(chr_name)
        self.chr = chr_index.chr_from_name(chr_name)
        self.signal_set = constants.SV_SIGNALS_BY_TYPE[signal_set_type]
        self.bins = defaultdict(dict)
        n_bins = self.chr.len // bin_size + 2
        for signal in self.signal_set:
            if signal in constants.SV_SIGNAL_SCALAR:
                self.bins[signal][self.tid] = [0] * n_bins
            else:
                self.bins[signal][self.tid] = [set() for _ in range(n_bins)]

    #####################
    #   Index editing   #
    #####################

    def add(self, read, min_mapq_dict):
        for signal in self.signal_set:
            self.add_by_signal(read, signal, min_mapq_dict)

    def add_by_signal(self, read, signal, min_mapq_dict, min_clipped_len=10):
        if signal not in self.bins or \
           read.mapping_quality < min_mapq_dict[signal] or \
           (signal in constants.SV_SIGNAL_PAIRED and (read.mate_is_unmapped or
                                                      read.next_reference_name != read.reference_name)):
            return

        # filter out singleton alignments with low quality regardless of config 
        if (read.mate_is_unmapped or read.next_reference_name != read.reference_name) and read.mapping_quality < 20:
            return 

        if read.reference_end is not None: 
            bin_id = self.get_bin_id(read.pos + (read.reference_end - read.reference_start) // 2)
        else:
            bin_pos = min(read.pos + len(read.seq) // 2, self.chr.len - 1)
            bin_id = self.get_bin_id(bin_pos)
        assert len(self.bins[signal][self.tid]) > bin_id, "%d %d %d" % \
                                                          (read.pos, bin_id, len(self.bins[signal][self.tid]))

        if signal in constants.SV_SIGNAL_SCALAR:
            if signal == SVSignals.RD_CLIPPED:
                if (read.cigartuples[0][0] in [4, 5] and read.cigartuples[0][1] > min_clipped_len) or \
                   (read.cigartuples[-1][0] in [4, 5] and read.cigartuples[-1][1] > min_clipped_len):
                    # soft (op 4) or hard clipped (op 5)
                    # TODO(viq): expose min clipped length param
                    self.bins[signal][self.tid][bin_id] += 1
            else:
                self.bins[signal][self.tid][bin_id] += 1
        else:
            if signal == SVSignals.SM and read.has_tag('BX'):
                barcode = read.get_tag('BX')
                if isinstance(barcode, str):
                    barcode = utils.seq_to_num(barcode)
                self.bins[SVSignals.SM][self.tid][bin_id].add(barcode)
            else:
                if signal in constants.SV_SIGNAL_PAIRED:
                    rp_type = constants.get_read_pair_type(read)
                if (signal == SVSignals.LLRR and rp_type != constants.SV_SIGNAL_RP_TYPE.LLRR) or \
                   (signal == SVSignals.RL and rp_type != constants.SV_SIGNAL_RP_TYPE.RL):
                    return
                self.bins[signal][self.tid][bin_id].add(read.qname)

    ###################
    #   Index utils   #
    ###################

    def get_bin_id(self, pos):
        return pos // self.bin_size

    ########################
    #   Index lookup ops   #
    ########################

    def initialize_grid(self, interval_a, interval_b):
        start_bin_id_a = self.get_bin_id(interval_a.start)
        end_bin_id_a = self.get_bin_id(interval_a.end)
        n_bins_a = end_bin_id_a - start_bin_id_a
        start_bin_id_b = self.get_bin_id(interval_b.start)
        end_bin_id_b = self.get_bin_id(interval_b.end)
        n_bins_b = end_bin_id_b - start_bin_id_b
        return np.zeros((n_bins_a, n_bins_b)), start_bin_id_a, start_bin_id_b

    def intersect(self, signal, interval_a, interval_b):
        signal_bins = self.bins[signal][self.tid]
        counts, start_bin_id_a, start_bin_id_b = self.initialize_grid(interval_a, interval_b)
        for i in range(counts.shape[0]):
            for j in range(counts.shape[1]):
                bin_a = i + start_bin_id_a
                bin_b = j + start_bin_id_b
                if signal not in [SVSignals.LLRR, SVSignals.RL] or bin_a != bin_b:
                    counts[i][j] = len(signal_bins[bin_a].intersection(signal_bins[bin_b]))
        return counts

    def scalar_apply(self, signal, interval_a, interval_b, op=operator.sub):
        counts, start_bin_id_a, start_bin_id_b = self.initialize_grid(interval_a, interval_b)
        for i in range(counts.shape[0]):
            for j in range(counts.shape[1]):
                counts[i][j] = op(self.bins[signal][self.tid][i + start_bin_id_a],
                                  self.bins[signal][self.tid][j + start_bin_id_b])
        return counts

    #################
    #   Index IO    #
    #################

    @staticmethod
    def get_fname(chr_name, data_config):
        return "%s.%s.%d.%s.auxindex" % (data_config.bam, chr_name, data_config.bin_size, data_config.signal_set_origin)

    def store(self, fname):
        with open(fname, 'wb') as fp:
            pickle.dump(self, fp)

    @staticmethod
    def load(fname):
        logging.info("Loading AUX index: %s" % fname)
        with open(fname, 'rb') as fp:
            return pickle.load(fp)

    @staticmethod
    def generate(chr_name, data_config):
        fname = AlnIndex.get_fname(chr_name, data_config)
        logging.info("Generating AUX index: %s" % fname)
        chr_index = io.load_faidx(data_config.fai)
        aux_index = AlnIndex(chr_index, data_config.bin_size, chr_name, data_config.signal_set)
        for read in io.bam_iter(data_config.bam, 0, chr_name, bx_tag=False):
            aux_index.add(read, data_config.min_mapq_dict)
        aux_index.store(fname)
        return aux_index

    @staticmethod
    def generate_or_load(chr_name, data_config):
        fname = AlnIndex.get_fname(chr_name, data_config)
        if os.path.isfile(fname):  # load the index if it already exists
            return AlnIndex.load(fname)
        # compute the index
        return AlnIndex.generate(chr_name, data_config)


