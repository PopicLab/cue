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
        self.chr_index = chr_index
        self.bin_size = bin_size
        self.tid = self.chr_index.tid(chr_name)
        self.signal_set = constants.SV_SIGNALS_BY_TYPE[signal_set_type]
        self.bins = defaultdict(dict)
        n_bin = int(int(chr_index.chr(self.tid).len) / bin_size + 2)
        for signal in self.signal_set:
            if signal in constants.SV_SIGNAL_SCALAR:
                self.bins[signal][self.tid] = [0] * n_bin
            else:
                self.bins[signal][self.tid] = [set() for _ in range(n_bin)]

    def get_bin_id(self, pos):
        return pos // self.bin_size

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
            bin_pos = min(read.pos + len(read.seq) // 2, self.chr_index.chr(self.tid).len - 1)
            bin_id = self.get_bin_id(bin_pos)
        assert len(self.bins[signal][self.tid]) > bin_id, "%d %d %d" % \
                                                          (read.pos, bin_id, len(self.bins[signal][self.tid]))

        if signal in constants.SV_SIGNAL_PAIRED:
            rp_type = self.get_read_pair_type(read)

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
                if (signal == SVSignals.LLRR and rp_type != constants.SV_SIGNAL_RP_TYPE.LLRR) or \
                   (signal == SVSignals.RL and rp_type != constants.SV_SIGNAL_RP_TYPE.RL):
                    return
                self.bins[signal][self.tid][bin_id].add(read.qname)


    @staticmethod
    def get_read_pair_type(read, dist_thr=5):
        rp_type = constants.SV_SIGNAL_RP_TYPE.LR  # normally paired reads (LR orientation)
        # TODO: if one of the reads in the pair is ambiguously mapped, can't infer the correct orientation
        if read.is_reverse == read.mate_is_reverse:
            # read pairs in RR or LL orientation
            rp_type = constants.SV_SIGNAL_RP_TYPE.LLRR
        elif ((read.reference_start + dist_thr) < read.next_reference_start and read.is_read2 and read.is_reverse) or \
                (read.reference_start > (read.next_reference_start + dist_thr) and read.is_read1 and not read.is_reverse):
            # read pairs in RL (R2F1) orientation
            rp_type = constants.SV_SIGNAL_RP_TYPE.RL
        return rp_type

    def add(self, read, min_mapq_dict):
        for signal in self.signal_set:
            self.add_by_signal(read, signal, min_mapq_dict)

    @staticmethod
    def intersection(tid1, bin_id1, tid2, bin_id2, bins):
        return len(bins[tid1][bin_id1].intersection(bins[tid2][bin_id2]))

    @cached(
        cache=LRUCache(maxsize=1000000),
        key=lambda self, tid1, bin_id1, tid2, bin_id2, bins: hashkey(min(bin_id1, bin_id2), max(bin_id1, bin_id2))
    )
    def cached_intersection(self, tid1, bin_id1, tid2, bin_id2, bins):
        return self.intersection(tid1, bin_id1, tid2, bin_id2, bins)

    def initialize_grid(self, interval_a, interval_b):
        start_bin_id_a = self.get_bin_id(interval_a.start)
        end_bin_id_a = self.get_bin_id(interval_a.end)
        n_bins_a = end_bin_id_a - start_bin_id_a
        start_bin_id_b = self.get_bin_id(interval_b.start)
        end_bin_id_b = self.get_bin_id(interval_b.end)
        n_bins_b = end_bin_id_b - start_bin_id_b
        return np.zeros((n_bins_a, n_bins_b)), start_bin_id_a, start_bin_id_b

    def intersect(self, signal, interval_a, interval_b):
        counts, start_bin_id_a, start_bin_id_b = self.initialize_grid(interval_a, interval_b)
        # cache = {}
        # start = time.time()
        for i in range(counts.shape[0]):
            for j in range(counts.shape[1]):
                bin_a = i + start_bin_id_a
                bin_b = j + start_bin_id_b
                # key = (min(bin_a, bin_b), max(bin_a, bin_b))
                # if key not in cache:
                if signal not in [SVSignals.LLRR, SVSignals.RL] or bin_a != bin_b:
                    counts[i][j] = self.intersection(self.chr_index.tid(interval_a.chr_name), bin_a,
                                                     self.chr_index.tid(interval_b.chr_name), bin_b, self.bins[signal])
                # bc_overlap_counts[i][j] = cache[key]
        # print(time.time() - start)
        return counts

    def combine(self, signal1, signal2, interval_a, interval_b, op1=np.mean, op2=operator.truediv):
        counts, start_bin_id_a, start_bin_id_b = self.initialize_grid(interval_a, interval_b)
        tid_a = self.chr_index.tid(interval_a.chr_name)
        tid_b = self.chr_index.tid(interval_b.chr_name)
        for i in range(counts.shape[0]):
            for j in range(counts.shape[1]):
                bin_a = i + start_bin_id_a
                bin_b = j + start_bin_id_b
                s1_a = self.bins[signal1][tid_a][bin_a]
                s1_b = self.bins[signal1][tid_b][bin_b]
                s2_a = self.bins[signal2][tid_a][bin_a]
                s2_b = self.bins[signal2][tid_b][bin_b]
                if signal1 not in constants.SV_SIGNAL_SCALAR:
                    s1_op = len(s1_a.intersection(s1_b))
                else:
                    s1_op = op1([s1_a, s1_b])
                if signal2 not in constants.SV_SIGNAL_SCALAR:
                    s2_op = len(s2_a.intersection(s2_b))
                else:
                    s2_op = op1([s2_a, s2_b])
                if s1_op == 0:
                    counts[i][j] = 0
                else:
                    counts[i][j] = op2(s1_op, (s1_op + s2_op))
        return counts

    def scalar_apply(self, signal, interval_a, interval_b, op=operator.sub):
        counts, start_bin_id_a, start_bin_id_b = self.initialize_grid(interval_a, interval_b)
        for i in range(counts.shape[0]):
            for j in range(counts.shape[1]):
                counts[i][j] = op(self.bins[signal][self.chr_index.tid(interval_a.chr_name)][i + start_bin_id_a],
                                  self.bins[signal][self.chr_index.tid(interval_b.chr_name)][j + start_bin_id_b])
        return counts

    def store(self, fname):
        with open(fname, 'wb') as fp:
            pickle.dump(self, fp)

    @staticmethod
    def load(fname):
        with open(fname, 'rb') as fp:
            return pickle.load(fp)

    @staticmethod
    def load_chr(fname_prefix, bin_size, chr_name, signal_set_origin):
        fname = "%s.%s.%d.%s.auxindex" % (fname_prefix, chr_name, bin_size, signal_set_origin)
        return AlnIndex.load(fname)


    @staticmethod
    def merge_chr(bam_fname1, bam_fname2, chr_name, bin_size, signal_set1, signal_set2):
        aux_index1 = AlnIndex.load_chr(bam_fname1, bin_size, chr_name, signal_set1.name)
        aux_index2 = AlnIndex.load_chr(bam_fname2, bin_size, chr_name, signal_set2)
        merged_index_fname = "%s.%s.%d.%s_%s.auxindex" % (bam_fname1, chr_name, bin_size, signal_set1.name, signal_set2)
        logging.info("Generating AUX index %s" % merged_index_fname)
        merged_auxindex = AlnIndex(aux_index1.chr_index, bin_size, chr_name, constants.SV_SIGNAL_SET.EMPTY)
        tid = merged_auxindex.tid
        for signal in aux_index1.signal_set:
            merged_auxindex.bins[signal][tid] = aux_index1.bins[signal][tid] 
        for signal in aux_index2.signal_set:
            merged_auxindex.bins[constants.to_ref_signal(signal)][tid] = aux_index2.bins[signal][tid]
        merged_auxindex.store(merged_index_fname)

    @staticmethod
    def generate_or_load_chr(bam_fname, chr_name, fai_fname, bin_size, min_mapq_dict, signal_set,
                             signal_set_origin, bam_type):
        # load the index if it already exists
        aux_index = None
        index_fname = "%s.%s.%d.%s.auxindex" % (bam_fname, chr_name, bin_size, signal_set_origin)
        if os.path.isfile(index_fname):
            aux_index = AlnIndex.load(index_fname)
            logging.info("Loaded AUX index: %s" % index_fname)
        if aux_index is None:  # compute the index
            aux_index = AlnIndex.generate(bam_fname, chr_name, fai_fname, bin_size, min_mapq_dict, signal_set,
                                          signal_set_origin, bam_type)
        return aux_index

    @staticmethod
    def generate(bam_fname, chr_name, fai_fname, bin_size, min_mapq_dict, signal_set, signal_set_origin, bam_type):
        index_fname = "%s.%s.%d.%s.auxindex" % (bam_fname, chr_name, bin_size, signal_set_origin)
        logging.info("Generating AUX index %s" % index_fname)
        chr_index = io.load_faidx(fai_fname)
        aux_index = AlnIndex(chr_index, bin_size, chr_name, signal_set)
        assert bam_type != constants.BAM_TYPE.LONG
        for read in io.bam_iter(bam_fname, 0, chr_name, bx_tag=False,
                                am_tag=(bam_type == constants.BAM_TYPE.LINKED)):
            aux_index.add(read, min_mapq_dict)
        aux_index.store(index_fname)
        return aux_index
