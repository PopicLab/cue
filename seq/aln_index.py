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
import itertools

from seq import io
from seq import utils
import numpy as np
import pickle
import os.path
from img import constants
from img.constants import SVSignals
from collections import defaultdict
import operator
import logging
import os


class AlnIndex:
    def __init__(self, chr_name, config):
        self.chr_index = io.load_faidx(config.fai, all=True)
        self.chr = self.chr_index.chr_from_name(chr_name)
        self.config = config
        self.signal_set = constants.SV_SIGNALS_BY_TYPE[config.signal_set]
        self.bins = {}
        self.n_bins = self.chr.len // self.config.bin_size + 2
        for signal in self.signal_set:
            if signal in constants.SV_SIGNAL_SCALAR:
                self.bins[signal] = [0] * self.n_bins
            else:
                self.bins[signal] = [set() for _ in range(self.n_bins)]
        logging.info("Number of bins: %d" % self.n_bins)
        if self.config.scan_target_intervals:
            self.valid_interval_reads = set()
            self.interval_pair_support = defaultdict(int)  # stores the number of discordant pairs across the two intervals
            self.intervals = set()
            self.interval_pairs = []
            self.steps_per_interval = self.config.interval_size // self.config.step_size
        if self.config.stream:
            assert self.config.scan_target_intervals, "Streaming is only supported when scan_target_intervals=True"
            self.stream_handle = None
            self.last_read_block_id = 0
            self.last_write_block_id = None
            self.n_stream_blocks = self.n_bins // self.config.bins_per_block + 1

    #####################
    #   Index editing   #
    #####################

    @staticmethod
    def generate(chr_name, data_config):
        fname = AlnIndex.get_fname(chr_name, data_config)
        logging.info("Generating AUX index: %s" % fname)
        aux_index = AlnIndex(chr_name, data_config)
        if data_config.stream:
            aux_index.open_stream_handle('wb')
        n_reads = 0
        for read in io.bam_iter(data_config.bam, chr_name):
            aux_index.add(read)
            n_reads += 1
        if data_config.scan_target_intervals:
            aux_index.select_intervals()
            aux_index.finalize()
        aux_index.store(fname)
        logging.info("Processed %d reads" % n_reads)
        return aux_index

    def add(self, read):
        # 1. update the index bins
        bin_id = self.get_bin_id(self.get_read_bin_pos(read))
        rp_type = constants.get_read_pair_type(read)
        for signal in self.signal_set:
            if signal in constants.SV_SIGNAL_INDEX:
                self.add_by_signal(read, signal, bin_id, rp_type)

        if not self.config.scan_target_intervals: return
        # 2. collect the read pair information for interval selection
        if self.is_valid_interval_read(read) and read.qname not in self.valid_interval_reads:
            self.valid_interval_reads.add(read.qname)
            # get the interval pairs to which each read in the pair maps
            interval_ids_read = self.get_interval_ids(self.get_read_bin_pos(read))
            interval_ids_mate = self.get_interval_ids(self.get_mate_bin_pos(read))
            # if the pair falls into the same interval(s), add support only for such interval(s)
            if abs(max(interval_ids_read) - max(interval_ids_mate)) < self.steps_per_interval:
                for interval_id_shared in sorted(set(interval_ids_read).intersection(interval_ids_mate)):
                    self.interval_pair_support[(interval_id_shared, interval_id_shared)] += 1
            else:
                for i, j in itertools.product(interval_ids_read, interval_ids_mate):
                    self.interval_pair_support[tuple(sorted((i, j)))] += 1

        if not self.config.stream: return
        # 3. if streaming, store and deallocate stale bins
        current_block_id = self.get_streaming_block_id(bin_id)
        if current_block_id > self.last_read_block_id: # new block
            self.write_stream_and_free_mem(current_block_id)
            self.last_read_block_id = current_block_id

    def select_intervals(self):
        for interval_pair in sorted(self.interval_pair_support.keys()):
            count = self.interval_pair_support[interval_pair]
            if count >= self.config.min_pair_support:
                self.intervals.add(interval_pair[0])
                self.intervals.add(interval_pair[1])
                self.interval_pairs.append(interval_pair)
        logging.info("Selected %d intervals" % len(self.intervals))
        logging.info("Selected %d interval pairs out of %d pairs" % (len(self.interval_pairs), len(self.interval_pair_support)))

    def finalize(self):
        if self.config.stream:
            self.write_stream_and_free_mem(None)
            self.close_stream_handle()
            self.open_stream_handle('rb')
            self.last_read_block_id = None
        # we only need the data for selected intervals
        last_bin_id = 0
        for interval_id in sorted(self.intervals):
            bin_id_start, bin_id_end = self.get_bin_range(interval_id)
            if self.config.stream:
                self.load_stream(self.get_streaming_block_id(bin_id_end))
            self.clear_bin_sets(last_bin_id, bin_id_start)
            last_bin_id = bin_id_end
        self.clear_bin_sets(last_bin_id, self.n_bins)
        self.intervals = None
        self.interval_pair_support = None
        if self.config.stream:
            self.close_stream_handle(delete_file=True)

    def add_by_signal(self, read, signal, bin_id, rp_type, min_clipped_len=10):
        if signal not in self.bins or not self.is_valid_index_read(read, signal):
            return
        if signal in constants.SV_SIGNAL_SCALAR:
            # soft (op 4) or hard clipped (op 5)
            # TODO(viq): expose min clipped length param
            if signal == SVSignals.RD_CLIPPED and not (
               (read.cigartuples[0][0] in [4, 5] and read.cigartuples[0][1] > min_clipped_len) or \
               (read.cigartuples[-1][0] in [4, 5] and read.cigartuples[-1][1] > min_clipped_len)):
                return
            self.bins[signal][bin_id] += 1
        else:
            if (signal == SVSignals.LLRR and rp_type != constants.SV_SIGNAL_RP_TYPE.LLRR) or \
               (signal == SVSignals.RL and rp_type != constants.SV_SIGNAL_RP_TYPE.RL):
               return
            self.bins[signal][bin_id].add(read.qname)
                    

    ###################
    #   Index utils   #
    ###################

    def is_valid_index_read(self, read, signal):
        if read.is_unmapped or read.mapping_quality < self.config.signal_mapq[signal] or \
                (signal in constants.SV_SIGNAL_PAIRED and (read.mate_is_unmapped or
                                                           read.next_reference_name != read.reference_name)):
            return False
        # filter out singleton alignments with low quality regardless of config
        if (read.mate_is_unmapped or read.next_reference_name != read.reference_name) and read.mapping_quality < 20:
            return False
        return True

    def is_valid_interval_read(self, read):
        # read pair must be mapped to the same chromosome
        if read.is_unmapped or read.mate_is_unmapped or not read.is_paired or \
           read.is_secondary or read.mapping_quality == 0 or read.next_reference_name != read.reference_name:
            return False
        # read pair distance must be within the configured range
        return self.config.min_pair_distance <= \
               abs(read.reference_start - read.next_reference_start) <= self.config.max_pair_distance

    def get_bin_id(self, pos):
        return pos // self.config.bin_size

    def get_interval_ids(self, pos):
        start_interval_id = pos // self.config.step_size
        intervals = [start_interval_id - i for i in range(self.steps_per_interval) if start_interval_id - i >= 0]
        return intervals

    def get_streaming_block_id(self, bin_id):
        return bin_id // self.config.bins_per_block

    def get_bin_range(self, interval_id):
        interval_pos_start = interval_id * self.config.step_size
        interval_pos_end = min(interval_pos_start + self.config.interval_size, self.chr.len)
        return self.get_bin_id(interval_pos_start), self.get_bin_id(interval_pos_end)

    def get_read_bin_pos(self, read):
        if read.reference_end is not None:
            return read.pos + (read.reference_end - read.reference_start) // 2
        return min(read.pos + len(read.seq) // 2, self.chr.len - 1)

    def get_mate_bin_pos(self, read):
        return min(read.next_reference_start + len(read.seq) // 2, self.chr.len - 1)

    def initialize_grid(self, interval_a, interval_b):
        start_bin_id_a = self.get_bin_id(interval_a.start)
        end_bin_id_a = self.get_bin_id(interval_a.end)
        n_bins_a = end_bin_id_a - start_bin_id_a
        start_bin_id_b = self.get_bin_id(interval_b.start)
        end_bin_id_b = self.get_bin_id(interval_b.end)
        n_bins_b = end_bin_id_b - start_bin_id_b
        return np.zeros((n_bins_a, n_bins_b)), start_bin_id_a, start_bin_id_b

    def clear_bin_sets(self, start_bin_id, end_bin_id):
        for signal in self.signal_set:
            if signal not in constants.SV_SIGNAL_SCALAR:
                self.bins[signal][start_bin_id:end_bin_id] = [None] * (end_bin_id - start_bin_id) 

    ########################
    #   Index lookup ops   #
    ########################

    def intersect(self, signal, interval_a, interval_b, off_diagonal_only):
        signal_bins = self.bins[signal]
        counts, start_bin_id_a, start_bin_id_b = self.initialize_grid(interval_a, interval_b)
        for i in range(counts.shape[0]):
            for j in range(counts.shape[1]):
                bin_a = i + start_bin_id_a
                bin_b = j + start_bin_id_b
                if not signal_bins[bin_a] or not signal_bins[bin_b]: continue
                if not off_diagonal_only or bin_a != bin_b:
                    counts[i][j] = len(signal_bins[bin_a].intersection(signal_bins[bin_b]))
        return counts

    def scalar_apply(self, signal, interval_a, interval_b, op=operator.sub):
        counts, start_bin_id_a, start_bin_id_b = self.initialize_grid(interval_a, interval_b)
        for i in range(counts.shape[0]):
            for j in range(counts.shape[1]):
                counts[i][j] = op(self.bins[signal][i + start_bin_id_a],
                                  self.bins[signal][j + start_bin_id_b])
        return counts

    #################
    #   Index IO    #
    #################

    @staticmethod
    def get_fname(chr_name, data_config):
        return "%s.%s.%d.%s.auxindex" % (data_config.bam, chr_name, data_config.bin_size, data_config.signal_set_origin)

    def open_stream_handle(self, mode):
        stream_fname = "%s.stream" % AlnIndex.get_fname(self.chr.name, self.config)
        self.stream_handle = open(stream_fname, mode)    

    def close_stream_handle(self, delete_file=False):
        if self.stream_handle:
            self.stream_handle.close()
        self.stream_handle = None
        if delete_file:
            os.remove("%s.stream" % AlnIndex.get_fname(self.chr.name, self.config))

    def write_stream_and_free_mem(self, current_block_id, n_buffer_blocks=1):
        free_block_id = 0 if self.last_write_block_id is None else self.last_write_block_id + 1 
        free_block_id_thr = current_block_id - n_buffer_blocks if current_block_id else self.n_stream_blocks 
        while free_block_id < free_block_id_thr:
            self.last_write_block_id = free_block_id
            bin_id_start = free_block_id * self.config.bins_per_block
            bin_id_end = min(bin_id_start + self.config.bins_per_block, self.n_bins)
            for signal in self.signal_set:
                if signal in constants.SV_SIGNAL_SCALAR: continue
                pickle.dump(self.bins[signal][bin_id_start:bin_id_end], self.stream_handle)   
                self.bins[signal][bin_id_start:bin_id_end] = [None] * (bin_id_end - bin_id_start) 
            free_block_id += 1

    def load_stream(self, block_id):
        load_block_id = 0 if self.last_read_block_id is None else self.last_read_block_id + 1
        while load_block_id <= block_id:
            self.last_read_block_id = load_block_id
            bin_id_start = load_block_id * self.config.bins_per_block
            bin_id_end = min(bin_id_start + self.config.bins_per_block, self.n_bins)
            for signal in self.signal_set:
                if signal in constants.SV_SIGNAL_SCALAR: continue
                self.bins[signal][bin_id_start:bin_id_end] = pickle.load(self.stream_handle)
            load_block_id += 1
 
    def store(self, fname):
        with open(fname, 'wb') as fp:
            pickle.dump(self, fp)

    @staticmethod
    def load(fname):
        logging.info("Loading AUX index: %s" % fname)
        with open(fname, 'rb') as fp:
            return pickle.load(fp)
    
    @staticmethod
    def generate_or_load(chr_name, data_config):
        fname = AlnIndex.get_fname(chr_name, data_config)
        if os.path.isfile(fname):  # load the index if it already exists
            return AlnIndex.load(fname)
        # compute the index
        return AlnIndex.generate(chr_name, data_config)
