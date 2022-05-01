from collections import defaultdict
from intervaltree import IntervalTree
import random

class GenomeInterval(tuple):
    def __new__(cls, chr_name, start, end):
        return tuple.__new__(GenomeInterval, (chr_name, start, end))

    def __init__(self, chr_name, start, end):
        self.chr_name = chr_name
        self.start = start
        self.end = end

    def __len__(self):
        return self.end - self.start

    def mean(self):
        return self.start + int((self.end - self.start)/2)

    def pad(self, chr_index, padding, rand=False):
        start_pad = padding
        end_pad = padding
        if rand:
            split = random.randint(0, padding//2)
            start_pad = padding//2 + split
            end_pad = 2*padding - start_pad
        start = max(0, self.start - start_pad)
        end = min(self.end + end_pad, chr_index.chr(chr_index.tid(self.chr_name)).len)
        return GenomeInterval(self.chr_name, start, end)

    def __str__(self):
        return "%s_%d-%d" % (self.chr_name, self.start, self.end)

    def __lt__(self, interval):
        return self.start < interval.start

    def to_list(self, chr_index):
        return [chr_index.tid(self.chr_name), self.start, self.end]

    @staticmethod
    def from_list(interval_list, chr_index):
        return GenomeInterval(chr_index.chr(interval_list[0]).name, interval_list[1], interval_list[2])

    @staticmethod
    def to_interval(chr_name, pos):
        return GenomeInterval(chr_name, pos, pos + 1)


class GenomeIntervalPair:
    def __init__(self, intervalA, intervalB):
        self.intervalA = intervalA
        self.intervalB = intervalB

    def dist_mean_of_intervals(self):
        return abs(self.intervalA.mean() - self.intervalB.mean())

    def __len__(self):
        return self.intervalB.start - self.intervalA.start

    def __str__(self):
        return "%s_&_%s" % (str(self.intervalA), str(self.intervalB))

    def to_list(self, chr_index):
        return [self.intervalA.to_list(chr_index), self.intervalB.to_list(chr_index)]

    @staticmethod
    def from_list(interval_pair_list, chr_index):
        return GenomeIntervalPair(GenomeInterval.from_list(interval_pair_list[0], chr_index),
                                  GenomeInterval.from_list(interval_pair_list[1], chr_index))


class SVIntervalTree:
    def __init__(self, intervals):
        self.chr2tree = defaultdict(IntervalTree)
        for interval in intervals:
            if not self.overlaps(interval):
                self.add(interval)

    def overlaps(self, interval, delta=50, frac=0.2):
        start = max(0, interval.intervalA.start - delta)
        end = interval.intervalB.start + delta
        if not self.chr2tree[interval.intervalA.chr_name].overlaps(start, end):
            return False
        candidates = self.chr2tree[interval.intervalA.chr_name].overlap(start, end)
        for c in candidates:
            candidate_interval = c.data
            overlap_start = max(candidate_interval.intervalA.start, start)
            overlap_end = min(candidate_interval.intervalB.start, end)
            candidate_len = candidate_interval.intervalB.start - candidate_interval.intervalA.start
            if overlap_start < overlap_end:
                if float((overlap_end - overlap_start) / min(end - start, candidate_len)) >= frac:
                    return True
        return False

    def add(self, interval):
        self.chr2tree[interval.intervalA.chr_name].addi(interval.intervalA.start,
                                                        interval.intervalB.start, interval)
