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


from seq.intervals import GenomeInterval, GenomeIntervalPair
import seq.io as io
import logging


class GenomeScanner:
    def __init__(self, aln_index, interval_size, step_size):
        self.chr = aln_index.chr
        self.interval_size = interval_size
        self.step_size = step_size

    def log_intervals(self, x, y):
        logging.info("Interval pair: %s x=%d y=%d" % (self.chr.name, x, y))


class TargetIntervalScanner(GenomeScanner):
    def __init__(self, aln_index, interval_size, step_size):
        super().__init__(aln_index, interval_size, step_size)
        self.interval_pairs = aln_index.interval_pairs
        logging.info("Number of target interval pairs: %d" % len(self.interval_pairs))

    def __iter__(self):
        for x_id, y_id in self.interval_pairs:
            x = x_id * self.step_size
            y = y_id * self.step_size
            if x + self.interval_size > self.chr.len or y + self.interval_size > self.chr.len: continue 
            self.log_intervals(x, y)           
            yield GenomeIntervalPair(GenomeInterval(self.chr.name, x, x + self.interval_size),
                                     GenomeInterval(self.chr.name, y, y + self.interval_size))

class SlidingWindowScanner(GenomeScanner):
    def __init__(self, aln_index, interval_size, step_size, shift_size=None):
        super().__init__(aln_index, interval_size, step_size)
        self.shifts = shift_size if shift_size is not None else [0]

    def __iter__(self):
        for x in range(0, self.chr.len, self.step_size):
            if x + self.interval_size > self.chr.len: continue
            interval_x = GenomeInterval(self.chr.name, x, x + self.interval_size)
            for shift in self.shifts:
                y = x + shift
                if y + self.interval_size > self.chr.len: continue
                self.log_intervals(x, y)
                yield GenomeIntervalPair(interval_x, GenomeInterval(self.chr.name, y, y + self.interval_size))
