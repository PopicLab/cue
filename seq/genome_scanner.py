from seq.intervals import GenomeInterval, GenomeIntervalPair
import seq.io as io
import logging

class SVGenomeScanner:
    def __init__(self, chr_index, interval_size, step_size=None, shift_size=None, blacklist_bed=None,
                 include_chrs=None, exclude_chrs=None):
        self.chr_index = chr_index
        self.blacklist = None if blacklist_bed is None else io.GenomeBlacklist(blacklist_bed)
        self.chromosomes = self.chr_index.contigs()
        self.interval_size = interval_size
        self.step_size = interval_size if step_size is None else step_size
        self.shifts = shift_size if shift_size is not None else [0]
        self.min_interval_len = 1000
        self.include_chrs = include_chrs  # None: include all chromosomes
        self.exclude_chrs = exclude_chrs if exclude_chrs is not None else []

    def __iter__(self):
        for chr in self.chr_index.contigs():
            if chr.name in self.exclude_chrs or (self.include_chrs is not None and chr.name not in self.include_chrs):
                continue
            current_pos = 0
            while current_pos < chr.len:
                interval = GenomeInterval(chr.name, current_pos, min(current_pos + self.interval_size, chr.len))
                logging.debug("Interval: %s" % str(interval))
                if current_pos and current_pos % 1000000 == 0:
                    logging.info("Scanned %d loci on %s" % (current_pos, chr.name))
                next_gap = self.blacklist.next_gap_overlap(interval) if self.blacklist is not None else None
                # trim interval if overlapping a blacklist region
                if next_gap is not None:
                    interval.end = next_gap.start
                # check if this is a valid interval
                if len(interval) >= self.min_interval_len:
                    for shift in self.shifts:
                        shift_pos = current_pos + shift
                        if shift_pos + self.interval_size <= chr.len:  # todo: boundary
                            interval_shifted = GenomeInterval(chr.name, shift_pos, min(shift_pos + self.interval_size,
                                                                                       chr.len))
                            yield GenomeIntervalPair(interval, interval_shifted)
                # next interval to start at the end of the overlapping gap
                current_pos = current_pos + self.step_size if next_gap is None else next_gap.end

