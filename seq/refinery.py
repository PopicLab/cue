from collections import defaultdict
from seq.io import bam_iter_interval
import numpy as np


def refine_svs(sv_candidates, config, chr_index):
    sv_candidates_refined = []
    for sv_call in sv_candidates:
        new_call = refine_call(sv_call, config, chr_index.chr_from_name(sv_call.intervalA.chr_name))
        if new_call:
            sv_candidates_refined.append(new_call)
    return sv_candidates_refined

def refine_call(sv_call, config, chr):
    new_start = refine_bp(chr, sv_call, True, config)
    new_end = refine_bp(chr, sv_call, False, config)
    if not new_start and not new_end:
        return None
    if new_start:
        sv_call.intervalA.start = int(new_start)
        sv_call.intervalA.end = int(new_start) + 1
    if new_end:
        sv_call.intervalB.start = int(new_end)
        sv_call.intervalB.end = int(new_end) + 1
    return sv_call

def refine_bp(chr, sv_call, is_bp_start, config):
    # extract reads at predicted breakpoints
    # refine the position based on read-pair/split-read consensus
    init_bp = sv_call.intervalA.start if is_bp_start else sv_call.intervalB.start
    bp2evidence = collect_read_info(chr, sv_call, is_bp_start, config)
    if not bp2evidence: return None
    for kernel_size in config.refine_bp_kernels: 
        bp2evidence = convolve_evidence(bp2evidence, kernel_size, chr.len)
        max_evidence = max(bp2evidence.values())
        if max_evidence >= config.refine_min_support:
            if is_bp_start: return max([k for k, v in bp2evidence.items() if v == max_evidence])
            return min([k for k, v in bp2evidence.items() if v == max_evidence])
    return None

def convolve_evidence(evidence_dict, kernel_size, chr_len):
    evidence_dict_conv = defaultdict(int)
    for i in range(min(evidence_dict.keys()), max(evidence_dict.keys())+1):
        for j in range(-kernel_size, kernel_size+1):
            if i + j < 0 or i + j >= chr_len: continue  
            evidence_dict_conv[i] += evidence_dict[i+j]
    return evidence_dict_conv

def get_bp_buffer(bp, chr, sv_len, config):
    buffer_size = max(config.min_refine_buffer, sv_len//config.refine_buffer_frac_size)
    return max(0, bp - buffer_size), min(bp + buffer_size, chr.len)

def collect_read_info(chr, sv_call, is_bp_start, config):
    sv_len = sv_call.intervalB.end - sv_call.intervalA.end
    bp1_start, bp1_end = get_bp_buffer(sv_call.intervalA.start, chr, sv_len, config)
    bp2_start, bp2_end = get_bp_buffer(sv_call.intervalB.start, chr, sv_len, config)
    start, end, start_other, end_other = bp1_start, bp1_end, bp2_start, bp2_end
    if not is_bp_start:
        start, end, start_other, end_other = bp2_start, bp2_end, bp1_start, bp1_end
    bp2evidence = defaultdict(int)
    for read in bam_iter_interval(config.bam, chr.name, start, end):
        read_pos = read.reference_end if is_bp_start else read.reference_start
        if read.is_unmapped or read_pos < start or read_pos > end: continue 
        bp2evidence[read_pos] += split_support(read, start_other, end_other)
        bp2evidence[read_pos] += pair_support(read, start_other, end_other, max(config.min_pair_distance, sv_len//config.refine_pair_dist_frac_size))
    return bp2evidence


def pair_support(read, start_other, end_other, min_pair_distance):
    if read.mate_is_unmapped or read.next_reference_name != read.reference_name:
        return False
    return abs(read.reference_start - read.next_reference_start) >= min_pair_distance and start_other <= read.next_reference_start <= end_other


def split_support(read, start_other, end_other):
    if read.has_tag("SA"):
        sa_tags = read.get_tag('SA')
        for sa_tag in sa_tags.split(';'):
            fields = sa_tag.split(',')
            if read.reference_name == fields[0] and start_other <= int(fields[1]) <= end_other:
                return True
    return False
