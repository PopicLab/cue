from seq.intervals import SVIntervalTree
from seq.io import BedRecord
import functools
import seq.io as io
import logging

def invalid(sv_record):
    # detections below the diagonal (y < x)
    if sv_record.intervalB.start <= sv_record.intervalA.start:
        return True
    return False

def filter_by_region(sv_candidates, blacklist_bed):
    blacklist = io.GenomeBlacklist(blacklist_bed)
    results = []
    for sv in sv_candidates:
        if not blacklist.overlaps(sv):
            results.append(sv)
        else:
            logging.debug("FILTER blacklist: %s" % sv)
    return results

def nms1D(sv_candidates):
    # filter out any invalid and duplicated calls (e.g. calls from different images that overlap the same interval)
    sv_candidates = sorted(sv_candidates, key=functools.cmp_to_key(BedRecord.compare_by_score), reverse=True)
    sv_tree = SVIntervalTree([])
    results = []
    for sv in sv_candidates:
        if not sv_tree.overlaps(sv):
            sv_tree.add(sv)
            results.append(sv)
        else:
            logging.debug("FILTER nms1D: %s" % sv)
    # TODO: record near duplicates (graph -> connected components)
    #       select consensus locations
    #       update confidence score based on consensus
    return results

def remove_filtered_dups(sv_candidates, filtered_candidates):
    # remove any SV keypoints near keypoints that were filtered out
    # this can happen when an overlapping event is called in a subset of images only
    sv_tree = SVIntervalTree(filtered_candidates)
    results = []
    for sv in sv_candidates:
        if not sv_tree.overlaps(sv):
            results.append(sv)
        else:
            logging.debug("FILTER filtered-dups: %s" % sv)
    return results


def filter_svs(sv_candidates, blacklist_bed, filtered_sv_candidates=None):
    sv_candidates = nms1D(sv_candidates)
    if blacklist_bed is not None:
         sv_candidates = filter_by_region(sv_candidates, blacklist_bed)
    return sv_candidates

def merge_sv_candidates(sv_candidates1, sv_candidates2):
    sv_tree = SVIntervalTree(sv_candidates1)
    results = sv_candidates1
    for sv in sv_candidates2:
        if not sv_tree.overlaps(sv):
            results.append(sv)
    return results
