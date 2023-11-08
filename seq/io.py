from seq.intervals import GenomeInterval
from collections import namedtuple
import os
import pysam
import functools
from pysam import VariantFile
import bisect
from enum import Enum
from collections import defaultdict
import argparse
from intervaltree import IntervalTree
import img.constants as constants

Chr = namedtuple('Chr', 'name len')


class ChrFAIndex:
    def __init__(self):
        self.tid2chr = {}
        self.chr2tid = {}

    def add(self, tid, chr):
        self.tid2chr[tid] = chr
        self.chr2tid[chr.name] = tid

    def tids(self):
        return self.tid2chr.keys()

    def chr(self, tid):
        return self.tid2chr[tid]

    def chr_from_name(self, chr_name):
        return self.tid2chr[self.tid(chr_name)]

    def tid(self, chr_name):
        return self.chr2tid[chr_name]

    def contigs(self):
        return self.tid2chr.values()

    def has(self, chr_name):
        return chr_name in self.chr2tid

    def chr_names(self):
        return self.chr2tid.keys()


def load_faidx(fai_fname, all=False):
    chr_index = ChrFAIndex()
    with open(fai_fname, "r") as faidx:
        for tid, line in enumerate(faidx):
            if not all and tid > 23:
                break  # only keep the autosomes
            # TODO: tid + 1 to correspond to chr names
            name, length, _, _, _ = line[:-1].split()
            chr_index.add(tid, Chr(name, int(length)))
    return chr_index


BED_FILE_TYPE = Enum("BED_FILE_TYPE", 'BED, BEDPE')


class BedRecord:
    def __init__(self, sv_type, intervalA, intervalB=None, aux=None):
        self.intervalA = intervalA
        self.intervalB = intervalB
        self.sv_type = sv_type
        self.aux = aux
        self.format = BED_FILE_TYPE.BEDPE if intervalB is not None else BED_FILE_TYPE.BED

    @staticmethod
    def parse_bed_line(line, bed_file_type=BED_FILE_TYPE.BEDPE):
        fields = line.strip().split()  # "\t")
        assert len(fields) >= 3, "Unexpected number of fields in BED: %s" % line
        chr_name, start, end = fields[:3]
        intervalA = GenomeInterval(chr_name, int(start), int(end))
        intervalB = None
        if bed_file_type == BED_FILE_TYPE.BEDPE:
            assert len(fields) >= 6, "Unexpected number of fields in BEDPE: %s" % line
            chrB, startB, endB = fields[3:6]
            intervalB = GenomeInterval(chrB, int(startB), int(endB))
        req_fields = 3 if bed_file_type == BED_FILE_TYPE.BED else 6
        name = fields[req_fields] if len(fields) > req_fields else 'NA'
        aux = {'score': fields[req_fields + 1] if len(fields) > req_fields + 1 else 0,
               'zygosity': constants.ZYGOSITY_ENCODING_BED[fields[req_fields + 2]] if len(fields) > req_fields + 2
               else constants.ZYGOSITY.UNK}  # TODO: strand vs zygosity
        return BedRecord(name, intervalA, intervalB, aux)

    def get_sv_type(self, to_vcf_format=False):
        if to_vcf_format:
            return "<%s>" % self.sv_type
        return self.sv_type

    def get_score(self):
        return self.aux['score']

    def get_zygosity(self):
        return self.aux['zygosity']

    def get_sv_type_with_zyg(self):
        return "%s-%s" % (self.sv_type, self.get_zygosity().value)

    @staticmethod
    def parse_sv_type_with_zyg(sv_type):
        if "-" in sv_type:
            sv_type, zyg = sv_type.split("-")
            return sv_type, constants.ZYGOSITY(zyg)
        else:
            return sv_type, None

    def __str__(self):
        return "%s, %s, %s" % (self.sv_type, str(self.intervalA), str(self.intervalB))

    def get_name(self):
        return "%s_%s_%s" % (self.sv_type, str(self.intervalA), str(self.intervalB))

    def to_bedpe(self):
        assert self.format == BED_FILE_TYPE.BEDPE
        return "%s\t%s\t%s\t%s\t%s\t%s\t%s" % (self.intervalA.chr_name, self.intervalA.start, self.intervalA.end,
                                               self.intervalB.chr_name, self.intervalB.start, self.intervalB.end,
                                               self.sv_type)

    def to_bedpe_aux(self):
        assert self.format == BED_FILE_TYPE.BEDPE
        return "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s" % (self.intervalA.chr_name,
                                                       self.intervalA.start, self.intervalA.end,
                                                       self.intervalB.chr_name,
                                                       self.intervalB.start, self.intervalB.end,
                                                       self.sv_type, self.aux['score'],
                                                       constants.ZYGOSITY_GT_BED[self.aux['zygosity']])
    
    def to_bed(self):
        assert self.format == BED_FILE_TYPE.BED
        return "%s\t%s\t%s\t%s" % (self.intervalA.chr_name, self.intervalA.start, self.intervalA.end, self.sv_type)

    @staticmethod
    def get_bedpe_header():
        return '#chrom1\tstart1\tstop1\tchrom2\tstart2\tstop2\tname'

    @staticmethod
    def get_bedpe12_header():
        return '#chrom1\tstart1\tstop1\tchrom2\tstart2\tstop2\tname\tscore\tstrand1\tstrand2\tfilter\tinfo'

    @staticmethod
    def get_bedpe_aux_header():
        return '#chrom1\tstart1\tstop1\tchrom2\tstart2\tstop2\tname\tscore\tgt'

    @staticmethod
    def compare(rec1, rec2):
        return rec1.intervalA.start - rec2.intervalA.start

    @staticmethod
    def compare_by_score(rec1, rec2):
        return rec1.get_score() - rec2.get_score()

    def __lt__(self, rec):
        return self.intervalA.__lt__(rec.intervalA)


def bed_iter(bed_fname, bed_file_type=BED_FILE_TYPE.BEDPE, keep_chrs=None, exclude_names=None):
    with open(bed_fname, 'r') as bed_file:
        for line in bed_file:
            if line.startswith('#') or line.isspace():
                continue
            record = BedRecord.parse_bed_line(line, bed_file_type)
            if exclude_names is not None and record.sv_type in exclude_names:
                continue
            if keep_chrs is None or (record.intervalA.chr_name in keep_chrs and
                                     (record.intervalB is None or record.intervalB.chr_name in keep_chrs)):
                yield record


def vcf_iter(vcf_fname, min_size=0, include_types=None):
    vcf_file = VariantFile(vcf_fname)
    for rec in vcf_file.fetch():
        if 'SVTYPE' not in rec.info:
            continue
        sv_type = rec.info['SVTYPE']
        # filter by type
        if include_types is not None and sv_type not in include_types:
            continue
        if 'SVLEN' in rec.info:
            if isinstance(rec.info['SVLEN'], tuple):
                sv_len = int(rec.info['SVLEN'][0])
            else:
                sv_len = int(rec.info['SVLEN'])
        else:
            sv_len = rec.stop - rec.pos
        sv_len = abs(sv_len)
        # filter by length
        if sv_len < min_size:
            continue
        start = int(rec.pos) - 1  # 0-based
        end = start + sv_len
        intervalA = GenomeInterval(rec.contig, start, start + 1)
        intervalB = GenomeInterval(rec.contig, end, end + 1)
        if 'GT' in rec.samples[rec.samples[0].name]:
            gt = rec.samples[rec.samples[0].name]['GT']
        else:
            gt = (None, None)
        if gt[0] == 0 and gt[1] == 0:
            print("!Found a HOM ref entry in the VCF: ", rec)
        zygosity = constants.ZYGOSITY_ENCODING[gt] if gt[0] is not None else constants.ZYGOSITY.UNK
        aux = {'score': rec.qual,
               'zygosity': zygosity}
        bedpe_record = BedRecord(sv_type, intervalA, intervalB, aux)
        yield bedpe_record


def load_bed(bed_fname, bed_file_type=BED_FILE_TYPE.BEDPE, sort=True, keep_chrs=None, exclude_names=None):
    records = []
    for record in bed_iter(bed_fname, bed_file_type, keep_chrs=keep_chrs, exclude_names=exclude_names):
        records.append(record)
    if sort:
        records = sorted(records, key=functools.cmp_to_key(BedRecord.compare))
    return records


class BedRecordContainer:
    def __init__(self, fname):
        self.chr2rec = defaultdict(list)
        self.chr2starts = defaultdict(list)
        self.chr2ends = defaultdict(list)
        iterator = None
        if 'bed' in fname:
            iterator = bed_iter(fname, bed_file_type=BED_FILE_TYPE.BEDPE)
        elif 'vcf' in fname:
            iterator = vcf_iter(fname)
        assert iterator is not None
        for i, record in enumerate(iterator):
            self.chr2rec[record.intervalA.chr_name].append(record)
            self.chr2starts[record.intervalA.chr_name].append(
                (record.intervalA.start, len(self.chr2rec[record.intervalA.chr_name]) - 1))
            self.chr2ends[record.intervalA.chr_name].append(
                (record.intervalB.start, len(self.chr2rec[record.intervalA.chr_name]) - 1))
        for chr_name in self.chr2rec:
            self.chr2starts[chr_name] = sorted(self.chr2starts[chr_name])
            self.chr2ends[chr_name] = sorted(self.chr2ends[chr_name])

    def coords_in_interval(self, interval, coords):
        idx_left = bisect.bisect_left(coords, (interval.start, 0))
        if idx_left >= len(coords):
            return []
        idx_right = bisect.bisect_left(coords, (interval.end, 0))
        return coords[idx_left:idx_right]

    def overlap(self, interval):
        starts = self.coords_in_interval(interval, self.chr2starts[interval.chr_name])
        ends = self.coords_in_interval(interval, self.chr2ends[interval.chr_name])
        # remove dups (some records can start and end in this interval)
        records = set()
        for _, i in starts:
            records.add(self.chr2rec[interval.chr_name][i])
        for _, i in ends:
            records.add(self.chr2rec[interval.chr_name][i])
        return list(records)

    def __iter__(self):
        for chr_name in self.chr2rec:
            for rec in self.chr2rec[chr_name]:
                yield rec


class GenomeBlacklist:
    def __init__(self, bed_fname):
        self.chr2gaps = defaultdict(list)
        self.chr2tree = defaultdict(IntervalTree)

        for record in bed_iter(bed_fname, bed_file_type=BED_FILE_TYPE.BED):
            self.chr2gaps[record.intervalA.chr_name].append(record.intervalA)
            self.chr2tree[record.intervalA.chr_name].addi(record.intervalA.start, record.intervalA.end,
                                                          record.intervalA)
        for chr_name in self.chr2gaps:
            self.chr2gaps[chr_name] = sorted(self.chr2gaps[chr_name])

    def next_gap_overlap(self, interval):
        gaps = self.chr2gaps[interval.chr_name]
        idx_geq_start = bisect.bisect_left(gaps, interval)
        if idx_geq_start >= len(gaps):
            return None
        next_gap = gaps[idx_geq_start]
        if next_gap.end > interval.start:
            return None
        return next_gap

    def overlaps(self, sv_record):
        start = sv_record.intervalA.start
        end = sv_record.intervalB.start
        # TODO: trans-chromosomal events
        return self.chr2tree[sv_record.intervalA.chr_name].overlaps(start, end)


def bam_iter(bam_fname, chr_name=None, read_filters=None):
    file_mode = "rc" if bam_fname.endswith('cram') else "rb"
    input_bam = pysam.AlignmentFile(bam_fname, file_mode)
    n_filtered_reads = 0
    n_reads = 0
    for i, read in enumerate(input_bam.fetch(chr_name)):
        n_reads += 1
        if i % 1000000 == 0:
            print("Processed %d reads" % i)
        if read_filters and any([f(read) for f in read_filters]):
            n_filtered_reads += 1
            continue
        yield read
    print("Read: %d reads" % n_reads)
    print("Filtered: %d reads" % n_filtered_reads)
    input_bam.close()


def bam_iter_interval(bam_fname, chr_name, start, end):
    file_mode = "rc" if bam_fname.endswith('cram') else "rb"
    input_bam = pysam.AlignmentFile(bam_fname, file_mode)
    for read in input_bam.fetch(chr_name, start, end):
        yield read
    input_bam.close()

def get_vcf_format_variant_file(vcf_fname, contigs, ctg_no_len=False):
    with open(vcf_fname, "w") as vcf:
        vcf.write("##fileformat=VCFv4.2\n")
        for ctg in contigs:
            if not ctg_no_len:
                vcf.write("##contig=<ID=%s,length=%d>\n" % (ctg.name, ctg.len))
            else:
                vcf.write("##contig=<ID=%s>\n" % ctg.name)
        # vcf.write("#%s\n" % "\t".join(["CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT"]))
        vcf.write("#%s\n" % "\t".join(["CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO"]))
    vcf_file = VariantFile(vcf_fname)
    # SV fields
    vcf_file.header.info.add('END', number=1, type='Integer', description="End position of the variant "
                                                                          "described in this record")
    vcf_file.header.info.add('CIPOS', number=2, type='Integer', description="Confidence interval around POS for "
                                                                            "imprecise variants")
    vcf_file.header.info.add('CIEND', number=2, type='Integer', description="Confidence interval around END for "
                                                                            "imprecise variants")
    vcf_file.header.info.add('SVTYPE', number=1, type='String', description="Type of structural variant")
    vcf_file.header.info.add('SVLEN', number=1, type='Integer', description="Length of structural variant")
    vcf_file.header.info.add('SVMETHOD', number=1, type='String', description="SV detection method")
    vcf_file.header.formats.add('GT', number=1, type='String', description="Genotype")
    vcf_file.header.add_sample("SAMPLE")
    return vcf_file


def write_bed(out_bed_file, sv_calls):
    bed_file = open(out_bed_file, 'w')
    bed_file.write(BedRecord.get_bedpe_aux_header() + "\n")
    for sv in sv_calls:
        bed_file.write(sv.to_bedpe_aux() + "\n")
    bed_file.close()


def bed2sv_calls(bedpe_file, bed_file_type=BED_FILE_TYPE.BEDPE):
    sv_calls = []
    for record in bed_iter(bedpe_file, bed_file_type):     
        sv_calls.append(record)
    return sv_calls


def bed2vcf(bedpe_file, vcf_fname, fai_fname, bed_file_type=BED_FILE_TYPE.BEDPE, sv_types=None, min_score=None, min_len=0):
    chr_index = load_faidx(fai_fname, all=True)
    vcf_format_file = get_vcf_format_variant_file(vcf_fname + ".format", chr_index.contigs())
    vcf_out_file = VariantFile(vcf_fname, 'w', header=vcf_format_file.header)
    for record in bed_iter(bedpe_file, bed_file_type):
        if not chr_index.has(record.intervalA.chr_name) or not chr_index.has(record.intervalB.chr_name):
            continue
        if sv_types is not None and record.sv_type not in sv_types:
            continue
        start = int(record.intervalA.start) + 1  # 1-based
        stop = int(record.intervalA.end) + 1  # 1-based
        if bed_file_type == BED_FILE_TYPE.BEDPE:
            start = int(record.intervalA.end)  # 1-based
            stop = int(record.intervalB.end)  # 1-based
        zygosity = (None, None) if record.aux['zygosity'] is None else constants.ZYGOSITY_GT_VCF[record.aux['zygosity']]
        qual = int(float(record.aux['score'])*100)
        sv_len = stop - start
        if abs(sv_len) < min_len:
            continue
        if min_score is not None and qual < min_score:
            continue
        vcf_record = vcf_format_file.header.new_record(contig=str(record.intervalA.chr_name),
                                                       start=start,
                                                       stop=stop,
                                                       alleles=['N', record.get_sv_type(to_vcf_format=True)],
                                                       id=record.sv_type,
                                                       info={'SVTYPE': record.sv_type, 'SVLEN': sv_len},
                                                       qual=qual,
                                                       filter='.',
                                                       samples=[{'GT': zygosity}])
        vcf_out_file.write(vcf_record)
    vcf_out_file.close()
    os.remove(vcf_fname + ".format")


def main():
    parser = argparse.ArgumentParser(description='BED to VCF converter')
    parser.add_argument('--bed', help='Input BED file with SVs')
    parser.add_argument('--vcf', help='Output VCF')
    parser.add_argument('--fai', help='FAI index file', default=None)
    parser.add_argument('--sv_types', help='Types of SVs to keep', default=None)
    parser.add_argument('--min_score', help='Min score', default=None, type=float)
    parser.add_argument('--min_len', help='Min length', default=0, type=float)
    parser.add_argument('--ftype', help='BED file type', default=BED_FILE_TYPE.BEDPE)
    args = parser.parse_args()
    bed2vcf(args.bed, args.vcf, args.fai, args.ftype, args.sv_types, args.min_score)


if __name__ == '__main__':
    main()
