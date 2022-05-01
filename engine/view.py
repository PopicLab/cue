import engine.config_utils as config_utils
import argparse
import img.datasets as datasets
from seq.aln_index import AlnIndex
import seq.utils as utils
from joblib import Parallel, delayed


# ------ CLI ------
parser = argparse.ArgumentParser(description='View an SV callset')
parser.add_argument('--config', help='Dataset config')
args = parser.parse_args()
# -----------------

def view(chr_names):
    for chr_name in chr_names:
        aln_index = AlnIndex.generate_or_load_chr(config.bam, chr_name, config.fai, config.bin_size,
                                                  config.signal_mapq, config.signal_set, config.signal_set_origin,
                                                  config.bam_type)
        dataset = datasets.SVBedScanner(config, config.interval_size[0], allow_empty=False, store=True,
                                        include_chrs=[chr_name], aln_index=aln_index)
        for _, target in dataset:
            continue
    return True

config = config_utils.load_config(args.config, config_type=config_utils.CONFIG_TYPE.DATA)
chr_name_chunks, _ = utils.partition_chrs(config.chr_names, config.fai, config.n_cpus)
_ = Parallel(n_jobs=config.n_cpus)(delayed(view)(chr_name_chunks[i]) for i in range(config.n_cpus))
