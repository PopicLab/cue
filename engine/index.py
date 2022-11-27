import engine.config_utils as config_utils
import argparse
from seq.aln_index import AlnIndex
import logging
import seq.utils as utils
from joblib import Parallel, delayed

# ------ CLI ------
parser = argparse.ArgumentParser(description='Generate the BAM alignment signal index')
parser.add_argument('--config', help='Dataset config')
args = parser.parse_args()
# -----------------

def index(chr_names):
    for chr_name in chr_names:
        AlnIndex.generate_or_load(chr_name, config)
    return True


config = config_utils.load_config(args.config, config_type=config_utils.CONFIG_TYPE.DATA)
chr_name_chunks, _ = utils.partition_chrs(config.chr_names, config.fai, config.n_cpus)
logging.info("Running on %d CPUs" % config.n_cpus)
logging.info("Chromosome lists processed by each process: " + str(chr_name_chunks))
_ = Parallel(n_jobs=config.n_cpus)(
    delayed(index)(chr_name_chunks[i]) for i in range(config.n_cpus))
