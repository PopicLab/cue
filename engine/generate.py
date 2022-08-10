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


import engine.config_utils as config_utils
import argparse
import img.datasets as datasets
from img.data_metrics import DatasetStats
from seq.aln_index import AlnIndex
import logging
import seq.utils as utils
from joblib import Parallel, delayed
import os
import torch

# ------ CLI ------
parser = argparse.ArgumentParser(description='Generate an SV image dataset')
parser.add_argument('--config', help='Dataset config')
args = parser.parse_args()
# -----------------

def generate(chr_names):
    logging.root.setLevel(logging.INFO)
    # generates images/annotations for the specified list of chromosomes
    for chr_name in chr_names:
        aln_index = AlnIndex.generate_or_load_chr(config.bam, chr_name, config.fai, config.bin_size, config.signal_mapq,
                                                  config.signal_set, config.signal_set_origin, config.bam_type)
        dataset = datasets.SVStreamingDataset(config, config.interval_size[0], config.step_size[0], allow_empty=config.allow_empty,
                                              store=config.store_img, include_chrs=[chr_name], aln_index=aln_index, remove_annotation=config.empty_annotation)
        chr_stats = DatasetStats("%s/%s" % (config.info_dir, chr_name), classes=config.classes)
        for _, target in dataset:
            chr_stats.update(target)
        chr_stats.report()
    return True


config = config_utils.load_config(args.config, config_type=config_utils.CONFIG_TYPE.DATA)
chr_name_chunks, _ = utils.partition_chrs(config.chr_names, config.fai, config.n_cpus)
logging.info("Running on %d CPUs" % config.n_cpus)
logging.info("Chromosome lists processed by each process: " + str(chr_name_chunks))
_ = Parallel(n_jobs=config.n_cpus)(
    delayed(generate)(chr_name_chunks[i]) for i in range(config.n_cpus))

# generate stats for the entire dataset
stats = DatasetStats("%s/%s" % (config.info_dir, "full"), classes=config.classes)
targets = list(os.listdir(config.annotation_dir))
for target_fname in targets:
    target_path = os.path.join(config.annotation_dir, target_fname)
    target = torch.load(target_path)
    stats.update(target)
stats.report()

