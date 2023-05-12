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
import logging
import img.datasets as datasets
from seq.aln_index import AlnIndex
import seq.utils as utils
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings("ignore")

def main():
    # ------ CLI ------
    parser = argparse.ArgumentParser(description='View an SV callset')
    parser.add_argument('--config', help='Dataset config')
    args = parser.parse_args()
    # -----------------

    def view(chr_names):
        for chr_name in chr_names:
            aln_index = AlnIndex.generate_or_load(chr_name, config)
            logging.info("Generating SV images for %s" % chr_name)
            dataset = datasets.SVBedScanner(config, config.interval_size, allow_empty=False, store=True,
                                            include_chrs=[chr_name], aln_index=aln_index)
            for _, target in dataset:
                continue
        return True


    config = config_utils.load_config(args.config, config_type=config_utils.CONFIG_TYPE.DATA)
    chr_name_chunks, _ = utils.partition_chrs(config.chr_names, config.fai, config.n_cpus)
    _ = Parallel(n_jobs=config.n_cpus)(delayed(view)(chr_name_chunks[i]) for i in range(config.n_cpus))


if __name__ == "__main__":
    main()
