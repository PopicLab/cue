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
import engine.core as engine
import argparse
from torch.utils.data import DataLoader
import torch
import models.cue_net as models
import logging
import seq.io as io
import img.datasets as datasets
from img.refinery import SVKeypointRefinery
from img.datasets import SVStreamingDataset
import img.utils as utils
import seq.filters as sv_filters
from seq.aln_index import AlnIndex
from joblib import Parallel, delayed
import seq.utils as seq_utils
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")


# ------ CLI ------
parser = argparse.ArgumentParser(description='SV calling functionality')
parser.add_argument('--data_config', help='Data config')
parser.add_argument('--model_config', help='Trained model config')
parser.add_argument('--refine_config', help='Trained refine model config', default=None)
parser.add_argument('--skip_inference', action='store_true', help='Do not re-run image-based inference', default=False)
args = parser.parse_args()

# load the configs
config = config_utils.load_config(args.model_config, config_type=config_utils.CONFIG_TYPE.TEST)
data_config = config_utils.load_config(args.data_config, config_type=config_utils.CONFIG_TYPE.DATA)
refine_config = None
if args.refine_config is not None:
    refine_config = config_utils.load_config(args.refine_config)
given_ground_truth = data_config.bed is not None  # (benchmarking mode)

def call(device, chr_names, uid):
    # runs SV calling on the specified device for the specified list of chromosomes
    # load the pre-trained model on the specified device
    model = models.MultiSVHG(config)
    model.load_state_dict(torch.load(config.model_path, device))
    model.to(device)
    logging.root.setLevel(logging.getLevelName(config.logging_level))
    logging.info("Loaded model: %s on %s" % (config.model_path, str(device)))

    # process each chromosome, loaded as a separate dataset
    for chr_name in chr_names:
        predictions_dir = "%s/predictions/%s.%s/" % (config.report_dir, uid, chr_name)
        Path(predictions_dir).mkdir(parents=True, exist_ok=True)
        aln_index = AlnIndex.generate_or_load_chr(data_config.bam, chr_name, data_config.fai,
                                                  data_config.bin_size, data_config.signal_mapq,
                                                  data_config.signal_set, data_config.signal_set_origin,
                                                  data_config.bam_type)
        dataset = SVStreamingDataset(data_config, interval_size=interval_size, step_size=step_size, store=False,
                                     include_chrs=[chr_name], allow_empty=True, aln_index=aln_index)
        data_loader = DataLoader(dataset=dataset, batch_size=config.batch_size, shuffle=False,
                                 collate_fn=datasets.collate_fn)
        logging.info("Generating SV predictions for %s" % chr_name)
        predictions = engine.evaluate(model, data_loader, config, device, output_dir=predictions_dir,
                                      collect_data_metrics=True, given_ground_truth=given_ground_truth)
        torch.save(predictions, "%s/predictions.pkl" % predictions_dir)
    return True



# ------ Image-based discovery ------
n_procs = len(config.devices)
if not n_procs:  # temp workaround when no GPUs are available
    n_procs = data_config.n_cpus
    for _ in range(n_procs):
        config.devices.append(torch.device("cpu"))
chr_name_chunks, chr_names = seq_utils.partition_chrs(data_config.chr_names, data_config.fai, n_procs)
logging.info("Running on %d CPUs/GPUs" % n_procs)
logging.info("Chromosome lists processed by each process: " + str(chr_name_chunks))
outputs_per_scan = []
for interval_size in data_config.interval_size:
    for step_size in data_config.step_size:
        scan_id = len(outputs_per_scan)
        if not args.skip_inference:
            _ = Parallel(n_jobs=n_procs)(
                delayed(call)(config.devices[i], chr_name_chunks[i], scan_id) for i in range(n_procs))
        outputs = []
        for chr_name in chr_names:
            predictions_dir = "%s/predictions/0.%s/" % (config.report_dir, chr_name)
            logging.debug("Loading: ", predictions_dir)
            predictions_per_chr = torch.load("%s/predictions.pkl" % predictions_dir)
            outputs.extend(predictions_per_chr)
        outputs_per_scan.append(outputs)

# ------ Genome-based post-processing ------
chr_index = io.load_faidx(data_config.fai)
candidates_per_scan = []
for outputs in outputs_per_scan:
    candidates = []
    filtered_candidates = []
    for output in outputs:
        svs, filtered_svs = utils.img_to_svs(output, data_config, chr_index)
        candidates.extend(svs)
        filtered_candidates.extend(filtered_svs)
    candidates = sv_filters.filter_svs(candidates, data_config.blacklist_bed, filtered_candidates)
    candidates_per_scan.append(candidates)

sv_calls = candidates_per_scan[0]
for i in range(1, len(candidates_per_scan)):
    sv_calls = sv_filters.merge_sv_candidates(sv_calls, candidates_per_scan[i])

candidate_out_bed_file = "%s/candidate_svs.bed" % config.report_dir
io.write_bed(candidate_out_bed_file, sv_calls)

# ------ NN-aided breakpoint refinement ------
if refine_config is not None and refine_config.pretrained_model is not None:
    def refine(device, chr_names):
        refinet = models.CueModelConfig(refine_config).get_model()
        refinet.load_state_dict(torch.load(refine_config.pretrained_model, refine_config.device))
        refinet.to(device)
        refinet.eval()
        refinery = SVKeypointRefinery(refinet, device, refine_config.padding, refine_config.image_dim)
        sv_calls = io.bed2sv_calls(candidate_out_bed_file)
        for chr_name in chr_names:
            refinery.bam_index = AlnIndex.generate_or_load_chr(refine_config.bam, chr_name, refine_config.fai,
                                                               refine_config.bin_size, refine_config.signal_mapq,
                                                               refine_config.signal_set, refine_config.signal_set_origin,
                                                               refine_config.bam_type)
            refinery.image_generator = SVStreamingDataset(refine_config, interval_size=None, store=False,
                                                          allow_empty=True, aln_index=refinery.bam_index)
            chr_calls = []
            for sv_call in sv_calls:
                if sv_call.intervalA.chr_name == chr_name:
                    refinery.refine_call(sv_call)
                    chr_calls.append(sv_call)
            chr_out_bed_file = "%s/svs.%s.bed" % (config.report_dir, chr_name)
            io.write_bed(chr_out_bed_file, chr_calls)
        return True
    refine(config.devices[0], chr_names)

    sv_calls = []
    for chr_name in chr_names:
        chr_sv_calls = io.bed2sv_calls("%s/svs.%s.bed" % (config.report_dir, chr_name))
        sv_calls.extend(chr_sv_calls)

# ------ IO ------
# write SV calls to file (bed and vcf)
out_bed_file = "%s/svs.bed" % config.report_dir
io.write_bed(out_bed_file, sv_calls)
io.bed2vcf(out_bed_file, "%s/svs.vcf" % config.report_dir, data_config.fai)
