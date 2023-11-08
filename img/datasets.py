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


import time
from torch.utils.data import Dataset, IterableDataset
import matplotlib.image as img
import os
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data.dataloader import default_collate
from seq.aln_index import AlnIndex
from img import constants
from seq.intervals import GenomeInterval, GenomeIntervalPair
from img.plotting import heatmap_with_filters_demo, heatmap_np, annotate, save, save_channel_overlay
import img.utils as utils
import seq.io as io
import torch
import logging
from seq.genome_scanner import TargetIntervalScanner, SlidingWindowScanner
from img.constants import TargetType
import math
import itertools
from scipy.ndimage import convolve
from sys import exit


class SVStreamingDataset(IterableDataset):
    def __init__(self, config, interval_size, step_size=None, include_chrs=None, exclude_chrs=None,
                 allow_empty=False, store=False, aln_index=None, view_mode=False,
                 generate_empty=False, convert_to_empty=False, remove_annotation=False):
        super().__init__()
        self.config = config
        self.interval_size = interval_size
        self.step_size = step_size
        self.include_chrs = include_chrs
        self.exclude_chrs = exclude_chrs if exclude_chrs is not None else []
        self.allow_empty = allow_empty
        self.generate_empty = generate_empty
        self.convert_to_empty = convert_to_empty
        self.view_mode = view_mode
        self.store = store
        self.convolve_nbrs = False
        self.apply_filters = False
        self.aln_index = aln_index
        self.annotate = config.bed is not None
        self.remove_annotation = remove_annotation
        self.ground_truth = io.BedRecordContainer(config.bed) if self.annotate else None
        self.transform = transforms.Compose([transforms.ToTensor()])  # TODO(viq): normalize
        self.signal_set = config.signal_set
        self.n_signals = len(constants.SV_SIGNALS_BY_TYPE[self.signal_set])
        self.n_channels = ((self.n_signals + 2) // 3) * 3

    def get_ground_truth_target(self, interval_pair):
        labels = []
        boxes = []
        keypoints = []
        records = self.ground_truth.overlap(interval_pair.intervalA)
        if len(records) == 0:
            if not self.generate_empty:
                return None
            else:
                return self.get_background_target(interval_pair)
        else:
            any_visible_keypoints = 0
            for record in records:
                if self.config.class_set in constants.SV_ZYGOSITY_SETS:
                    labels.append(constants.SV_LABELS[record.get_sv_type_with_zyg()])
                elif self.config.class_set == constants.SV_CLASS_SET.BINARY:
                    labels.append(constants.SV_LABELS[constants.CLASS_SV])
                else:
                    labels.append(constants.SV_LABELS[record.get_sv_type()])
                x_min = utils.bp_to_pixel(record.intervalA.start, interval_pair.intervalA, self.config.heatmap_dim)
                x_max = utils.bp_to_pixel(record.intervalB.start, interval_pair.intervalA,
                                          self.config.heatmap_dim)
                y_min = self.config.heatmap_dim - x_max  # - 1
                y_max = self.config.heatmap_dim - x_min  # - 1
                # adjust y to account for an interval shift in the pair
                # TODO: this works for intervals on the same chr
                shit_pos = 2 * interval_pair.intervalB.start - interval_pair.intervalA.start
                delta_pixels = utils.bp_to_pixel(shit_pos, interval_pair.intervalB, self.config.heatmap_dim)
                y_min += delta_pixels
                y_max += delta_pixels
                box_x_min = max(x_min - self.config.bbox_padding, 0)
                box_x_max = min(x_max + self.config.bbox_padding, self.config.heatmap_dim)  # - 1
                box_y_min = min(max(y_min - self.config.bbox_padding, 0), self.config.heatmap_dim)
                box_y_max = max(min(y_max + self.config.bbox_padding, self.config.heatmap_dim), 0)  # - 1
                boxes.append([box_x_min, box_y_min, box_x_max, box_y_max])
                sv_keypoints = [[x_min, y_min, (x_min > 0 and y_min > 0)]]
                any_visible_keypoints += (x_min > 0 and y_min > 0)
                if self.config.num_keypoints == 2:
                    sv_keypoints.append([x_max, y_max, (x_max < self.config.heatmap_dim
                                                        and y_max < self.config.heatmap_dim)])
                keypoints.append(sv_keypoints)
                logging.info("%s: [%d %d %d %d], [%d %d %d %d], %s %s" % (record, x_min, x_max, y_min, y_max,
                                                                        box_x_min, box_x_max, box_y_min, box_y_max,
                                                                        record.get_sv_type(), record.get_sv_type_with_zyg()))
                logging.debug(keypoints)
                assert box_x_min <= box_x_max, "Given: %d %d " % (box_x_min, box_x_max)
                assert box_y_min <= box_y_max, "Given: %d %d " % (box_y_min, box_y_max)
            if not any_visible_keypoints and self.convert_to_empty:
                return self.get_background_target(interval_pair)
                #elif not self.generate_empty:
                #    return None
        # create the target tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])  # TODO: fix by 1
        target = {TargetType.boxes: boxes,
                  TargetType.keypoints: keypoints,
                  TargetType.labels: labels,
                  TargetType.area: area,
                  TargetType.gloc: torch.as_tensor(interval_pair.to_list(self.aln_index.chr_index))}
        return target

    def get_background_target(self, interval_pair):
        labels = torch.as_tensor([constants.SV_LABELS[constants.CLASS_BACKGROUND]], dtype=torch.int64)
        keypoints = torch.as_tensor([[[0, 0, False]]], dtype=torch.float32)
        boxes = torch.as_tensor([[0, 0, 0, 0]], dtype=torch.float32)
        area = torch.as_tensor([0], dtype=torch.float32)
        target = {TargetType.boxes: boxes,
                  TargetType.keypoints: keypoints,
                  TargetType.labels: labels,
                  TargetType.area: area,
                  TargetType.gloc: torch.as_tensor(interval_pair.to_list(self.aln_index.chr_index))}
        return target

    def get_empty_target(self):
        target = {TargetType.boxes: torch.as_tensor([[]], dtype=torch.float32),
                  TargetType.keypoints: torch.as_tensor([[]], dtype=torch.float32),
                  TargetType.labels: torch.as_tensor([], dtype=torch.int64),
                  TargetType.area: torch.as_tensor([], dtype=torch.float32)}
        return target

    def get_genome_iterator(self):
        if self.config.scan_target_intervals:
            return TargetIntervalScanner(self.aln_index, self.config.interval_size, self.config.step_size)
        return SlidingWindowScanner(self.aln_index, self.interval_size, self.step_size, shift_size=self.config.shift_size)

    def make_image(self, interval_pair):
        # generate the heatmaps for each signal channel
        start = time.time()
        image = np.zeros((self.config.heatmap_dim, self.config.heatmap_dim, self.n_channels))
        total_counts = 0
        llrr_counts = None
        for i, signal in enumerate(constants.SV_SIGNALS_BY_TYPE[self.signal_set]):
            if signal in constants.SV_SIGNAL_SCALAR:
                counts = self.aln_index.scalar_apply(signal, interval_pair.intervalA, interval_pair.intervalB)
                vmin = -self.config.signal_vmax[signal]
            else:
                if signal == constants.SVSignals.LLRR_VS_LR:
                    assert llrr_counts is not None
                    rd_counts = self.aln_index.scalar_apply(constants.SVSignals.RD, interval_pair.intervalA,
                                                            interval_pair.intervalB, op=min)
                    if self.convolve_nbrs:
                        kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
                        nbr_counts = convolve(llrr_counts, kernel, mode='constant')
                        counts = (llrr_counts + nbr_counts) / (rd_counts + nbr_counts)
                    else:
                        
                        if np.sum(rd_counts) > 0:
                            if not np.isfinite(llrr_counts).all() or not np.isfinite(rd_counts).all():
                                logging.error(f"DON'T IGNORE THE WARNING!\nExiting...")
                                exit(1)
                            else:
                                np.seterr(divide='ignore', invalid='ignore')
                                counts = llrr_counts // (llrr_counts + rd_counts)
                        else:
                            counts = llrr_counts
                        counts[np.isnan(counts)] = 0
                else:
                    counts = self.aln_index.intersect(signal, interval_pair.intervalA, interval_pair.intervalB,
                                                      signal in [constants.SVSignals.LLRR, constants.SVSignals.RL])
                vmin = 0
            total_counts += np.sum(counts)
            if self.view_mode:
                logging.debug("%s: %d %d" % (signal, np.max(counts), np.min(counts)))
            if signal == constants.SVSignals.RD and np.max(counts) == 0 and np.min(counts) == 0:
                return None, 0
            if self.apply_filters and signal in [constants.SVSignals.SR_RP, constants.SVSignals.LLRR,
                                                 constants.SVSignals.RL, constants.SVSignals.LLRR_VS_LR]:
                image[:, :, i] = heatmap_with_filters_demo(counts, img_size=self.config.heatmap_dim,
                                                      vmin=vmin, vmax=self.config.signal_vmax[signal], cvresize=self.view_mode)
            else:
                image[:, :, i] = heatmap_np(counts, img_size=self.config.heatmap_dim,
                                            vmin=vmin, vmax=self.config.signal_vmax[signal], cvresize=self.view_mode)
            if signal == constants.SVSignals.LLRR:
                llrr_counts = counts  # used in combination with RD

        # debug
        if self.view_mode:
            save_channel_overlay(image, self.config.image_dir + "%s_%s.png" % (self.config.uid, str(interval_pair)))
        logging.debug("Image: %d %d" % (time.time() - start, total_counts))
        return image, total_counts

    def save_image(self, image, idx, interval_pair):
        img_fname = "%s_%d_%s.png" % (self.config.uid, idx, str(interval_pair))
        for i in range(0, self.n_channels, 3):
            save(image[:, :, i:i + 3], self.config.image_dir + ("split%d/" % (i // 3)) + img_fname)

    def __iter__(self):
        for idx, interval_pair in enumerate(self.get_genome_iterator()):
            target = None
            image = None
            total_counts = 0
            if not self.annotate:
                image, total_counts = self.make_image(interval_pair)
                if total_counts == 0:
                    continue
            if self.annotate:
                target = self.get_ground_truth_target(interval_pair)
                if target is None and not self.allow_empty:  # skip images with no events
                    continue
                image, total_counts = self.make_image(interval_pair)
                if total_counts == 0:  # skip images with no signal
                    continue
                if self.store:
                    self.save_image(image, idx, interval_pair)
                    if self.remove_annotation:
                        target = None 
                    torch.save(target, self.config.annotation_dir + "%s_%d_%s.target" % (self.config.uid, idx,
                                                                                         str(interval_pair)))
                    # debug
                    #overlay = save_channel_overlay(image, self.config.image_dir + "%s_%d_%s.png" %
                    #                               (self.config.uid, idx, str(interval_pair)))
                    annotated_image = annotate(image[:,:,:3], target, self.config.classes)
                    save(annotated_image,
                         self.config.annotated_images_dir + "%s_%d_%s.png" % (self.config.uid, idx, str(interval_pair)))
            elif self.store and total_counts != 0:
                self.save_image(image, idx, interval_pair)
            image = image[:, :, :self.n_signals]
            if self.transform is not None:
                image = self.transform(image).float()
            if target is None:
                target = {}
            image, target = utils.downscale_tensor(image, self.config.image_dim, target)
            target[TargetType.image_id] = torch.tensor([idx])
            target[TargetType.gloc] = torch.as_tensor(interval_pair.to_list(self.aln_index.chr_index))
            yield image, target


class SVStaticDataset(Dataset):
    def __init__(self, image_dir, annotation_dir, target_image_dim, signal_set, signal_set_origin, dataset_id=0):
        super().__init__()
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.target_image_dim = target_image_dim
        self.annotations = list(sorted(os.listdir(annotation_dir)))
        self.images = {}
        self.n_signals = len(constants.SV_SIGNALS_BY_TYPE[signal_set])
        self.channel_ids = constants.SV_SIGNAL_SET_CHANNEL_IDX[signal_set_origin][signal_set]
        n_signals_origin = len(constants.SV_SIGNALS_BY_TYPE[signal_set_origin])
        n_channels_origin = ((n_signals_origin + 2) // 3) * 3
        for i in range(math.ceil(n_channels_origin / 3)):
            self.images[i] = list(sorted(os.listdir(image_dir + ("split%d" % i))))
            assert len(self.images[i]) == len(self.annotations), "Invalid dataset directory structure %d %d" % \
                                                                 (len(self.images[i]), len(self.annotations))
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.dataset_id = dataset_id

    def __len__(self):
        return min(len(self.images[0]), 1000000)

    def __getitem__(self, index):
        image = []
        for i in range(len(self.images)):
            img_path = os.path.join(self.image_dir + ("split%d" % i), self.images[i][index])
            image.append(img.imread(img_path)[:, :, :3])
        image = np.concatenate(image, axis=2)
        image = np.take(image, self.channel_ids, axis=2)
        target_path = os.path.join(self.annotation_dir, self.annotations[index])
        target = torch.load(target_path)
        if target is None:
            target = {TargetType.keypoints: torch.as_tensor([], dtype=torch.float32),
                      TargetType.labels: torch.as_tensor([], dtype=torch.int64)}
        target[TargetType.image_id] = torch.tensor([index])
        if self.transform is not None:
            image = self.transform(image)
        image, target = utils.downscale_tensor(image, self.target_image_dim, target)
        target[TargetType.dataset_id] = torch.tensor([self.dataset_id])
        return image, target


class SVBedScanner(SVStreamingDataset):
    def __init__(self, config, interval_size, step_size=None, include_chrs=None, exclude_chrs=None,
                 allow_empty=False, store=False, padding=50000, aln_index=None):
        super().__init__(config, interval_size, step_size, include_chrs, exclude_chrs, allow_empty, store,
                         aln_index=aln_index, view_mode=True)
        self.padding = padding

    def get_genome_iterator(self):
        for rec in self.ground_truth:
            if rec.intervalA.chr_name in self.exclude_chrs or \
                    (self.include_chrs is not None and rec.intervalA.chr_name not in self.include_chrs):
                continue
            interval = GenomeInterval(rec.intervalA.chr_name, rec.intervalA.start, rec.intervalB.start)
            interval = interval.pad(self.aln_index.chr_index, self.padding)
            logging.info("%s %s %s" % (rec.to_bedpe_aux(), rec.get_sv_type_with_zyg(), interval))
            if len(interval) < 600000:  # limit the size of the heatmap to be generated
                yield GenomeIntervalPair(interval, interval)
            else:  # shift the y axis to show the breakpoints only
                print("Record too large to display fully: ", rec, len(interval))
                x_interval = GenomeInterval(rec.intervalA.chr_name, rec.intervalA.start, rec.intervalA.end)
                y_interval = GenomeInterval(rec.intervalB.chr_name, rec.intervalB.start, rec.intervalB.end)
                x_interval = x_interval.pad(self.aln_index.chr_index, 2 * self.padding)
                y_interval = y_interval.pad(self.aln_index.chr_index, 2 * self.padding)
                yield GenomeIntervalPair(x_interval, y_interval)


class SVKeypointDataset(SVStreamingDataset):
    def __init__(self, config, include_chrs=None, exclude_chrs=None, store=False):
        self.chr_index = io.load_faidx(config.fai, all=True)
        super().__init__(config, 1, store=store, include_chrs=include_chrs, exclude_chrs=exclude_chrs,
                         generate_empty=False, convert_to_empty=False)
        self.no_empty_patches = False #True
        self.annotate = True
        self.padding = config.padding
        self.max_patches = 10

    def get_genome_iterator(self):
        chr_names = list(self.chr_index.chr_names()) if self.include_chrs is None else self.include_chrs
        for chr_name in chr_names:
            if chr_name in self.exclude_chrs:
                continue
            logging.info("Loading annotations for %s" % chr_name)
            annotation_dir = self.config.annotation_dirs[0]
            annotations = [fn for fn in os.listdir(annotation_dir) if "%s_" % chr_name in fn]
            if len(annotations) == 0:
                continue
            self.aln_index = AlnIndex.load(AlnIndex.get_fname(chr_name, self.config))
            for ann_fname in annotations:
                ann_path = os.path.join(annotation_dir, ann_fname)
                annotation = torch.load(ann_path)
                if annotation is None:
                    continue
                gloc = GenomeIntervalPair.from_list(annotation[TargetType.gloc].tolist(), self.aln_index.chr_index)
                # TODO: temp workaround for varying size intervals
                if len(gloc.intervalA) != len(gloc.intervalB):
                    continue
                n_keypoint_patches = 0
                for keypoint in annotation[TargetType.keypoints]:
                    x, y, v = keypoint.tolist()[0]  # assuming a one-keypoint per SV dataset
                    if v != constants.KP_VISIBLE:  # skip invisible keypoints
                        continue
                    y = self.config.heatmap_dim_ann - y
                    x_bp = utils.pixel_to_bp(x, gloc.intervalA, self.config.heatmap_dim_ann)
                    y_bp = utils.pixel_to_bp(y, gloc.intervalB, self.config.heatmap_dim_ann)
                    x_interval = GenomeInterval(gloc.intervalA.chr_name, x_bp, x_bp)
                    y_interval = GenomeInterval(gloc.intervalB.chr_name, y_bp, y_bp)
                    x_interval = x_interval.pad(self.aln_index.chr_index, self.padding, rand=True)
                    y_interval = y_interval.pad(self.aln_index.chr_index, self.padding, rand=True)
                    if len(x_interval) != len(y_interval) or len(x_interval) != 2*self.padding:
                        continue
                    yield GenomeIntervalPair(x_interval, y_interval)
                    n_keypoint_patches += 1
                if self.no_empty_patches:
                    continue
                # generate potentially empty patches
                n_any_patches = 0
                for i, j in itertools.product(range(0, len(gloc.intervalA), self.padding),
                                              range(0, len(gloc.intervalB), self.padding)):
                    patch_x_bp = gloc.intervalA.start + i + self.padding // 2
                    patch_y_bp = gloc.intervalB.start + j + self.padding // 2
                    x_interval = GenomeInterval(gloc.intervalA.chr_name, patch_x_bp, patch_x_bp)
                    y_interval = GenomeInterval(gloc.intervalB.chr_name, patch_y_bp, patch_y_bp)
                    x_interval = x_interval.pad(self.aln_index.chr_index, self.padding, rand=False)
                    y_interval = y_interval.pad(self.aln_index.chr_index, self.padding, rand=False)
                    if len(x_interval) != len(y_interval) or len(x_interval) != 2*self.padding:
                        continue
                    yield GenomeIntervalPair(x_interval, y_interval)
                    n_any_patches += 1
                    if n_any_patches > n_keypoint_patches:
                        break


def aux_collate(batch):
    core_batch = []
    aux = []
    for item in batch:
        core_batch.append(item[:-1])
        aux.append(item[-1])
    return default_collate(core_batch), aux


def collate_fn(batch):
    return zip(*batch)
