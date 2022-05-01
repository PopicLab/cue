import torchvision.transforms as transforms
from img import constants
from img.constants import TargetType
from _collections import defaultdict
import numpy as np
from scipy.ndimage.filters import maximum_filter, gaussian_filter
import cv2
import torch
import img.utils as utils


class SVKeypointHeatmapUtility:
    def __init__(self, image_dim, num_kps_per_sv=1, num_sv_labels=3, sigma=20, stride=4, peak_threshold=0.4):
        self.heatmap_stride = stride
        self.sigma = sigma
        self.peak_threshold = peak_threshold
        self.num_kps_per_sv = num_kps_per_sv
        self.heatmap_dim = int(image_dim / self.heatmap_stride)
        self.num_heatmap_channels = num_sv_labels * num_kps_per_sv
        self.refine = True

    def keypoints2heatmaps(self, target):
        # generate keypoint heatmaps: one heatmap channel per SV type and keypoint type (e.g. upper left corner)
        heatmaps = np.zeros((self.heatmap_dim, self.heatmap_dim, self.num_heatmap_channels))

        # collect keypoints for each type of SV
        label2keypoints = defaultdict(list)
        for label, keypoints in zip(target[TargetType.labels], target[TargetType.keypoints]):
            if label != constants.LABEL_BACKGROUND:
                label2keypoints[label.item()].append(np.array(keypoints.cpu()))

        for sv_label in label2keypoints.keys():
            for kp_idx in range(self.num_kps_per_sv):
                heatmap_idx = (sv_label - 1) * self.num_kps_per_sv + kp_idx  # -1 to account for background label
                keypoints = [sv_kps[kp_idx] for sv_kps in label2keypoints[sv_label]]
                for point in keypoints:
                    if point[2] > 0:  # if visible
                        current_heatmap = heatmaps[:, :, heatmap_idx]
                        heatmaps[:, :, heatmap_idx] = self.add_gaussian_at_point(
                            point[:2], current_heatmap, self.sigma,
                            self.heatmap_dim, self.heatmap_dim, self.heatmap_stride)
        target[TargetType.heatmaps] = transforms.ToTensor()(heatmaps).float()

    def heatmaps2predictions(self, target):
        kp_list_per_sv_type = []
        num_kps = 0
        heatmaps = target[TargetType.heatmaps].permute(1, 2, 0).detach().cpu().numpy()
        labels = []
        keypoints_out = []
        scores = []
        for idx in range(self.num_heatmap_channels):
            sv_label = idx + 1
            heatmap = heatmaps[:, :, idx]
            peaks = self.find_peaks(heatmap, self.peak_threshold)
            keypoints = np.zeros((len(peaks), 4))  # x, y, score, id
            for i, peak in enumerate(peaks):
                if self.refine:
                    pw = 4
                    x_min, y_min = np.maximum(0, peak - pw)
                    x_max, y_max = np.minimum(np.array(heatmap.T.shape) - 1, peak + pw)
                    patch = heatmap[y_min:y_max + 1, x_min:x_max + 1]
                    location_of_patch_center = utils.upscale_keypoints(peak[::-1] - [y_min, x_min], self.heatmap_stride)
                    patch_upscaled = cv2.resize(patch, None, fx=self.heatmap_stride, fy=self.heatmap_stride,
                                                interpolation=cv2.INTER_CUBIC)
                    patch_upscaled = gaussian_filter(patch_upscaled, sigma=5)
                    location_of_max = np.unravel_index(patch_upscaled.argmax(), patch_upscaled.shape)
                    peak_score = patch_upscaled[location_of_max]
                    refined_center = (location_of_max - location_of_patch_center)
                else:
                    refined_center = [0, 0]
                    peak_score = heatmap[tuple(peak[::-1])]
                refined_kp = utils.upscale_keypoints(peak, self.heatmap_stride) + refined_center[::-1]
                keypoints[i, :] = tuple(x for x in refined_kp) + (peak_score, num_kps)
                num_kps += 1
                labels.append(sv_label)
                keypoints_out.append([[refined_kp[0], refined_kp[1], 1]])
                scores.append(min(1, peak_score))
            kp_list_per_sv_type.append(keypoints)
        target[TargetType.labels] = torch.as_tensor(labels, dtype=torch.int64)
        target[TargetType.keypoints] = torch.as_tensor(keypoints_out, dtype=torch.float32)
        target[TargetType.scores] = torch.as_tensor(scores, dtype=torch.float32)
        return kp_list_per_sv_type

    @staticmethod
    def add_gaussian_at_point(point, heatmap, sigma, grid_y, grid_x, stride):
        x, y = np.meshgrid([i for i in range(int(grid_y))], [i for i in range(int(grid_x))])
        offset = stride / 2.0 - 0.5
        x = x * stride + offset
        y = y * stride + offset
        d = (x - point[0]) ** 2 + (y - point[1]) ** 2
        exponent = d / 2.0 / sigma / sigma
        mask = exponent <= 4.6052
        gauss_peak = np.multiply(mask, np.exp(-exponent))
        heatmap += gauss_peak
        heatmap[heatmap > 1.0] = 1.0
        return heatmap

    @staticmethod
    def find_peaks(heatmap, threshold):
        keypoints_binary = (maximum_filter(heatmap, size=8, mode='constant') == heatmap) * (heatmap > threshold)
        return np.array(np.nonzero(keypoints_binary)[::-1]).T

