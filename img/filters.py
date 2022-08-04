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


from img.constants import TargetType
import img.constants as constants
from torchvision.ops import nms
import torch


def bp_to_pixel(genome_pos, genome_interval, pixels_in_interval):
    bp_per_pixel = len(genome_interval) / pixels_in_interval
    pos_pixels = int((genome_pos - genome_interval.start) / bp_per_pixel)
    return pos_pixels


def pixel_to_bp(pixel_pos, genome_interval, pixels_in_interval):
    bp_per_pixel = len(genome_interval) / pixels_in_interval
    bp = int(round(pixel_pos * bp_per_pixel)) + genome_interval.start
    return bp


def get_bbox(kp, interval_pair_tensor, image_dim):
    x_min, y_min = kp[0:2]
    delta_bp = interval_pair_tensor[1][1] - interval_pair_tensor[0][1]
    delta_pixels = delta_bp/((interval_pair_tensor[0][2] - interval_pair_tensor[0][1]) / image_dim)
    y_max = image_dim - x_min + delta_pixels
    x_max = image_dim - (y_min - delta_pixels)
    return [x_min, y_min, x_max, y_max]

def bbox_overlap(bbox1, bbox2):
    x_min1, x_max1, y_min1, y_max1 = bbox1
    x_min2, x_max2, y_min2, y_max2 = bbox2
    return x_max1 >= x_min2 and x_max2 >= x_min1 and y_max1 >= y_min2 and y_max2 >= y_min1


def box_overlap(bbox1, bbox2):
    dx = min(bbox1[2], bbox2[2]) - max(bbox1[0], bbox2[0])
    dy = min(bbox1[3], bbox2[3]) - max(bbox1[1], bbox2[1])
    if (dx >= 0) and (dy >= 0):
        area1 = (bbox1[3] - bbox1[1]) * (bbox1[2] - bbox1[0])
        area2 = (bbox2[3] - bbox2[1]) * (bbox2[2] - bbox2[0])
        return float((dx * dy) / min(area1, area2))
    else:
        return 0


def nms2D_iom(output, iom_threshold=0.1):
    _, indices = output[TargetType.scores].sort(descending=True)
    for i in range(len(indices)):
        sv_idx1 = indices[i]
        if output[TargetType.keypoints][sv_idx1][0][2] == constants.KP_FILTERED:
            continue
        for j in range(i+1, len(indices)):
            sv_idx2 = indices[j]
            if output[TargetType.keypoints][sv_idx2][0][2] == constants.KP_FILTERED:
                continue
            if box_overlap(output[TargetType.boxes][sv_idx1], output[TargetType.boxes][sv_idx2]) >= iom_threshold:
                output[TargetType.keypoints][sv_idx2][0][2] = constants.KP_FILTERED


def nms2D_iou(output, image_dim, iou_threshold=0.1):
    # find SV keypoints with the highest scores and remove other candidates with high overlap
    boxes = []
    for sv_idx in range(len(output[TargetType.labels])):
        kp = output[TargetType.keypoints][sv_idx].tolist()[0]  # top left corner
        output[TargetType.keypoints][sv_idx][0][2] = constants.KP_FILTERED
        boxes.append(get_bbox(kp, output[TargetType.gloc], image_dim))
    if not boxes:
        return
    output[TargetType.boxes] = torch.tensor(boxes)
    keep = nms(output[TargetType.boxes], output[TargetType.scores], iou_threshold)
    for sv_idx in keep:
        output[TargetType.keypoints][sv_idx][0][2] = constants.KP_VISIBLE


def filter_keypoints(outputs, config):
    for output in outputs:
        nms2D_iou(output, config.image_dim)
        nms2D_iom(output)
