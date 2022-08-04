import torch.nn.functional as F
from img.constants import TargetType
import img.constants as constants
import seq.intervals as intervals
import seq.io as io
import seq.filters as filters
import logging

def upscale_keypoints(keypoints, ratio):
    return keypoints * ratio

def downscale_target(target, ratio):
    if TargetType.boxes in target:
        target[TargetType.boxes] = target[TargetType.boxes]//ratio
        target[TargetType.area] = target[TargetType.area] // (ratio * ratio)
    if TargetType.keypoints in target:
        for i in range(len(target[TargetType.keypoints])):
            for j in range(len(target[TargetType.keypoints][i])):
                target[TargetType.keypoints][i][j][:2] = target[TargetType.keypoints][i][j][:2]//ratio
    return target


def downscale_tensor(image, to_image_dim, target=None):
    ratio = image.shape[1] / to_image_dim
    if target is not None:
        target = downscale_target(target, ratio)
    return F.interpolate(image.unsqueeze(0), size=(to_image_dim, to_image_dim)).squeeze(0), target


def downscale_image(image, to_image_dim, target=None):
    image_dim_orig = image.shape[0]
    assert image_dim_orig >= to_image_dim, "Input image size cannot be smaller than the target image size"
    assert image_dim_orig % to_image_dim == 0, "Input image size must be a multiple of the target image size"
    ratio = int(image_dim_orig / to_image_dim)
    if target is not None:
        target = downscale_target(target, ratio)
    return image.reshape((to_image_dim, ratio, to_image_dim, ratio, 3)).max(3).max(1), target


def bp_to_pixel(genome_pos, genome_interval, pixels_in_interval):
    bp_per_pixel = len(genome_interval) / pixels_in_interval
    pos_pixels = int((genome_pos - genome_interval.start) / bp_per_pixel)
    return pos_pixels


def pixel_to_bp(pixel_pos, genome_interval, pixels_in_interval):
    bp_per_pixel = len(genome_interval) / pixels_in_interval
    bp = int(round(pixel_pos * bp_per_pixel)) + genome_interval.start
    return bp

def batch_images(images):
    batch_shape = [len(images)] + list(images[0].shape)
    batched_imgs = images[0].new_full(batch_shape, 0)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
    return batched_imgs


def img_to_svs(target, data_config, chr_index):
    if len(target[TargetType.labels]) > 20:
        return [], []
    svs = []
    filtered_svs = []
    genome_interval_pair = intervals.GenomeIntervalPair.from_list(target[TargetType.gloc].tolist(), chr_index)
    n_high_score_svs = 0
    for sv_idx in range(len(target[TargetType.labels])):
        sv_type = data_config.classes[target[TargetType.labels][sv_idx]]
        x, y, v = target[TargetType.keypoints][sv_idx].tolist()[0]  # top left corner
        y = data_config.image_dim - y
        x_bp = pixel_to_bp(x, genome_interval_pair.intervalA, data_config.image_dim)
        y_bp = pixel_to_bp(y, genome_interval_pair.intervalB, data_config.image_dim)
        logging.debug("img2sv: %s %s %d %d %d %d" % (genome_interval_pair, sv_type, x, y, x_bp, y_bp))
        sv_interval_pair = intervals.GenomeIntervalPair(
            intervals.GenomeInterval(genome_interval_pair.intervalA.chr_name, x_bp, x_bp + 1),
            intervals.GenomeInterval(genome_interval_pair.intervalB.chr_name, y_bp, y_bp + 1))
        sv_type, zygosity = io.BedRecord.parse_sv_type_with_zyg(sv_type)
        aux = {'score': target[TargetType.scores][sv_idx].item(),
               'zygosity': zygosity}
        if aux['score'] == 100:
            n_high_score_svs += 1
        sv_record = io.BedRecord(sv_type, sv_interval_pair.intervalA, sv_interval_pair.intervalB, aux)
        if filters.invalid(sv_record):
            continue
        if v == constants.KP_FILTERED:
            filtered_svs.append(sv_record)
        else:
            svs.append(sv_record)
    if n_high_score_svs > 10:
        return [], []
    return svs, filtered_svs
