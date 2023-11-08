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


from img import constants
from img.constants import TargetType
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
from matplotlib import cm
import seaborn as sns
import io
import cv2
import copy
import scipy.signal
from scipy.ndimage.filters import maximum_filter, gaussian_filter
from scipy.ndimage.morphology import generate_binary_structure


def heatmap_np(overlap2D, img_size=1000, vmin=0, vmax=100, cvresize=False):
    overlap2D = overlap2D.transpose()
    overlap2D = np.flip(overlap2D, 0)
    if cvresize:
        x = cv2.resize(overlap2D, dsize=(img_size, img_size), interpolation=cv2.INTER_CUBIC)
    elif img_size <= overlap2D.shape[0]:
        x = overlap2D.reshape(img_size, overlap2D.shape[0] // img_size, img_size, overlap2D.shape[0] // img_size).sum(3).sum(1)
    else:
        x = overlap2D.repeat(img_size//overlap2D.shape[0], axis=0).repeat(img_size//overlap2D.shape[0], axis=1)
    x = np.clip(x, vmin, vmax).astype(float)
    if vmin < 0:
        x = 0.5 + 0.5*x/vmax
    else:
        x = x/vmax
    return x

def heatmap_with_filters(overlap2D, img_size=1000, vmin=0, vmax=100, cvresize=False):
    neighborhood = generate_binary_structure(2, 2)
    heatmap = heatmap_np(overlap2D, img_size=img_size, vmin=vmin, vmax=vmax, cvresize=cvresize)
    heatmap = maximum_filter(heatmap, size=20, footprint=neighborhood, mode='nearest')
    return gaussian_filter(heatmap, sigma=5, mode="nearest")

def heatmap_with_filters_demo(overlap2D, img_size=1000, vmin=0, vmax=100, cvresize=False):
    neighborhood = generate_binary_structure(2, 2)
    heatmap = heatmap_np(overlap2D, img_size=img_size, vmin=vmin, vmax=vmax, cvresize=cvresize)
    heatmap = maximum_filter(heatmap, size=200, footprint=neighborhood, mode='nearest')
    h2 = heatmap.copy()
    h2[h2 > 0.1] = 0
    heatmap += 3*h2
    heatmap = scipy.signal.convolve2d(heatmap, np.ones((20,20), dtype=int),
                              mode='same', boundary='fill', fillvalue=0)
    return gaussian_filter(heatmap, sigma=5, mode="nearest")

def heatmap(overlap2D, img_size=1000, dpi=1000, fig_name=None, vmin=0, vmax=100):
    overlap2D = overlap2D.transpose()
    fig = plt.figure(figsize=(img_size/dpi, img_size/dpi), dpi=dpi)
    plt.gca().set_axis_off()
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    cmap = cm.get_cmap('bone_r', 1000)
    ax = sns.heatmap(overlap2D, cmap=cmap, cbar=False, square=True,
                     xticklabels=False, yticklabels=False, vmin=vmin, vmax=vmax)
    ax.invert_yaxis()
    buffer = io.BytesIO()
    save_to = fig_name if fig_name is not None else buffer
    plt.tight_layout(pad=0)
    fig.savefig(save_to, dpi=dpi, format='png', pad_inches=0)
    plt.close(fig)
    if fig_name is None:
        buffer.seek(0)
        return img.imread(buffer)
    else:
        return None


def save_separate_channels(image, fig_name, dpi=1000):
    fig = plt.figure(figsize=(image.shape[0] / dpi, image.shape[1] / dpi), dpi=dpi)
    plt.gca().set_axis_off()
    for i in range(image.shape[2]):
        plt.imsave("%s.ch%d.png" % (fig_name, i), image[:, :, i], cmap="Spectral_r")
    plt.close(fig)

def save_channel_overlay(image, fig_name, dpi=1000):
    fig = plt.figure(figsize=(image.shape[0] / dpi, image.shape[1] / dpi), dpi=dpi)
    plt.gca().set_axis_off()
    plt.imshow(image[:, :, 1], cmap="Spectral_r", alpha=1)
    h = image[:, :, 0].copy()
    h[h < 0.3] = np.nan
    plt.imshow(h, cmap='Spectral_r', alpha=0.7)
    h = image[:, :, 2].copy()
    h[h < 0.005] = np.nan
    plt.imshow(h, cmap='Spectral_r', alpha=0.5)
    plt.margins(0, 0)
    plt.tight_layout(pad=0)
    plt.savefig(fig_name + ".overlay.png", dpi=dpi, format='png', pad_inches=0)
    buffer = io.BytesIO()
    plt.tight_layout(pad=0)
    fig.savefig(buffer, dpi=dpi, format='png', pad_inches=0)
    plt.close(fig)
    buffer.seek(0)
    save_separate_channels(image, fig_name, dpi=dpi)
    return img.imread(buffer)

def save(image, fig_name, dpi=1000, no_border=True):
    fig = plt.figure(figsize=(image.shape[0]/dpi, image.shape[1]/dpi), dpi=dpi)
    if no_border:
        plt.gca().set_axis_off()
    plt.imshow(image[:, :, :3])
    plt.margins(0, 0)
    plt.tight_layout(pad=0)
    plt.savefig(fig_name, dpi=dpi, format='png', pad_inches=0)
    plt.close(fig)

def save_n_channel(image, fig_name, dpi=1000, no_border=True):
    fig = plt.figure(figsize=(image.shape[0]/dpi, image.shape[1]/dpi), dpi=dpi)
    if no_border:
        plt.gca().set_axis_off()
    plt.imshow(image)
    plt.margins(0, 0)
    plt.tight_layout(pad=0)
    plt.savefig(fig_name, dpi=dpi, format='png', pad_inches=0)
    plt.close(fig)

def annotate(image, target, classes, display_boxes=True, display_classes=True, color=(153 / 255, 153 / 255, 10 / 255)):
    if target is None:
        return image
    image_dim = image.shape[0]
    font_scale = max(0.8, int(image_dim * 4 / 1000))
    thickness = max(1, int(image_dim * 2 / 1000))
    image = np.ascontiguousarray(image[:, :, :3])
    display_classes = display_classes and TargetType.labels in target and len(target[TargetType.labels])
    display_boxes = display_boxes and TargetType.boxes in target and len(target[TargetType.boxes])
    if display_boxes:
        for i, box in enumerate(target[TargetType.boxes]):
            cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, thickness)

    if TargetType.keypoints in target:
        for i, points in enumerate(target[TargetType.keypoints]):
            filtered = False
            for p in points:
                if p[2] != constants.KP_FILTERED:
                    cv2.circle(image, (int(p[0]), int(p[1])), int(image_dim/100), color=color, thickness=-thickness)
                else:
                    cv2.circle(image, (int(p[0]), int(p[1])), int(image_dim/100), color=(128/255, 128/255, 128/255),
                               thickness=-thickness)
                    filtered = True
            if display_classes:
                label_color = color if not filtered else (128/255, 128/255, 128/255)
                cv2.putText(image, classes[target[TargetType.labels][i]] + ("-FILT" if filtered else ""),
                            (int(points[0][0]) + 5, int(points[0][1]) + 5),
                            cv2.FONT_HERSHEY_PLAIN, font_scale, label_color, thickness, lineType=cv2.LINE_AA)
    return image


def overlay_keypoint_heatmaps(image, target):
    keypoint_cmap = copy.copy(plt.cm.get_cmap('viridis'))
    keypoint_cmap.set_bad(alpha=0)
    heatmaps = target[TargetType.heatmaps].permute(1, 2, 0).detach().cpu().numpy()
    heatmaps = cv2.resize(heatmaps, (image.shape[0], image.shape[1]))
    if len(heatmaps.shape) > 2:
        for i in range(heatmaps.shape[2]):
            heatmap = heatmaps[:, :, i]
            heatmap[heatmap < 0.4] = np.nan  # insert 'bad' values into the heatmap to make them transparent
            plt.imshow(heatmap, vmin=0., vmax=1., cmap=keypoint_cmap, alpha=0.5)
    else:
        heatmaps[heatmaps < 0.4] = np.nan  # insert 'bad' values into the heatmap to make them transparent
        plt.imshow(heatmaps, vmin=0., vmax=1., cmap=keypoint_cmap, alpha=0.5)

def plot_images(images, targets, indices, classes, fig_name, targets2=None, max_dim_display=6, show_annotation=True):
    fig = plt.figure(figsize=(10, 10))
    plt.subplots_adjust(hspace=.4)
    rows = min(int(np.sqrt(len(indices))), max_dim_display)
    cols = min(int(np.sqrt(len(indices))), max_dim_display)
    for i in range(rows * cols):
        plt.subplot(rows, cols, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        image = images[indices[i]].permute(1, 2, 0).cpu().numpy()
        if image.shape[2] == 1:
            image = cv2.merge((image,image,image))
        if show_annotation:
            image = annotate(image, targets[indices[i]], classes, display_boxes=True, color=(0, 76 / 255, 153 / 255))
            if targets2 is not None:
                image = annotate(image, targets2[indices[i]], classes, display_boxes=True, display_classes=True,
                                 color=(153 / 255, 153 / 255, 10 / 255))
        plt.imshow(image, cmap='seismic')
        if TargetType.heatmaps in targets[indices[i]]:
            overlay_keypoint_heatmaps(image, targets[indices[i]])
        img_id = targets[indices[i]][TargetType.image_id].item()
        if TargetType.gloc in targets[indices[i]]:
            img_id = targets[indices[i]][TargetType.gloc]
        plt.xlabel("ID: %s" % str(img_id), fontsize=8)
    plt.savefig(fig_name)
    plt.close(fig)


def plot_heatmap_channels(image, heatmaps, fig_name):
    image = np.squeeze(image.permute(1, 2, 0).numpy()[:, :, :1])
    heatmaps = heatmaps.permute(1, 2, 0).detach().numpy()
    heatmaps = cv2.resize(heatmaps, (image.shape[0], image.shape[1]))
    keypoint_cmap = copy.copy(plt.cm.get_cmap('viridis'))
    keypoint_cmap.set_bad(alpha=0)
    fig = plt.figure(figsize=(10, 10))
    plt.subplots_adjust(hspace=.4)
    cols = 3  # TODO: update for a different class set
    rows = int(heatmaps.shape[2] / cols)
    for i in range(rows * cols):
        plt.subplot(rows, cols, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        heatmap = heatmaps[:, :, i]
        plt.imshow(image, cmap="Spectral_r")
        plt.imshow(heatmap, vmin=0., vmax=1., cmap=keypoint_cmap, alpha=0.5)
    plt.savefig(fig_name)
    plt.close(fig)

def plot_heatmap(heatmap, fig_name):
    keypoint_cmap = copy.copy(plt.cm.get_cmap('viridis'))
    keypoint_cmap.set_bad(alpha=0)
    fig = plt.figure(figsize=(10, 10))
    plt.imshow(heatmap, vmin=0., vmax=1., cmap=keypoint_cmap, alpha=1)
    plt.savefig(fig_name)
    plt.close(fig)

def plot_images_with_predictions(images, labels, predictions, display_classes, indices, image_info, fig_name,
                                 max_dim_display=6, is_labeled=True):
    fig = plt.figure(figsize=(10, 10))
    plt.subplots_adjust(hspace=.4)
    rows = min(int(np.sqrt(len(indices))), max_dim_display)
    cols = min(int(np.sqrt(len(indices))), max_dim_display)
    for i in range(rows * cols):
        plt.subplot(rows, cols, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        image = images[indices[i]].permute(1, 2, 0).numpy()
        plt.imshow(image)
        if predictions[0][indices[i]] != constants.LABEL_BACKGROUND:
            predicted_landmark = predictions[1][indices[i]]
            plt.scatter(predicted_landmark[0], predicted_landmark[1], c='r')
        if is_labeled and labels[0][indices[i]] != constants.LABEL_BACKGROUND:
            true_landmark = labels[1][indices[i]]
            plt.scatter(true_landmark[0], true_landmark[1], c='b')
        true_sv_type = display_classes[labels[0][indices[i]]] if is_labeled else 'UNK'
        predicted_sv_type = display_classes[predictions[0][indices[i]].item()]
        img_info = image_info[indices[i]]
        plt.xlabel("%s(%s)\n[%s]" % (predicted_sv_type, true_sv_type, img_info), fontsize=8)
    plt.savefig(fig_name)
    plt.close(fig)


def plot_dataset_classes(config, images, targets):
    for sv_label in range(len(config.classes)):
        idx = [i for i, target in enumerate(targets) if sv_label in target[TargetType.labels]]
        plot_images(images, targets, idx, config.classes,
                    fig_name="%s/img.%s.png" % (config.report_dir, config.classes[sv_label]))
