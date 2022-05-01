from collections import defaultdict
from collections import Counter
import logging
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.axes
from img.constants import TargetType


class DatasetStats:
    def __init__(self, output_prefix, classes):
        self.output_prefix = output_prefix
        self.classes = classes
        self.num_images = 0
        self.visible_svs_per_image = []
        self.all_svs_per_image = []
        self.type2area = defaultdict(list)
        self.type2counts = defaultdict(list)

    def update(self, target):
        if target is None:
            return
        self.num_images += 1
        if TargetType.labels not in target:
            return
        self.all_svs_per_image.append(len(target[TargetType.labels]))
        num_visible = 0
        type2counts = defaultdict(int)
        for i in range(len(target[TargetType.labels])):
            label = target[TargetType.labels][i].item()
            kps = target[TargetType.keypoints][i]
            visible = any([p[2] for p in kps])
            num_visible += visible
            if visible:
                self.type2area[label].append(target[TargetType.area][i].item())
                type2counts[label] += 1
        self.visible_svs_per_image.append(num_visible)
        for label, count in type2counts.items():
            self.type2counts[label].append(count)

    def batch_update(self, batch):
        for target in batch:
            self.update(target)

    def report(self):
        logging.info("Number of images: %d" % self.num_images)
        logging.info("Number of SVs per image:")
        logging.info(Counter(self.all_svs_per_image))
        logging.info("Number of visible SVs per image:")
        logging.info(Counter(self.visible_svs_per_image))
        for label, counts in self.type2counts.items():
            logging.info("Number of visible %s per image:" % self.classes[label])
            logging.info(Counter(counts))
        self.generate_plots()

    def generate_plots(self):
        palette = sns.color_palette("Set2")
        fig = plt.figure()
        plt.title("Number of SVs per image")
        self.plot_discrete_distribution(self.all_svs_per_image, palette[0], plt)
        plt.savefig("%ssvs_per_image.png" % self.output_prefix, format='png')
        plt.close(fig)

        fig = plt.figure()
        plt.title("Number of visible SVs per image")
        self.plot_discrete_distribution(self.visible_svs_per_image, palette[1], plt)
        plt.savefig("%ssvs_per_image_visible.png" % self.output_prefix, format='png')
        plt.close(fig)

        fig, axs = plt.subplots(len(self.classes)-1, 1, sharex=False, sharey=False)
        plt.subplots_adjust(hspace=1)
        axs = axs.ravel()
        for i, (label, counts) in enumerate(self.type2counts.items()):
            self.plot_discrete_distribution(counts, palette[2], axs[i])
            axs[i].set_title("Number of %ss per image" % self.classes[label])
            axs[i].set_xlabel('Number of SVs')
            axs[i].set_ylabel('Number of images')
        plt.savefig("%ssv_per_image_by_type.png" % self.output_prefix, format='png')
        plt.close(fig)

        fig, axs = plt.subplots(len(self.classes)-1, 1, figsize=(10, 10), sharex=False, sharey=False)
        plt.subplots_adjust(hspace=1)
        axs = axs.ravel()
        for i, (label, area) in enumerate(self.type2area.items()):
            sns.histplot(data=area, ax=axs[i], color=palette[3])
            axs[i].set_title("%s area distribution" % self.classes[label])
            axs[i].set_xlabel('SV area')
            axs[i].set_ylabel('Number of SVs')
        plt.savefig("%ssv_by_area.png" % self.output_prefix, format='png')
        plt.close(fig)

    @staticmethod
    def plot_discrete_distribution(data, color, plt_handle):
        xlabels, counts = np.unique(data, return_counts=True)
        plt_handle.bar(xlabels, counts, align='center', color=color)
        if issubclass(type(plt_handle), matplotlib.axes.SubplotBase):
            plt_handle.set_xticks(xlabels)
        else:
            plt_handle.xticks(xlabels)
