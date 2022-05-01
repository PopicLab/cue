import img.constants as constants
import yaml
from enum import Enum
import torch
from pathlib import Path
import logging
import sys
import os
import math
from random import randint
logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('tensorflow').setLevel(logging.WARNING)

CONFIG_TYPE = Enum("CONFIG_TYPE", 'TRAIN TEST DATA')


class Config:
    def __init__(self, config_file):
        self.config_file = config_file
        self.experiment_dir = str(Path(config_file).parent.resolve())
        self.devices = []
        if len(self.gpu_ids) > 0:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, self.gpu_ids))
        if torch.cuda.is_available():
            for i in range(len(self.gpu_ids)):
                self.devices.append(torch.device("cuda:%d" % i))
        else:
            self.devices.append(torch.device("cpu"))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tb_dir = self.experiment_dir + "/tb/"
        self.log_dir = self.experiment_dir + "/logs/"
        self.report_dir = self.experiment_dir + "/reports/"

        # setup the experiment directory structure
        Path(self.tb_dir).mkdir(parents=True, exist_ok=True)
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        Path(self.report_dir).mkdir(parents=True, exist_ok=True)

        # logging
        self.log_file = self.log_dir + 'main.log'
        # noinspection PyArgumentList
        logging.basicConfig(format='[%(levelname)s] %(message)s', level=logging.INFO,
                            handlers=[logging.FileHandler(self.log_file, mode='w'), logging.StreamHandler(sys.stdout)])

        # shared training and testing configs
        self.class_set = constants.SV_CLASS_SET[self.class_set]
        self.classes = constants.SV_CLASSES[self.class_set]
        self.num_classes = len(self.classes)
        self.signal_set = constants.SV_SIGNAL_SET[self.signal_set]
        self.n_signals = len(constants.SV_SIGNALS_BY_TYPE[self.signal_set]) 
        logging.info(self)

    def __str__(self):
        s = " ===== Config ====="
        s += "\n\tYAML config file: " + self.config_file
        s += "\n\tExperiment directory: " + self.experiment_dir
        s += "\n\tDevice: " + str(self.device)
        s += "\n\tMain LOG file: " + str(self.log_file) + "\n\t"
        return s


class TrainingConfig(Config):
    def __init__(self, config_file, **entries):
        self.__dict__.update(entries)
        super().__init__(config_file)
        self.image_dirs = [dataset_dir + "/images/" for dataset_dir in self.dataset_dirs]
        self.annotation_dirs = [dataset_dir + "/annotations/" for dataset_dir in self.dataset_dirs]
        self.model_path = self.experiment_dir + "/model.pt"
        self.epoch_dirs = []
        for epoch in range(self.num_epochs):
            output_dir = "%s/epoch%d/" % (self.report_dir, epoch)
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            self.epoch_dirs.append(output_dir)

    def __str__(self):
        s = super().__str__()
        s += '\n\t'.join("{}: {}".format(k, v) for k, v in self.__dict__.items())
        return s


class TestConfig(Config):
    def __init__(self, config_file, **entries):
        self.__dict__.update(entries)
        super().__init__(config_file)

    def __str__(self):
        s = super().__str__()
        s += '\n\t'.join("{}: {}".format(k, v) for k, v in self.__dict__.items())
        return s


class DatasetConfig:
    def __init__(self, config_file, **entries):
        self.__dict__.update(entries)
        self.config_file = config_file
        self.dataset_dir = str(Path(config_file).parent.resolve())
        self.info_dir = self.dataset_dir + "/info/"
        self.image_dir = self.dataset_dir + "/images/"
        self.annotation_dir = self.dataset_dir + "/annotations/"
        self.annotated_images_dir = self.dataset_dir + "/annotated_images/"
        self.class_set = constants.SV_CLASS_SET[self.class_set]
        self.classes = constants.SV_CLASSES[self.class_set]
        self.num_classes = len(self.classes)
        self.bam_type = constants.BAM_TYPE[self.bam_type]
        self.signal_set = constants.SV_SIGNAL_SET[self.signal_set]
        self.num_signals = len(constants.SV_SIGNALS_BY_TYPE[self.signal_set])
        self.uid = ''.join(["{}".format(randint(0, 9)) for _ in range(0, 10)])

        # setup the dataset directory structure
        Path(self.info_dir).mkdir(parents=True, exist_ok=True)
        Path(self.image_dir).mkdir(parents=True, exist_ok=True)
        for i in range(math.ceil(self.num_signals/3)):
            Path(self.image_dir + ("split%d" % i)).mkdir(parents=True, exist_ok=True)
        Path(self.annotation_dir).mkdir(parents=True, exist_ok=True)
        Path(self.annotated_images_dir).mkdir(parents=True, exist_ok=True)

        # logging
        self.log_file = self.info_dir + 'main.log'
        # noinspection PyArgumentList
        logging.basicConfig(format='[%(levelname)s] %(message)s', level=logging.DEBUG,
                            handlers=[logging.FileHandler(self.log_file, mode='w'), logging.StreamHandler(sys.stdout)])
        logging.info(self)

    def __str__(self):
        s = "==== Dataset ====\n"
        s += '\n\t'.join("{}: {}".format(k, v) for k, v in self.__dict__.items())
        return s


def load_config(fname, config_type=CONFIG_TYPE.TRAIN):
    """
    Load a YAML configuration file
    """
    with open(fname) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    if config_type == CONFIG_TYPE.TRAIN:
        return TrainingConfig(fname, **config)
    elif config_type == CONFIG_TYPE.TEST:
        return TestConfig(fname, **config)
    else:
        return DatasetConfig(fname, **config)


