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

import engine
import engine.core as core
import argparse
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import engine.config_utils as config_utils
import models.cue_net as models
import img.datasets as datasets
import torch
from enum import Enum
import logging
import img.constants as constants

torch.manual_seed(0)

def main():
    print("*********************************")
    print("*  cue (%s): training mode  *" % engine.__version__)
    print("*********************************")

    # ------ CLI ------
    parser = argparse.ArgumentParser(description='Cue model training')
    parser.add_argument('--config', help='Training config')
    parser.add_argument('--data_config', help='(Optional) Dataset config for streaming', default=None)
    args = parser.parse_args()
    # -----------------

    # ------ Initialization ------
    # load the model configs / setup the experiment
    config = config_utils.load_config(args.config)
    PHASES = Enum('PHASES', 'TRAIN VALIDATE')

    # ---------Training dataset--------
    streaming = False
    if args.data_config is None:
        # static (pre-generated data)
        input_datasets = []
        for dataset_id in range(len(config.dataset_dirs)):
            input_datasets.append(datasets.SVStaticDataset(config.image_dirs[dataset_id],
                                                        config.annotation_dirs[dataset_id],
                                                        config.image_dim, config.signal_set,
                                                        constants.SV_SIGNAL_SET[config.signal_set_origin],
                                                        dataset_id))
        dataset = torch.utils.data.ConcatDataset(input_datasets)
        validation_size = int(config.validation_ratio * len(dataset))
        train_size = len(dataset) - validation_size
        train_data, validation_data = random_split(dataset, [train_size, validation_size])
        images, targets = next(iter(DataLoader(dataset=dataset, batch_size=min(len(dataset), 400), shuffle=True,
                                            collate_fn=datasets.collate_fn)))
    else:
        # streaming (on-the-fly data generation)
        # data divided into training / validation using chromosomes (since dataset length unknown)
        # data_config.chr_names defines the split (exclude/include)
        streaming = True
        data_config = config_utils.load_config(args.data_config, config_type=config_utils.CONFIG_TYPE.DATA)
        train_data = datasets.SVStreamingDataset(data_config, interval_size=data_config.interval_size[0],
                                                step_size=data_config.step_size[0],
                                                exclude_chrs=data_config.chr_names,
                                                store=True, allow_empty=False)
        validation_data = datasets.SVStreamingDataset(data_config, interval_size=data_config.interval_size[0],
                                                    step_size=data_config.step_size[0],
                                                    include_chrs=data_config.chr_names,
                                                    store=True, allow_empty=False)

    # ---------Data loaders--------
    data_loaders = {PHASES.TRAIN: DataLoader(dataset=train_data, batch_size=config.batch_size, shuffle=True,
                                            collate_fn=datasets.collate_fn),
                    PHASES.VALIDATE: DataLoader(dataset=validation_data, batch_size=config.batch_size, shuffle=False,
                                                collate_fn=datasets.collate_fn)}
    logging.info("Size of train set: %d; validation set: %d" % (len(data_loaders[PHASES.TRAIN]),
                                                                len(data_loaders[PHASES.VALIDATE])))

    # ---------Model--------
    model = models.CueModelConfig(config).get_model()
    if config.pretrained_model is not None:
        model.load_state_dict(torch.load(config.pretrained_model, config.device))
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.learning_rate_decay_interval,
                                                gamma=config.learning_rate_decay_factor)
    model.to(config.device)

    # ------ Training ------
    for epoch in range(config.num_epochs):
        core.train(model, optimizer, data_loaders[PHASES.TRAIN], config, epoch, collect_data_metrics=(epoch == 0))
        torch.save(model.state_dict(), "%s.epoch%d" % (config.model_path, epoch))
        core.evaluate(model, data_loaders[PHASES.VALIDATE], config, config.device, config.epoch_dirs[epoch],
                        collect_data_metrics=(epoch == 0), given_ground_truth=True, filters=False)
        lr_scheduler.step()

    torch.save(model.state_dict(), config.model_path)


if __name__ == "__main__":
    main()
