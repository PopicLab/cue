import engine.core as engine
import argparse
import torch.optim as optim
from torch.utils.data import DataLoader
import engine.config_utils as config_utils
import models.cue_net as models
import img.datasets as datasets
import torch
from utils import types
from enum import Enum

# ------ CLI ------
parser = argparse.ArgumentParser(description='RefiNet training')
parser.add_argument('--config', help='YAML training config file')
args = parser.parse_args()
# -----------------

# ------ Initialization ------
# load the model YAML configs / setup the experiment
torch.manual_seed(0)
config = config_utils.load_config(args.config)
PHASES = Enum('PHASES', 'TRAIN VALIDATE')

# ---------Training dataset--------
train_data = datasets.SVKeypointDataset(config, exclude_chrs=["chr1", "chr2", "chr3", "chr4", "chr5", "chr6", "chr16",
                                                              "chr17", "chr18"], store=False)
validation_data = datasets.SVKeypointDataset(config, include_chrs=["chr16", "chr17", "chr18"], store=False)

# ---------Data loaders--------
data_loaders = {PHASES.TRAIN: DataLoader(dataset=train_data, batch_size=config.batch_size, shuffle=False,
                                         collate_fn=datasets.collate_fn),
                PHASES.VALIDATE: DataLoader(dataset=validation_data, batch_size=config.batch_size, shuffle=False,
                                            collate_fn=datasets.collate_fn)}

# ---------Model--------
model = models.CueModelConfig(config).get_model()
if config.pretrained_model is not None:
    model.load_state_dict(torch.load(config.pretrained_model, config.device))
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.learning_rate_decay_interval,
                                               gamma=config.learning_rate_decay_factor)
model.to(config.device)

# ------ Training ------
epoch_metrics = types.NestedDict(types.NestedDict(types.NestedDict(float)))
for epoch in range(config.num_epochs):
    engine.train(model, optimizer, data_loaders[PHASES.TRAIN], config, epoch,
                 collect_data_metrics=False, classify=False)
    torch.save(model.state_dict(), "%s.epoch%d" % (config.model_path, epoch))
    engine.evaluate(model, data_loaders[PHASES.VALIDATE], config, config.device, config.epoch_dirs[epoch],
                    streaming=True, collect_data_metrics=False, given_ground_truth=True,
                    filters=False, coco=False)
    lr_scheduler.step()

torch.save(model.state_dict(), config.model_path)
