from enum import Enum
import torch.nn as nn
import torch
from img import constants
from models import modules
from img import utils
from img.heatmap import SVKeypointHeatmapUtility

class CueModelConfig:
    NETWORK_TYPE = Enum("TYPE", 'HG ')

    def __init__(self, config):
        self.config = config

    def get_model(self):
        network_type = self.NETWORK_TYPE[self.config.model_architecture]
        return {
            self.NETWORK_TYPE.HG: self.hourglass,
        }[network_type]()

    def hourglass(self):
        return MultiSVHG(self.config)

class MultiSVHG(nn.Module):
    # Stacked hourglass network for SV breakpoint prediction
    # Implementation based on Newell et al human pose estimation models (ECCV 2016, NeurIPS 2017)
    # PoseNet (https://github.com/princeton-vl/pose-ae-train)

    def __init__(self, config):
        super(MultiSVHG, self).__init__()
        self.config = config
        self.heatmap_generator = SVKeypointHeatmapUtility(config.image_dim, num_kps_per_sv=config.num_keypoints,
                                                          num_sv_labels=config.num_classes-1, sigma=config.sigma,
                                                          stride=config.stride,
                                                          peak_threshold=config.heatmap_peak_threshold)
        self.hg_in_dim = 256
        self.hg_out_dim = self.heatmap_generator.num_heatmap_channels
        self.hg_expansion = 128
        self.hg_depth = 4
        self.hg_stack_size = 4
        self.backbone = modules.HourglassBackbone(self.config.n_signals, self.hg_in_dim)
        self.hg_stack = nn.ModuleList([nn.Sequential(
            modules.Hourglass(self.hg_depth, self.hg_in_dim, self.hg_expansion)
        ) for _ in range(self.hg_stack_size)])
        self.features = nn.ModuleList([nn.Sequential(
            modules.Residual(self.hg_in_dim, self.hg_in_dim),
            modules.Conv(self.hg_in_dim, self.hg_in_dim, kernel_size=1, pool=False, bn=True, relu=True)
        ) for _ in range(self.hg_stack_size)])
        self.outs = nn.ModuleList([modules.Conv(self.hg_in_dim, self.hg_out_dim, 1, pool=False, relu=False, bn=False)
                                   for _ in range(self.hg_stack_size)])
        self.merge_features = nn.ModuleList([
            modules.Conv(self.hg_in_dim, self.hg_in_dim, kernel_size=1, pool=False, relu=False, bn=False)
            for _ in range(self.hg_stack_size)])
        self.merge_preds = nn.ModuleList([
            modules.Conv(self.hg_out_dim, self.hg_in_dim, kernel_size=1, pool=False, relu=False, bn=False)
            for _ in range(self.hg_stack_size)])

    def forward(self, images, targets=None):
        images = utils.batch_images(images)
        x = self.backbone(images)
        stage_outputs = []
        for i in range(self.hg_stack_size):
            hg = self.hg_stack[i](x)
            feature = self.features[i](hg)
            stack_output = self.outs[i](feature)
            stage_outputs.append(stack_output)
            if i < self.hg_stack_size - 1:
                x = x + self.merge_preds[i](stack_output) + self.merge_features[i](feature)

        outputs = [{constants.TargetType.heatmaps: heatmaps} for heatmaps in stage_outputs[-1]]
        for output in outputs:
            self.heatmap_generator.heatmaps2predictions(output)
        if self.training:
            losses = {'loss_heatmaps': self.loss(stage_outputs, targets)}
            return losses, outputs
        return outputs

    def loss(self, stage_outputs, targets):
        for target in targets:
            self.heatmap_generator.keypoints2heatmaps(target)
        heatmaps_gt = torch.stack([t[constants.TargetType.heatmaps].to(self.config.device) for t in targets], dim=0)
        stage_outputs = torch.stack(stage_outputs, dim=0)
        stage_weights = [1] * stage_outputs.shape[0]
        loss = self.focal_loss(stage_outputs, heatmaps_gt, stage_weights=stage_weights)
        return loss

    def focal_loss(self, outputs, targets, gamma=1, stage_weights=None, alpha=0.1, beta=0.02, theta=0.01):
        # Focal L2 loss adapted from SimplePose (Li et al, AAAI 2020)
        dkt = torch.where(torch.ge(targets, theta), outputs - alpha, 1 - outputs - beta)
        factor = torch.abs(1. - dkt) ** gamma
        lkt = (outputs - targets) ** 2 * factor
        fl = lkt.sum(dim=(1, 2, 3, 4))
        weight_loss = [fl[i] * stage_weights[i] for i in range(len(stage_weights))]
        loss = sum(weight_loss) / sum(stage_weights)
        print('Focal L2 loss: ', fl.detach().cpu().numpy())
        return loss


