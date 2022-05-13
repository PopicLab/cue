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
from seq.intervals import GenomeInterval, GenomeIntervalPair
from img.constants import TargetType
import torch
import torchvision.transforms as transforms
import img.utils as utils
from img import plotting


class SVKeypointRefinery:
    def __init__(self, refine_net, device, padding, refine_image_dim):
        self.refine_net = refine_net
        self.device = device
        self.image_generator = None
        self.bam_index = None
        self.padding = padding
        self.refine_image_dim = refine_image_dim
        self.idx = 0

    def refine_input(self, sv_call):
        x_bp = sv_call.intervalA.start
        y_bp = sv_call.intervalB.start
        x_interval = GenomeInterval(sv_call.intervalA.chr_name, x_bp, x_bp)
        y_interval = GenomeInterval(sv_call.intervalB.chr_name, y_bp, y_bp)
        x_interval = x_interval.pad(self.bam_index.chr_index, self.padding)
        y_interval = y_interval.pad(self.bam_index.chr_index, self.padding)
        interval = GenomeIntervalPair(x_interval, y_interval)
        image = self.image_generator.make_image(interval)[0]
        image = transforms.ToTensor()(image).float()
        image, _ = utils.downscale_tensor(image, self.refine_image_dim, {})
        self.idx += 1
        return image, interval

    def classify_call(self, sv_call):
        image, _ = self.refine_input(sv_call)
        images = [transforms.ToTensor()(image).float().to(self.device)]
        output = self.refine_net(images)[0]
        label_prob = output[constants.TargetType.labels][0].item()
        label = int(round(label_prob))
        if label == constants.LABEL_BACKGROUND:
            print("Filtered as background: ", sv_call, sv_call.aux['score'], label_prob)
        else:
            sv_call.aux['score'] = label_prob
        return label != constants.LABEL_BACKGROUND 

    def refine_call(self, sv_call):
        image, kp_interval = self.refine_input(sv_call)
        images = [image.to(self.device)]
        output = self.refine_net(images)[0]
        output[TargetType.image_id] = torch.tensor([self.idx])
        output[TargetType.gloc] = torch.as_tensor(kp_interval.to_list(self.image_generator.aln_index.chr_index))
        plotting.plot_images(images, [output], range(len(images)), self.image_generator.config.classes,
                             fig_name="%s/predictions.image%s.png" % (self.image_generator.config.report_dir, sv_call))
        if len(output[TargetType.labels]) > 1 or len(output[TargetType.labels]) == 0:
            # TODO: logic for multiple predictions
            print("Multiple preditions for call ", sv_call)
            return
   
        refined_kp = output[TargetType.keypoints].to(torch.device("cpu")).tolist()[0][0]
        x, y, _ = refined_kp
        y = self.refine_image_dim - y
        x_bp_r = utils.pixel_to_bp(x, kp_interval.intervalA, self.refine_image_dim)
        y_bp_r = utils.pixel_to_bp(y, kp_interval.intervalB, self.refine_image_dim)
        if abs(y_bp_r - x_bp_r) < 2000:
            print("Prediction too small for call ", sv_call)
            return 
        print("Refined %d %d to %d %d %d %d %s" % (sv_call.intervalA.start, sv_call.intervalB.start, x_bp_r, y_bp_r,
                                                   x, y, kp_interval))
        sv_call.intervalA.start = x_bp_r
        sv_call.intervalA.end = x_bp_r + 1
        sv_call.intervalB.start = y_bp_r
        sv_call.intervalB.end = y_bp_r + 1
