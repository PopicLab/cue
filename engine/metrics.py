import torch
from img import constants
import logging
from collections import defaultdict

class MetricTracker:
    def __init__(self, report_interval, prefix=""):
        self.metrics = defaultdict(float)
        self.n_batch_updates = 0
        self.report_interval = report_interval
        self.prefix = prefix

    def batch_update(self, loss_dict):
        for loss_metric, value in loss_dict.items():
            self.metrics[loss_metric] += value.item()
        self.n_batch_updates += 1
        self.report()

    def batch_update_accuracy(self, outputs, targets):
        correct = 0
        for output, target in zip(outputs, targets):
            if target[constants.TargetType.labels][0].item() == \
                    output[constants.TargetType.labels][0].round().int().item():
                correct += 1
        self.metrics['accuracy'] += 100 * float(correct / len(outputs))

    def get_average(self, metric):
        return self.metrics[metric] / self.n_batch_updates

    def report(self):
        if self.n_batch_updates % self.report_interval != 0:
            pass
        metric_string = " | ".join(['%s: %.3f' % (metric, self.get_average(metric))
                                    for metric, value in self.metrics.items()])
        logging.info('%s [%d] %s ' % (self.prefix, self.n_batch_updates, metric_string))


def accuracy(outputs, labels):
    _, predicted = torch.max(outputs.data, 1)
    correct = (predicted == labels).sum()
    acc = 100 * correct.float() / labels.shape[0]
    return acc
