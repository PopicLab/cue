import copy
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import img.constants as constants
import matplotlib.pyplot as plt
import math
import logging

class CocoKeypointEvaluator:
    def __init__(self, labels, num_keypoints, classes, output_dir):
        self.classes = classes
        self.coco_eval = COCOeval(iouType="keypoints")
        self.coco_eval.params.catIds = labels
        self.coco_eval.params.kpt_oks_sigmas = np.array([.5] * num_keypoints)
        self.coco_eval.params.areaRng = [[0, 1e5 ** 2], [1 ** 2, 16 ** 2], [16 ** 2, 64 ** 2], [64 ** 2, 1e5 ** 2]]
        self.coco_eval.params.areaRngLbl = ['all', 'small', 'medium', 'large']
        self.img_ids = []
        self.eval_imgs = []
        self.empty_image_ids = []  # ids of images with no objects
        self.output_dir = output_dir

    def batch_update(self, predictions, batch):
        self.coco_eval.cocoGt = convert_to_coco(batch)
        predictions = convert_results_to_coco(predictions)
        self.coco_eval.cocoDt = self.coco_eval.cocoGt.loadRes(predictions) if predictions else COCO()
        self.coco_eval.params.imgIds = self.coco_eval.cocoGt.getImgIds()
        self.img_ids.extend(self.coco_eval.params.imgIds)
        self.coco_eval.evaluate()
        self.eval_imgs.extend(self.coco_eval.evalImgs)

    def report(self):
        eval_images_ordered = []
        eval_images_by_type = {}
        for e in self.eval_imgs:
            if e is None:
                continue
            eval_images_by_type[e['image_id'], e['category_id'], tuple(e['aRng'])] = e

        self.coco_eval.params.imgIds = self.img_ids
        for catId in self.coco_eval.params.catIds:
            for areaRng in self.coco_eval.params.areaRng:
                for imgId in self.coco_eval.params.imgIds:
                    if (imgId, catId, tuple(areaRng)) in eval_images_by_type:
                        eval_images_ordered.append(eval_images_by_type[imgId, catId, tuple(areaRng)])
                    else:
                        eval_images_ordered.append(None)

        self.coco_eval.evalImgs = eval_images_ordered
        self.coco_eval._paramsEval = copy.deepcopy(self.coco_eval.params)
        self.coco_eval.accumulate()
        self.summarize()
        self.plot_roc()

    def summarize(self):
        # adapted from pycocotools.cocoeval
        def _summarize(show_precision=True, iouThr=None, areaRng='all', maxDets=20, show_cats=True):
            p = self.coco_eval.params
            iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
            titleStr = 'Average Precision' if show_precision else 'Average Recall'
            typeStr = '(AP)' if show_precision else '(AR)'
            iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
                if iouThr is None else '{:0.2f}'.format(iouThr)

            aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
            if show_precision:
                # dimension of precision: [TxRxKxAxM]
                s = self.coco_eval.eval['precision']
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:, :, :, aind, mind]
            else:
                # dimension of recall: [TxKxAxM]
                s = self.coco_eval.eval['recall']
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:, :, aind, mind]
            if len(s[s > -1]) == 0:
                mean_s = -1
            else:
                mean_s = np.mean(s[s > -1])
            logging.info(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))

            # per category stats
            if show_cats and mean_s > -1:
                cat_ids = self.coco_eval.params.catIds
                if show_precision:
                    for i in range(len(cat_ids)):
                        logging.info('\t{0}: {1}'.format(self.classes[cat_ids[i]], np.mean(s[:, :, i, :])))
                else:
                    for i in range(len(cat_ids)):
                        logging.info('\t{0}: {1}'.format(self.classes[cat_ids[i]], np.mean(s[:, i, :])))
            return mean_s

        assert self.coco_eval.eval
        _summarize()
        _summarize(iouThr=.5)
        _summarize(iouThr=.75)
        _summarize(iouThr=.95)
        _summarize(areaRng='medium')
        _summarize(areaRng='large')
        _summarize(show_precision=False)
        _summarize(show_precision=False, iouThr=.5)
        _summarize(show_precision=False, iouThr=.75)
        _summarize(show_precision=False, iouThr=.95)
        _summarize(show_precision=False, areaRng='medium')
        _summarize(show_precision=False, areaRng='large')

    def plot_roc(self):
        ps = self.coco_eval.eval['precision']
        rs = self.coco_eval.params.recThrs
        iou_thresholds = self.coco_eval.params.iouThrs
        color_map = plt.cm.get_cmap('hsv', len(iou_thresholds) + 1)
        fig, axs = plt.subplots(2, math.ceil(len(self.coco_eval.params.areaRngLbl)/2), figsize=(10, 10), edgecolor='k',
                                sharex=False, sharey=False)
        axs = axs.ravel()
        for i, area_label in enumerate(self.coco_eval.params.areaRngLbl):
            axs[i].set_title(area_label)
            axs[i].set_xlabel('Recall')
            axs[i].set_ylabel('Precision')
            axs[i].set_xlim(0, 1.)
            axs[i].set_ylim(0, 1.)
            axs[i].set_xticks(np.arange(0, 1.1, 0.1))
            axs[i].set_yticks(np.arange(0, 1.1, 0.1))

            area_ps = ps[..., i, 0]
            aps = [ps_.mean() for ps_ in area_ps]
            ps_curve = [
                ps_.mean(axis=1) if ps_.ndim > 1 else ps_ for ps_ in area_ps
            ]
            ps_curve.insert(0, np.zeros(ps_curve[0].shape))
            for t, iou_thr in enumerate(iou_thresholds):
                axs[i].plot(rs, ps_curve[t + 1], color=color_map(t), marker='o', linewidth=0.5,
                            label="IoU=%.2f:%.3f" % (iou_thr, aps[t]))
                #axs[i].fill_between(rs, ps_curve[t], ps_curve[t + 1], color=color_map(t),
                #                label=str('[{:.3f}'.format(aps[t]) + ']' + str(iou_thr)))
            axs[i].legend()
        #plt.show()
        fig.savefig(self.output_dir + '/ROC.png')
        plt.close(fig)

def convert_results_to_coco(outputs):
    coco_results = []
    for output in outputs:
        if len(output["keypoints"]) == 0:
            continue
        scores = output["scores"].tolist()
        labels = output["labels"].tolist()
        keypoints = output["keypoints"]
        keypoints = keypoints.flatten(start_dim=1).tolist()
        coco_results.extend([{"image_id": output["image_id"].item(),
                              "category_id": labels[k],
                              'keypoints': keypoint,
                              "score": scores[k]} for k, keypoint in enumerate(keypoints)])
    return coco_results


def convert_to_coco(batch):
    dataset = {'images': [], 'categories': [], 'annotations': []}
    categories = set()
    empty_image_ids = set()
    ann_id = 1
    for image, target in batch:
        image_id = target[constants.TargetType.image_id].item()
        img_dict = {'id': image_id, 'height': image.shape[-2], 'width': image.shape[-1]}
        dataset['images'].append(img_dict)
        if constants.TargetType.boxes not in target:
            empty_image_ids.add(image_id)
            continue
        bboxes = copy.deepcopy(target[constants.TargetType.boxes])
        bboxes[:, 2:] -= bboxes[:, :2]
        bboxes = bboxes.tolist()
        labels = target[constants.TargetType.labels].tolist()
        areas = target[constants.TargetType.area].tolist()
        keypoints = target[constants.TargetType.keypoints]
        keypoints = keypoints.reshape(keypoints.shape[0], -1).tolist()
        num_objs = len(bboxes)
        for i in range(num_objs):
            ann = {'id': ann_id, 'image_id': image_id, 'bbox': bboxes[i], 'category_id': labels[i], 'area': areas[i],
                   'iscrowd': 0, 'keypoints': keypoints[i], 'num_keypoints': sum(k != 0 for k in keypoints[i][2::3])}
            dataset['annotations'].append(ann)
            categories.add(labels[i])
            ann_id += 1
    dataset['categories'] = [{'id': i} for i in sorted(categories)]
    coco = COCO()
    coco.dataset = dataset
    coco.empty_image_ids = list(empty_image_ids)
    coco.createIndex()
    return coco
