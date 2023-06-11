import numpy as np
import torch
import sys
import gdown
import os.path as osp

from .sort.nn_matching import NearestNeighborDistanceMetric
from .sort.detection import Detection
from .sort.tracker import Tracker
from .deep.reid_model_factory import show_downloadeable_models, get_model_url, get_model_name

from torchreid.utils import FeatureExtractor
from torchreid.utils.tools import download_url

__all__ = ['StrongSORT']

# Pre-trained model from https://github.com/KaiyangZhou/deep-person-reid
DEFAULT_EXTRACTOR = ('osnet_x1_0', 'https://drive.google.com/uc?id=1LaG1EJpHrxdAxKnSCJ_i0u-nbxSAeiFY')
EXTRACTOR_PATH = '%s_imagenet.pth' %(
    osp.abspath(osp.join(osp.dirname(__file__), "..", "weights", DEFAULT_EXTRACTOR[0])))


class StrongSORT(object):
    def __init__(
            self, 
            device, 
            max_appearance_distance = 0.2, 
            nn_budget = 100, 
            max_iou_distance = 0.7, 
            max_age = 70, 
            n_init = 3, 
            ema_alpha = 0.9, 
            mc_lambda = 0.995, 
            matching_cascade = False,
            only_position = False,            
            motion_gate_coefficient = 1.0, 
            max_centroid_distance = None, 
            max_velocity = None, 
            track_killer = None,
            iou_cost_increment = 0.0
        ):
        if not osp.isfile(EXTRACTOR_PATH):
            print(f'Feature extractor not found in {EXTRACTOR_PATH}')
            print(f'Downloading from torchreid model zoo...')
            print('    - Downloading {} from {}'.format(*DEFAULT_EXTRACTOR))
            print('    - To {}'.format(EXTRACTOR_PATH))
            download_url(DEFAULT_EXTRACTOR[1], EXTRACTOR_PATH)

        assert osp.isfile(EXTRACTOR_PATH), 'Couldn\'t load feature extractor.'

        self.extractor = FeatureExtractor(
            model_name=DEFAULT_EXTRACTOR[0], 
            model_path=EXTRACTOR_PATH, 
            device=str(device), 
            image_size=(256, 128), 
            pixel_norm=True, 
            verbose=False
        )

        appearance_metric = NearestNeighborDistanceMetric("cosine", nn_budget)
        self.tracker = Tracker(
            appearance_metric, max_appearance_distance, max_iou_distance, max_age, n_init, 
            ema_alpha, mc_lambda, matching_cascade, only_position, motion_gate_coefficient, 
            max_centroid_distance, max_velocity, track_killer, iou_cost_increment)

    def update(self, bboxes_xyxy, confidences, classes, image):
        bboxes_tlwh = self.xyxy2tlwh(bboxes_xyxy)
        detections = [
            Detection(classes[i], bboxes_tlwh[i], conf, self.get_detection_feature(bboxes_xyxy[i], image)) \
            for i, conf in enumerate(confidences)
        ]

        # update tracker
        self.tracker.predict()
        self.tracker.update(detections)

        # output bbox identities
        outputs = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update:
                continue
            det = track.last_association
            outputs.append([track.track_id, det.class_id, *det.tlwh, det.confidence])
        return np.asarray(outputs)

    def increment_ages(self):
        self.tracker.increment_ages()

    def xyxy2tlwh(self, bboxes_xyxy):
        # Convert nx4 boxes from [x1, y1, x2, y2] to [x1, y1, w, h] where xy1=top-left, xy2=bottom-right
        tlwh = bboxes_xyxy.copy()
        tlwh[:, 2] = bboxes_xyxy[:, 2] - bboxes_xyxy[:, 0]
        tlwh[:, 3] = bboxes_xyxy[:, 3] - bboxes_xyxy[:, 1]
        return tlwh

    def get_detection_feature(self, bbox_xyxy, image):
        xmin, ymin, xmax, ymax = bbox_xyxy
        crop = image[ymin : ymax, xmin : xmax].copy()
        crop[np.random.uniform(size=crop.shape[:2]) > 0.75] = (0, 0, 0)
        feature = self.extractor(crop)[0].detach().cpu().numpy().astype(np.float32)
        feature[:] = feature / np.linalg.norm(feature)
        return feature
    
    def restart(self):
        self.tracker.restart()
