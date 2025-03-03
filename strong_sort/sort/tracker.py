# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
from . import kalman_filter
from . import linear_assignment
from .iou_matching import iou_cost
from .track import Track


def default_track_killer(track: Track) -> bool:
    return False  # do not kill any


class Tracker:
    """
    This is the multi-target tracker.
    Parameters
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        A distance metric for measurement-to-track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.
    Attributes
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        The distance metric used for measurement to track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of frames that a track remains in initialization phase.
    kf : kalman_filter.KalmanFilter
        A Kalman filter to filter target trajectories in image space.
    tracks : List[Track]
        The list of active tracks at the current time step.
    """

    def __init__(
            self, 
            appearance_metric, 
            max_appearance_distance = 0.2, 
            max_iou_distance = 0.9, 
            max_age = 30, 
            n_init = 3, 
            ema_alpha = 0.9, 
            mc_lambda = 0.995, 
            matching_cascade = False, 
            only_position = False, 
            motion_gate_coefficient = 1.0, 
            max_centroid_distance = None, 
            max_velocity = None, 
            track_killer = None, 
            iou_distance_cost = False
        ):
        self.appearance_metric = appearance_metric
        self.max_appearance_distance = max_appearance_distance / 2
        self.max_motion_distance = kalman_filter.chi2inv95[2 if only_position else 4]
        self.max_motion_distance = (self.max_motion_distance * motion_gate_coefficient) ** 0.5
        self.max_iou_distance = max_iou_distance
        self.max_centroid_distance = max_centroid_distance
        self.max_velocity = max_velocity
        self.max_age = max_age
        self.n_init = n_init
        self.ema_alpha = ema_alpha
        self.lmb = mc_lambda
        self.matching_cascade = matching_cascade
        self.only_position = only_position
        self.iou_distance_cost = iou_distance_cost

        self.jump_gater = self.get_association_jump_gater()
        self.tracks = []
        self.track_killer = track_killer if track_killer else default_track_killer
        self._next_id = 1

    def get_association_jump_gater(self):
        if self.max_centroid_distance and self.max_velocity:
            def gating(tracks, detections, track_indices, detection_indices):
                centroid_cost = self.__centroid_distance_cost(
                    tracks, detections, track_indices, detection_indices)
                distance_gate = centroid_cost > self.max_centroid_distance
                for i, track_idx in enumerate(track_indices):
                    centroid_cost[i] = centroid_cost[i] / tracks[track_idx].time_since_update
                velocity_gate = centroid_cost > self.max_velocity
                return np.logical_or(distance_gate, velocity_gate)
        elif self.max_centroid_distance:
            def gating(tracks, detections, track_indices, detection_indices):
                centroid_cost = self.__centroid_distance_cost(
                    tracks, detections, track_indices, detection_indices)
                return centroid_cost > self.max_centroid_distance
        elif self.max_velocity:
            def gating(tracks, detections, track_indices, detection_indices):
                velocity_cost = self.__centroid_distance_cost(
                    tracks, detections, track_indices, detection_indices)
                for i, track_idx in enumerate(track_indices):
                    velocity_cost[i] = velocity_cost[i] / tracks[track_idx].time_since_update
                return velocity_cost > self.max_velocity
        else:
            def gating(*args, **kwargs):
                return False
        return gating

    def __centroid_distance_cost(self, tracks, detections, track_indices, detection_indices):
        tks_centroids = np.array(
            [tracks[i].last_associated_xyah[:2] for i in track_indices], dtype=np.int32)
        dts_centroids = np.array(
            [detections[i].to_xyah()[:2] for i in detection_indices], dtype=np.int32)
        return np.array(
            [np.linalg.norm(dts_centroids - c, axis=1) for c in tks_centroids], dtype=np.float32)

    def gated_iou_metric(self, *args, **kwargs):
        cost_matrix = iou_cost(*args, **kwargs)
        gate = cost_matrix > self.max_iou_distance
        cost_matrix[gate] = cost_matrix[gate] + linear_assignment.INFTY_COST
        return cost_matrix

    def predict(self):
        """Propagate track state distributions one time step forward.

        This function should be called once every time step, before `update`.
        """
        for track in self.tracks:
            track.predict()

    def increment_ages(self):
        for track in self.tracks:
            track.increment_age()
            track.mark_missed()

    def camera_update(self, previous_img, current_img):
        for track in self.tracks:
            track.camera_update(previous_img, current_img)

    def update(self, detections):
        """Perform measurement update and track management.

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.
        """
        # Run matching cascade.
        matches, unmatched_tracks, unmatched_detections = self._match(detections)

        # Update track set.
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(detections[detection_idx])
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx])
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # Update distance metric.
        updated_targets = []
        updated_features = []
        active_tracks = []
        for track in self.tracks:
            active_tracks.append(track.track_id)
            if not track.time_since_update:
                updated_targets.append(track.track_id)
                updated_features.append(track.feature)
        self.appearance_metric.partial_fit(updated_features, updated_targets, active_tracks)

    def gated_metric(self, tracks, detections, track_indices, detection_indices):
        features = [detections[i].feature for i in detection_indices]
        targets = [tracks[i].track_id for i in track_indices]
        measurements = np.asarray([detections[i].to_xyah() for i in detection_indices])
        # Appearance cost
        appearance_cost = self.appearance_metric.distance(features, targets)
        # Mahalanobis motion cost
        mahalanobis_cost = np.zeros_like(appearance_cost)
        for row, track_idx in enumerate(track_indices):
            track = tracks[track_idx]
            mahalanobis_cost[row] = track.kf.gating_distance(
                track.mean, track.covariance, measurements, self.only_position)
        mahalanobis_cost = np.sqrt(mahalanobis_cost) / self.max_motion_distance
        motion_gate = mahalanobis_cost > 1.0
        # Gate
        gate = appearance_cost > self.max_appearance_distance
        gate[:] = np.logical_or(gate, motion_gate)
        gate[:] = np.logical_or(gate, self.jump_gater(tracks, detections, track_indices, detection_indices))
        # Final cost matrix
        if self.iou_distance_cost:
            iou_dist_cost = iou_cost(tracks, detections, track_indices, detection_indices)
            appearance_cost[:] = self.lmb * appearance_cost + (1 - self.lmb) * iou_dist_cost
        else:
            appearance_cost[:] = self.lmb * appearance_cost + (1 - self.lmb) * mahalanobis_cost
        appearance_cost[gate] += linear_assignment.INFTY_COST
        return appearance_cost

    def _match(self, detections):
        # Split track set into confirmed and unconfirmed tracks.
        confirmed_tracks = [i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [i for i, t in enumerate(self.tracks) if t.is_tentative()]

        # Associate confirmed tracks using appearance and motion cost.
        matches_a, unmatched_tracks_a, unmatched_detections = linear_assignment.matching_cascade(
            self.gated_metric, self.max_age, self.tracks, detections, 
            confirmed_tracks, matching_cascade=self.matching_cascade)

        # Associate remaining tracks together with unconfirmed tracks using IOU.
        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update == 1]
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update != 1]
        matches_b, unmatched_tracks_b, unmatched_detections = \
            linear_assignment.min_cost_matching(
                self.gated_iou_metric, self.tracks, detections, 
                iou_track_candidates, unmatched_detections)

        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection):
        track = Track(self._next_id, self.n_init, self.max_age, 
                      self.ema_alpha, self.track_killer)
        if track.initialize(detection):
            self._next_id += 1
            self.tracks.append(track)
    
    def restart(self):
        self.tracks = []
        self._next_id = 1
        self.appearance_metric.restart()
