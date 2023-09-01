import os
import os.path as osp
import sys
import argparse

# limit the number of cpus used by high performance libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import time
from pathlib import Path

import cv2
import torch
import numpy as np

import warnings
warnings.filterwarnings('ignore')

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # strongsort root directory
WEIGHTS = ROOT / 'weights'

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / 'yolov7') not in sys.path:
    sys.path.append(str(ROOT / 'yolov7'))  # add yolov7 ROOT to PATH
if str(ROOT / 'strong_sort') not in sys.path:
    sys.path.append(str(ROOT / 'strong_sort'))  # add strong_sort ROOT to PATH
ROOT = Path(osp.relpath(ROOT, Path.cwd()))  # relative

from yolov7.models.experimental import attempt_load
from yolov7.utils.plots import get_rgb_colors
from yolov7.utils.datasets import LoadImages
from yolov7.utils.general import (check_img_size, yolov5_non_max_suppression, 
                                  scale_coords, strip_optimizer, set_logging, 
                                  increment_path, save_argparser_arguments)
from yolov7.utils.plots import plot_one_box
from yolov7.utils.torch_utils import select_device, time_synchronized, TracedModel

from strong_sort.strong_sort import StrongSORT


def get_detection_filter(rect_pt1, rect_pt2):
    assert rect_pt1 != rect_pt2, 'rect_pt1 and rect_pt2 must be different'
    xs = tuple(sorted([rect_pt1[0], rect_pt2[0]]))
    ys = tuple(sorted([rect_pt1[1], rect_pt2[1]]))
    def detection_filter(detections):
        centroids = ((detections[:, :2] + detections[:, 2:4]) / 2).round().type(torch.int32)
        x_inside = torch.logical_and(centroids[:, 0].ge(xs[0]), centroids[:, 0].le(xs[1]))
        y_inside = torch.logical_and(centroids[:, 1].ge(ys[0]), centroids[:, 1].le(ys[1]))
        return detections[torch.logical_and(x_inside, y_inside)]
    return detection_filter

def get_line_function(pt1, pt2):
    """
    Returns a function that approximates the equation 
    of the line that passes throught points pt1 and pt2.
    """
    coefs = np.linalg.inv([[pt1[0], 1], [pt2[0], 1]]) @ [[pt1[1]], [pt2[1]]]
    def line(x):
        return x * coefs[0, 0].item() + coefs[1, 0].item()
    return line

def get_line_reference_checker(pt1, pt2, higher):
    """
    Returns a function to tell if a (x, y) point is above or below a straight line defined by points 
    `pt1` and `pt2`. If `higher` is True, checks if the point is above the line, if False checks if it 
    is below or intercepting. If `pt1` and `pt2` both have the same x coordinate (is a vertical line), 
    `higher` equal True tells if the point is to the right of the line. `pt1` and `pt2` cannot be the same point.
    """
    assert tuple(pt1) != tuple(pt2), 'pt1 and pt2 must be different'
    if pt1[0] == pt2[0]:
        def checker(point):
            return not (higher ^ (point[0] > pt1[0]))
    elif pt1[1] == pt2[1]:
        def checker(point):
            return not (higher ^ (point[1] > pt1[1]))
    else:
        line_func = get_line_function(pt1, pt2)
        def checker(point):
            y = line_func(point[0])
            return not (higher ^ (point[1] > y))
    return checker

def get_track_killer(line_killers):
    if line_killers:
        checkers = [get_line_reference_checker((x1, y1), (x2, y2), bool(higher)) \
                    for x1, y1, x2, y2, higher in line_killers]
        def track_killer(track):
            centroid = track.last_associated_xyah[:2].tolist()
            checks = np.array([checker(centroid) for checker in checkers], dtype=np.bool_)
            return checks.any()
        return track_killer
    else:
        return None

def _build_strong_sort(opt):
    return StrongSORT(
        device=select_device(opt.device), 
        max_appearance_distance=opt.appearance_gate, 
        nn_budget=opt.feature_bank_size, 
        max_iou_distance=opt.iou_gate, 
        max_age=opt.max_age, 
        n_init=opt.init_period, 
        ema_alpha=opt.feature_momentum, 
        mc_lambda=opt.appearance_lambda, 
        matching_cascade=opt.matching_cascade,
        only_position=opt.motion_only_position,
        motion_gate_coefficient=opt.motion_gate_coefficient,
        max_centroid_distance=opt.max_centroid_distance, 
        max_velocity=opt.max_velocity,
        track_killer=get_track_killer(opt.line_track_kill),
        iou_distance_cost=opt.iou_distance_cost)


def detect(opt):
    assert osp.isdir(opt.source) or osp.isfile(opt.source), 'Source must be a video file or a directory'

    # Directories
    save_dir = Path(increment_path(Path(opt.project).absolute() / opt.name, exist_ok=opt.exist_ok))
    (save_dir / 'labels' if opt.save_txt else save_dir).mkdir(parents=True, exist_ok=True)

    save_argparser_arguments(opt, str(save_dir / 'arguments.txt'), False)

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(opt.yolo_weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(opt.img_size, s=stride)  # check img_size

    if opt.trace:
        model = TracedModel(model, device, imgsz)

    if half:
        model.half()  # to FP16
    
    if opt.filter_detections:
        pt1 = tuple(opt.filter_detections[:2])
        pt2 = tuple(opt.filter_detections[2:])
        detection_filter = get_detection_filter(pt1, pt2)
    else:
        detection_filter = lambda x: x

    # Set Dataloader
    dataset = LoadImages(opt.source, img_size=imgsz, stride=stride)
    num_videos = sum(dataset.video_flag)
    num_images = dataset.nf - num_videos
    num_sources = num_videos + bool(num_images)
    source_padding = len(str(num_sources))

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = get_rgb_colors(len(names), cmin=50, cmax=200, gray_colors=False)

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    
    # Run tracking
    t0 = time.time()
    dt = [0.0, 0.0, 0.0, 0.0]
    vid_writer, curr_frames, prev_frames, txt_path = [None] * 4
    strong_sort = _build_strong_sort(opt)
    trajectorys = {}
    
    if num_images:
        num_frames = num_images
        num_frames_padding = len(str(num_frames))
        txt_path = str(save_dir / 'labels' / 'image_sequence.txt')
        if opt.save_vid:
            vid_writer = cv2.VideoWriter(str(save_dir / 'video_from_imgs.mp4'), 
                                         cv2.VideoWriter_fourcc(*'mp4v'), 
                                         15, im0.shape[:2][::-1])
        if opt.save_img:
            imgs_path = save_dir / 'images' / 'imgs'
            imgs_path.mkdir(parents=True, exist_ok=True)

    for path, img, im0, vid_cap in dataset:
        frame_id = getattr(dataset, 'frame', dataset.count)
        curr_frames = im0
        base_name = osp.splitext(osp.basename(path))[0]
        
        if vid_cap and dataset.frame == 1:
            strong_sort.restart()
            trajectorys = {}
            num_frames = dataset.nframes
            num_frames_padding = len(str(num_frames))
            txt_path = str(save_dir / 'labels' / f'{base_name}.txt')
            if opt.save_img:
                imgs_path = save_dir / 'images' / base_name
                imgs_path.mkdir(parents=True, exist_ok=True)
            if opt.save_vid:
                try: vid_writer.release()
                except AttributeError: pass
                video_save_path = str(save_dir / f'{base_name}.mp4')
                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                resolution = (int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 
                              int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                vid_writer = cv2.VideoWriter(video_save_path, fourcc, fps, resolution)
        
        result_message = 'source %d/%d (%dx%d %s) | frame %d/%d |' %(
            1 if dataset.mode == 'image' else (1 + dataset.count - max(num_images - 1, 0)), 
            num_sources, *im0.shape[:2][::-1], dataset.mode, frame_id, num_frames)

        t1 = time_synchronized()
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0
        t2 = time_synchronized()
        dt[0] += t2 - t1
        
        # Inference
        pred = model(img, augment=opt.augment)[0]
        t3 = time_synchronized()
        dt[1] += t3 - t2

        # Apply NMS
        detections = yolov5_non_max_suppression(
            pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)[0]
        dt[2] += time_synchronized() - t3
        
        # Process detections
        # Rescale boxes from img_size to im0 size
        detections[:, :4] = scale_coords(img.shape[2:], detections[:, :4], im0.shape).round()
        detections = detection_filter(detections).detach().cpu().numpy()
        
        if opt.ecc:  # camera motion compensation
            strong_sort.tracker.camera_update(prev_frames, curr_frames)
        
        if detections.any():
            xyxys = detections[:, :4].astype(np.int32)
            confs = detections[:, 4].astype(np.float32)
            classes = detections[:, 5].astype(np.int32)

            cls_counts = zip(*np.unique(classes, return_counts=True))
            cls_counts = [f'{names[i.item()]} x{j.item()}' for i, j in cls_counts]
            result_message += f' {" ".join(cls_counts)} |'

            # Pass detections to strongsort
            t4 = time_synchronized()
            sort_output = strong_sort.update(xyxys, confs, classes, im0[:, :, ::-1])
            t5 = time_synchronized()
            dt[3] += t5 - t4
            
            if sort_output.any():
                sort_output = np.c_[np.full((sort_output.shape[0], 1), frame_id), sort_output]
                if opt.save_txt:
                    if opt.mot_format:
                        # <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
                        mot_output = np.c_[sort_output[:, [0, 1, 3, 4, 5, 6, 7]], 
                                           np.full((sort_output.shape[0], 3), -1)]
                        with open(txt_path, mode='a') as f:
                            np.savetxt(f, mot_output, fmt='%d,%d,%d,%d,%d,%d,%.5f,%d,%d,%d')
                        del mot_output
                    else:
                        # <frame>, <id>, <class>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>
                        with open(txt_path, mode='a') as f:
                            np.savetxt(f, sort_output, fmt='%d,%d,%d,%d,%d,%d,%d,%.5f')
                
                for output in sort_output:
                    track_id, cls = output[[1, 2]].astype(np.int32) 
                    tlwh = output[3:7]
                    conf = output[7]
                    if opt.draw_trajectory:
                        center = tuple((tlwh[:2] + tlwh[2:] / 2).round().astype(np.int32).tolist())
                        track_trajs = trajectorys.setdefault(track_id, [])
                        track_trajs.append(center)
                        for c in range(-1, -21, -1):  # Draw only the last 20 points
                            try:
                                c1, c0 = track_trajs[c], track_trajs[c-1]
                            except IndexError:
                                break
                            cv2.line(im0, c0, c1, colors[cls], 3)

                    if opt.save_vid:  # Add bbox to image
                        xyxy = tlwh.astype(np.int32)
                        xyxy[2:] = xyxy[:2] + xyxy[2:]
                        label = None if opt.hide_labels else (str(track_id) if opt.hide_conf and opt.hide_class else \
                                                              f'{track_id} {names[cls]}' if opt.hide_conf else \
                                                              f'{track_id} {conf:.2f}' if opt.hide_class else \
                                                              f'{track_id} {names[cls]} {conf:.2f}')
                        plot_one_box(xyxy, im0, label=label, color=colors[cls], line_thickness=opt.line_thickness)

        else:
            strong_sort.increment_ages()
            result_message += ' None detections |'

        if opt.verbose:
            print(f'{result_message} YOLO {dt[1]:.3e}s, StrongSORT {dt[3]:.3e}s')
        else:
            p = 100.0 * frame_id / num_frames
            s = int(p // 5)
            result_message = '\r{} of {} sources [{: <20}] {:.1f}% '.format(
                f'{(1 if dataset.mode == "image" else (1 + dataset.count - max(num_images - 1, 0))): >{source_padding}}', 
                num_sources, "-" * (s - 1) + ("-" if s == 20 else ">" if s else ""), p)
            print(result_message, end='', flush=True)

        if opt.save_img:
            file_name = f'{frame_id:0>{num_frames_padding}}_{base_name}.jpg'
            save_img_result = cv2.imwrite(str(imgs_path / file_name), im0)
            if not save_img_result and opt.verbose:
                print('Error while saving image/frame:')
                print('    - Frame ID :', frame_id)
                print('    - File     :', path)
        
        if opt.save_vid:
            vid_writer.write(im0)

        prev_frames = curr_frames
    
    try:
        vid_writer.release()
        vid_cap.release()
    except AttributeError:
        pass
    
    if opt.save_txt or opt.save_vid or opt.save_img:
        print(f'Results saved to {save_dir}')
    print(f'All done in {time.time() - t0:.3f}s')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    
    ### YOLO
    parser.add_argument(
        '--yolo-weights', 
        nargs='+', type=str, default='weights/yolov7-tiny.pt', 
        help='Path to yolo model file'
    )

    parser.add_argument(
        '--img-size', 
        type=int, default=640, 
        help='Inference size (pixels) of yolo detector'
    )

    parser.add_argument(
        '--conf-thres', 
        type=float, default=0.5, 
        help='Object detection confidence threshold'
    )
    
    parser.add_argument(
        '--iou-thres', 
        type=float, default=0.5, 
        help='IoU threshold for NMS'
    )

    parser.add_argument(
        '--classes', 
        nargs='+', type=int, 
        help='Filter detections by class index, e.g.: --class 0 2 3'
    )

    parser.add_argument(
        '--agnostic-nms', 
        action='store_true', 
        help='Class-agnostic NMS'
    )

    parser.add_argument(
        '--augment', 
        action='store_true', 
        help='Augmented inference'
    )

    parser.add_argument(
        '--update', 
        action='store_true', 
        help='Update all models'
    )

    parser.add_argument(
        '--trace', 
        action='store_true', 
        help='Trace model'
    )

    parser.add_argument(
        '--filter-detections', type=int, nargs=4, default=None, 
        help='Select only the detections with centroid inside a rectangle defined using "--filter-detections x1 y1 x2 y2"'
    )

    ### General
    parser.add_argument(
        '--source', 
        type=str, default='.', 
        help='Source data (video file or directory with images or/and videos) for tracking'
    )

    parser.add_argument(
        '--device', 
        default='cpu', 
        help='CUDA device, i.e. 0 or 0,1,2,3 or cpu'
    )

    parser.add_argument(
        '--verbose', 
        action='store_true', 
        help='Report tracking informations'
    )

    ### Saving results
    parser.add_argument(
        '--project', 
        default='runs/track', 
        help='Save results to /project/name'
    )

    parser.add_argument(
        '--name', 
        default='exp', 
        help='Save results to /project/name'
    )

    parser.add_argument(
        '--exist-ok', 
        action='store_true', 
        help='Existing /project/name ok, do not increment dir path'
    )

    parser.add_argument(
        '--save-txt', 
        action='store_true', 
        help='Save tracking results to /project/name/labels/*.txt'
    )

    parser.add_argument(
        '--mot-format', 
        action='store_true', 
        help='Use the MOTChallenge format to save the text file with the tracking results'
    )

    parser.add_argument(
        '--save-img', 
        action='store_true', 
        help='Save processed images/frames to /project/name/images/*/*.jpg'
    )

    parser.add_argument(
        '--save-vid', 
        action='store_true', 
        help='Save videos with the tracking results'
    )

    parser.add_argument(
        '--line-thickness', 
        type=int, default=2, 
        help='Bounding box thickness in pixels'
    )

    parser.add_argument(
        '--hide-labels', 
        action='store_true', 
        help='Only draw bounding box rectangle'
    )

    parser.add_argument(
        '--hide-conf', 
        action='store_true', 
        help='Hide detection confidence in bounding box label'
    )

    parser.add_argument(
        '--hide-class', 
        action='store_true', 
        help='Hide class id in bounding box label'
    )

    parser.add_argument(
        '--draw-trajectory', 
        action='store_true', 
        help='Display object trajectory lines'
    )

    ### StrongSORT
    parser.add_argument(
        '--matching-cascade',
        action='store_true',
        help='Apply DeepSORT matching cascade'
    )
    
    parser.add_argument(
        '--appearance-lambda',
        type=float, default=0.995,
        help='Appearance cost weight for appearance-motion cost matrix calculation'
    )
    
    parser.add_argument(
        '--iou-gate',
        type=float, default=0.7,
        help='IoU distance gate for the final IoU matching'
    )

    parser.add_argument(
        '--ecc',
        action='store_true',
        help='Apply camera motion compensation using ECC'
    )
    
    parser.add_argument(
        '--feature-bank-size',
        type=int, default=1,
        help='Num of features to store per Track for appearance distance calculation'
    )

    parser.add_argument(
        '--init-period',
        type=int, default=3,
        help='Size of Track initialization period in frames'
    )
    
    parser.add_argument(
        '--max-age',
        type=int, default=10,
        help='Max period which a Track survive without assignments in frames'
    )
    
    parser.add_argument(
        '--feature-momentum',
        type=float, default=0.9,
        help='Momentum term for feature vector update'
    )
    
    parser.add_argument(
        '--appearance-gate',
        type=float, default=0.2,
        help='Track-detection associations with appearance cost greater than this value are disregarded'
    )

    parser.add_argument(
        '--motion-only-position', 
        action='store_true', 
        help='Use only centroid position to compute motion cost'
    )

    parser.add_argument(
        '--motion-gate-coefficient',
        type=float, default=1.0,
        help='Coefficient that multiplies the motion gate to control track-detection associations'
    )

    parser.add_argument(
        '--max-centroid-distance',
        type=int, default=None,
        help='Max distance in pixels between track and detection centroids for track-detection match'
    )

    parser.add_argument(
        '--max-velocity',
        type=float, default=None,
        help='Max velocity in px/frame between track and detection centroids for track-detection match'
    )

    parser.add_argument(
        '--line-track-kill',
        type=int, default=None, nargs=5, action='append',
        help=' '.join(['Kill a track based on its position in regard to a line.\n',
                       'Eg.: "--line-track-kill 10 10 90 90 1" will kill a track when it is above a line determined by points\n',
                       '     (10, 10) and (90, 90). Supply 0 instead of 1 at the end to kill when its below or intercepting the line.\n',
                       '     If the points have the same x coordinate, 1 at the end kills tracks at the right of the line'])
    )

    parser.add_argument(
        '--iou-distance-cost', 
        action='store_true', 
        help='Use IoU distance instead of Mahalanobis distance in track-detection association'
    )

    opt = parser.parse_args()
    print(opt)
    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            detect(opt)
            strip_optimizer(opt.yolo_weights)
        else:
            detect(opt)
