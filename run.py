import subprocess
import os
import os.path as osp

root = osp.dirname(__file__)
video = osp.join(root, 'santosdumontVSdesembargador_09jun21_0900a1100.mp4')
save_dir = osp.join(root, 'runs')
weight = osp.join(root, 'weights', 'best_exp2.pt')
tracker = osp.join(root, 'track.py')
os.makedirs(save_dir, exist_ok=True)

general_flags = ['--conf-thres', '0.75', '--max-age', '60',
                 '--appearance-lambda', '0.98', '--appearance-gate', '0.23',
                 '--feature-momentum', '0.5', '--iou-gate', '0.8',
                 '--max-centroid-distance', '150',  '--init-period', '3',
                 '--iou-thres', '0.7', '--agnostic-nms', '--iou-distance-cost',
                 '--feature-bank-size', '30', '--motion-only-position',
                 '--hide-class', '--draw-trajectory', '--augment',
                 '--save-vid', '--save-txt']

track_command = ['python', tracker, '--source', video, 
                 '--yolo-weights', weight, '--img-size', '1280', '--device', '0', 
                 '--project', save_dir, '--name', 'test_exp2']

subprocess.run(track_command + general_flags, check=True)
