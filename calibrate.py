import os
import os.path as osp
from itertools import product
import subprocess
from copy import deepcopy
from shutil import copyfile, rmtree
from time import time
import numpy as np

ROOT = osp.dirname(__file__)
TRACKER = osp.join(ROOT, 'track.py')
EVALUATOR = osp.join(ROOT, 'TrackEval', 'scripts', 'run_mot_challenge.py')


# Modify only here ----------------------------------
# Update ground truth sequences inside calibration/gt
results_folder = 'results'

calibration_parameters = {
    'matching-cascade': [False, True], 
    'appearance-lambda': [0.2, 0.5, 0.7, 0.9, 0.98], 
    'feature-momentum': [0.0, 0.25, 0.5, 0.75, 0.95], 
    'appearance-gate': [0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3, 0.325, 0.35], 
    'motion-only-position': [False, True], 
    'iou-distance-cost': [False, True]}

commom_parameters = {
    'conf-thres': 0.75, 
    'iou-thres': 0.6, 
    'max-age': 60, 
    'agnostic-nms': True, 
    'init-period': 3, 
    'iou-gate': 0.99, 
    'feature-bank-size': 60, 
    'device': 0, 
    'max-centroid-distance': 200, 
    'yolo-weights': osp.normpath(osp.join(ROOT, 'weights', 'best.pt')), 
    'img-size': 1280, 
    'augment': True}
# ---------------------------------------------------


CALIBRATION_DIR = osp.join(ROOT, 'calibration')
SEQS_DIR = osp.join(CALIBRATION_DIR, 'gt')
RESULTS_DIR = osp.join(CALIBRATION_DIR, results_folder)
TRACKERS_DIR = osp.join(RESULTS_DIR, 'evaluated')
RESULT_FILE = osp.join(RESULTS_DIR, 'results.txt')
VELOCITY_FILE = osp.join(RESULTS_DIR, 'vels.npy')
BEST_METRICS = osp.join(RESULTS_DIR, 'best.txt')
os.makedirs(TRACKERS_DIR, exist_ok=True)

def command_append_parameters(command, name, value):
    if isinstance(value, bool):
        if value: command.append(f'--{name}')
    elif isinstance(value, list):
        command.extend([f'--{name}'] + [str(p) for p in value])
    else:
        command.extend([f'--{name}', str(value)])

def save_best(combination):
    args_file = osp.join(TRACKERS_DIR, combination, 'arguments.txt')
    copyfile(args_file, RESULT_FILE)
    eval_file = osp.join(TRACKERS_DIR, combination, 'eval', 'pedestrian_summary.txt')
    copyfile(eval_file, BEST_METRICS)

def update_progress(comb_id):
    with open(RESULT_FILE, mode='w') as f:
        f.write(f'next_combination_id {comb_id + 1}')

def check_progress():
    try:
        with open(RESULT_FILE, mode='r') as f:
            progress = f.read()
    except FileNotFoundError:
        update_progress(0)
        return 1
    if progress.startswith('next_combination_id'):
        return int(progress.split(' ')[-1])
    return 0

def get_fitness(combination):
    metrics = ['HOTA', 'IDF1', 'MOTA']
    weights = [0.3, 0.7, 0.0]
    eval_file = osp.join(TRACKERS_DIR, combination, 'eval', 'pedestrian_summary.txt')
    with open(eval_file, mode='r') as f:
        metrics_names = f.readline().split()
        metrics_values = f.readline().split()
    fitness = sum(float(metrics_values[i]) * weights[metrics.index(m)] \
                  for i, m in enumerate(metrics_names) if m in metrics)
    return fitness

def scan_combinations():
    best_fit, best_comb = 0.0, None
    for combination in os.listdir(TRACKERS_DIR):
        fitness = get_fitness(combination)
        if fitness > best_fit or best_comb is None:
            best_fit, best_comb = fitness, combination
    save_best(best_comb)
    return best_fit, best_comb

def get_velocity_array():
    if osp.isfile(VELOCITY_FILE):
        return np.load(VELOCITY_FILE)
    else:
        return np.empty((0,), dtype=np.float32)

def update_velocity_array(array, duration):
    array = np.append(array, duration)
    np.save(VELOCITY_FILE, array)
    return array

def run_command(command):
    start = time()
    result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    end = time()
    if result.returncode: print(result.stderr.decode('utf-8'))
    result.check_returncode()
    return end - start

TRACK_BASE_COMMAND = ['python', TRACKER, '--project', TRACKERS_DIR, 
                      '--source', SEQS_DIR, '--mot-format', '--save-txt']

EVAL_COMMAND = ['python', EVALUATOR, '--USE_PARALLEL', 'False', '--NUM_PARALLEL_CORES', '1', 
                '--PRINT_RESULTS', 'False', '--PRINT_CONFIG', 'False', '--TIME_PROGRESS', 'False', 
                '--OUTPUT_DETAILED', 'False', '--PLOT_CURVES', 'False', 
                '--GT_FOLDER', SEQS_DIR, '--SEQ_INFO', *os.listdir(SEQS_DIR), 
                '--TRACKERS_FOLDER', TRACKERS_DIR, '--TRACKER_SUB_FOLDER', 'labels', 
                '--BENCHMARK', 'None', '--SPLIT_TO_EVAL', 'None', '--SKIP_SPLIT_FOL', 'True', 
                '--OUTPUT_SUB_FOLDER', 'eval', '--SKIP_SPLIT_FOL', 'True', 
                '--DO_PREPROC', 'False', '--METRICS', 'HOTA', 'CLEAR', 'Identity']

for param_name, param_value in commom_parameters.items():
    command_append_parameters(TRACK_BASE_COMMAND, param_name, param_value)

if __name__ == '__main__':
    start = time()
    tracking_velocity = get_velocity_array()
    num_combinations = 1
    for space in calibration_parameters.values():
        num_combinations *= len(space)
    name_padding = len(str(num_combinations))
    print('Total of %d combinations to study...' %num_combinations)
    combinations = product(*list(calibration_parameters.values()))
    next_id = check_progress()
    if next_id and next_id <= num_combinations:
        # Restart next combination from scratch
        if next_id > 1: print('Resuming from combination %d...' %next_id)
        next_comb_dir = osp.join(TRACKERS_DIR, f'combination{next_id:0{name_padding}d}')
        if osp.isdir(next_comb_dir): rmtree(next_comb_dir)
        # Skip to next combination
        for i in range(next_id - 1): next(combinations)
        # Continue tracking remaining combinations
        for comb_id, combination in enumerate(combinations, start=next_id):
            progress_str = f'\rTracking combination %{name_padding}d of %d [%.1f%%]' %(
                comb_id, num_combinations, 100.0 * comb_id / num_combinations)
            print(progress_str, end='', flush=True)
            track_command = deepcopy(TRACK_BASE_COMMAND)
            track_command.extend(['--name', f'combination{comb_id:0{name_padding}d}'])
            for i, param_name in enumerate(calibration_parameters):
                command_append_parameters(track_command, param_name, combination[i])
            duration = run_command(track_command)
            tracking_velocity = update_velocity_array(tracking_velocity, duration)
            update_progress(comb_id)
        # Tracking velocity analysis
        print('\nAll combinations tracked: mean %.3fs | std %.3fs | min %.3fs | max %.3fs' %(
            tracking_velocity.mean(), tracking_velocity.std(), 
            tracking_velocity.min(), tracking_velocity.max()))
        # Evaluate all combinations
        print('Starting combinations evaluation...')
        run_command(EVAL_COMMAND)
    # Search for best combination according to get_fitness()
    print('Searching best combination...')
    scan_combinations()
    print('All done in %.2fs. Results saved to %s' %(time() - start, RESULTS_DIR))
