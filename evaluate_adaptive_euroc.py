import cv2
import numpy as np
import glob
import os.path as osp
import os
from pathlib import Path

import datetime
from tqdm import tqdm

from dpvo.utils import Timer
from dpvo.adaptive_dpvo import DPVO
from dpvo.stream import image_stream
from dpvo.config import cfg
#from dpvo.plot_utils import plot_trajectory, save_trajectory_tum_format

import torch
from multiprocessing import Process, Queue

### evo evaluation library ###
import evo
from evo.core.trajectory import PoseTrajectory3D
from evo.tools import file_interface
from evo.core import sync
import evo.main_ape as main_ape
from evo.core.metrics import PoseRelation

#dynamic slam data log
import pickle
import math


SKIP = 0

def difficulty_gt_eval(traj_est, tstamps, traj_ref):
      
    try:
      traj_est = PoseTrajectory3D(
        positions_xyz=traj_est[:,:3],
        orientations_quat_wxyz=traj_est[:,3:],
        timestamps=np.array(tstamps, dtype = np.float64))
      
      traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est)

      result = main_ape.ape(traj_ref, traj_est, est_name='traj', 
          pose_relation=PoseRelation.translation_part, align=True, correct_scale=True)
      
      print("last err: " + str(result.np_arrays['error_array'][-1]))

      est_xyz           = result.trajectories['traj'].positions_xyz
      ref_xyz           = result.trajectories['reference'].positions_xyz


      error_vec2 = est_xyz[-3] - ref_xyz[-3]
      error_vec1 = est_xyz[-2] - ref_xyz[-2]


      change_of_error_vec = error_vec1 - error_vec2

      change_of_error = math.sqrt(change_of_error_vec[0]**2 + change_of_error_vec[1]**2 + change_of_error_vec[2]**2)


      return change_of_error*50 #scalling factor
    except:
        #cannot associate/scale the traj, return a high difficulty
        return 1

def show_image(image, t=0):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey(t)

@torch.no_grad()
def run(cfg, network, adaptive_network, imagedir, calib, stride=1, viz=False, traj_ref=None):

    slam = None

    queue = Queue(maxsize=8)
    reader = Process(target=image_stream, args=(queue, imagedir, calib, stride, 0))
    reader.start()
    
    adaptive_num_patch = 96
    difficulty_level = 10

    recent_difficulty = [1] * 30
    remove_frame_idx  = 0
    while 1:
        (t, image, intrinsics, tstamp) = queue.get()
        if t < 0: break

        image = torch.from_numpy(image).permute(2,0,1).cuda()
        intrinsics = torch.from_numpy(intrinsics).cuda()

        if viz: 
            show_image(image, 1)

        if slam is None:
            slam = DPVO(cfg, network, adaptive_network, ht=image.shape[1], wd=image.shape[2], viz=viz)

        image = image.cuda()
        intrinsics = intrinsics.cuda()

        with Timer("SLAM", enabled=False):
            print("num of patches: " + str(adaptive_num_patch))
            traj_est, tstamps ,_ = slam(t, image, intrinsics, tstamp, adaptive_num_patch, difficulty_level)

        #adaptive config selection
        if t > cfg.ADAPTIVE_INIT_LEN:
            #remove the unprocessed frame at the start
            
            if len(tstamps) < 100:
                #not enough data points
                continue
            
            difficulty_level = difficulty_gt_eval(traj_est[-105:], tstamps[-105:], traj_ref)
            print("difficulty level:" + str(difficulty_level))

            recent_difficulty = recent_difficulty[1:] + [difficulty_level]

            recent_max_diffc = max(recent_difficulty)

            if recent_max_diffc > cfg.MAX_CONFIG_THRES:
                adaptive_num_patch = 96
            elif recent_max_diffc < cfg.MIN_CONFIG_THRES:
                adaptive_num_patch = 16
            else:
                adaptive_num_patch = int(5*(recent_max_diffc - cfg.MIN_CONFIG_THRES)/(cfg.MAX_CONFIG_THRES - cfg.MIN_CONFIG_THRES)) * 16 + 16


    for _ in range(12):
        slam.update()

    reader.join()

    return slam.terminate()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', type=str, default='dpvo.pth')
    parser.add_argument('--adaptive_network', type=str, default='./dynamic_dpvo_selector/trained_weights/confidence_to_difficulty_model_fixed_agline_all_works.pth')
    parser.add_argument('--config', default="config/adaptive.yaml")
    parser.add_argument('--stride', type=int, default=2)
    parser.add_argument('--viz', action="store_true")
    parser.add_argument('--trials', type=int, default=1)
    parser.add_argument('--eurocdir', default="datasets/EUROC")
    parser.add_argument('--plot', action="store_true")
    parser.add_argument('--save_trajectory', action="store_true",default=False)
    args = parser.parse_args()

    cfg.merge_from_file(args.config)

    print("\nRunning with config...")
    print(cfg, "\n")

    torch.manual_seed(1234)

    euroc_scenes = [
        # "MH_01_easy",
        # "MH_02_easy",
        # "MH_03_medium",
        # "MH_04_difficult",
        # "MH_05_difficult",
        # "V1_01_easy",
        # "V1_02_medium",
        "V1_03_difficult",
        "V2_01_easy",
        # "V2_02_medium",
        # "V2_03_difficult",
    ]

    results = {}
    for scene in euroc_scenes:
        imagedir = os.path.join(args.eurocdir, scene, "mav0/cam0/data")
        groundtruth = "datasets/euroc_groundtruth/{}.txt".format(scene) 

        scene_results = []

        for i in range(args.trials):
            traj_ref = file_interface.read_tum_trajectory_file(groundtruth)
            traj_est, timestamps, dynamic_slam_logger = run(cfg, args.network, args.adaptive_network, imagedir, "calib/euroc.txt", args.stride, args.viz, traj_ref=traj_ref)

            images_list = sorted(glob.glob(os.path.join(imagedir, "*.png")))[::args.stride]
            tstamps = [float(x.split('/')[-1][:-4]) for x in images_list]

            traj_est = PoseTrajectory3D(
                positions_xyz=traj_est[:,:3],
                orientations_quat_wxyz=traj_est[:,3:],
                timestamps=np.array(tstamps))
            
            traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est)

            result = main_ape.ape(traj_ref, traj_est, est_name='traj', 
                pose_relation=PoseRelation.translation_part, align=True, correct_scale=True)
            ate_score = result.stats["rmse"]

            dynamic_slam_logger['result'] = result

            # if args.plot:
            #     scene_name = '_'.join(scene.split('/')[1:]).title()
            #     Path("trajectory_plots").mkdir(exist_ok=True)
            #     plot_trajectory(traj_est, traj_ref, f"Euroc {scene} Trial #{i+1} (ATE: {ate_score:.03f})",
            #                     f"trajectory_plots/Euroc_{scene}_Trial{i+1:02d}.pdf", align=True, correct_scale=True)

            # if args.save_trajectory:
            #     Path("saved_trajectories").mkdir(exist_ok=True)
            #     save_trajectory_tum_format(traj_est, f"saved_trajectories/Euroc_{scene}_Trial{i+1:02d}.txt")

            scene_results.append(ate_score)
            dynamic_log_picke_file = open("dynamic_slam_log/logs/dynamic_slam_log_gt_difficulty_validate_"+scene+"_trials_"+str(i)+".pickle", "wb")
            pickle.dump(dynamic_slam_logger, dynamic_log_picke_file)

        results[scene] = np.median(scene_results)
        print(scene, sorted(scene_results))

    
    xs = []
    for scene in results:
        print(scene, results[scene])
        xs.append(results[scene])

    print("AVG: ", np.mean(xs))

    

    
