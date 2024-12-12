import cv2
import numpy as np
import glob
import os.path as osp
import os
from pathlib import Path

import datetime
from tqdm import tqdm

from dpvo.utils import Timer
from dpvo.stereo_switching_dpvo import DPVO
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

def show_image(image, t=0):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey(t)

@torch.no_grad()
def run(cfg, network, imagedir_left, imagedir_rigt, calib, stride=1, viz=False):

    slam = None

    queue_left = Queue(maxsize=8)
    reader_left = Process(target=image_stream, args=(queue_left, imagedir_left, calib, stride, 0))
    reader_left.start()

    queue_rigt = Queue(maxsize=8)
    reader_rigt = Process(target=image_stream, args=(queue_rigt, imagedir_rigt, calib, stride, 0))
    reader_rigt.start()
    
    while 1:
        
        (t, image_left, intrinsics, tstamp) = queue_left.get()
        print("tstamp: " + str(int(tstamp)))
        if t < 0: 
            break

        (t_rigt, image_rigt, _  , tstamp_rigt) = queue_rigt.get()
        if t_rigt < 0:
            break
        
        if tstamp_rigt != tstamp:
            break

        image_left = torch.from_numpy(image_left).permute(2,0,1).cuda()
        image_rigt = torch.from_numpy(image_rigt).permute(2,0,1).cuda()
        intrinsics = torch.from_numpy(intrinsics).cuda()

        if slam is None:
            slam = DPVO(cfg, network, ht=image_left.shape[1], wd=image_left.shape[2], viz=viz)

        image_left = image_left.cuda()
        image_rigt = image_rigt.cuda()
        intrinsics = intrinsics.cuda()

        with Timer("SLAM", enabled=False):
            slam(t, image_left, image_rigt, intrinsics, tstamp)

    for _ in range(12):
        slam.update()

    reader_left.terminate()
    reader_rigt.terminate()

    return slam.terminate()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', type=str, default='dpvo.pth')
    parser.add_argument('--config', default="config/default.yaml")
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
        "MH_01_easy",
        "MH_02_easy",
        "MH_03_medium",
        "MH_04_difficult",
        "MH_05_difficult",
        "V1_01_easy",
        "V1_02_medium",
        "V1_03_difficult",
        "V2_01_easy",
        "V2_02_medium",
        "V2_03_difficult",
    ]

    results = {}

    result_log_file = open("stereo_log.txt3", 'w')
    result_log_file.write('\n')
    result_log_file.close()

    for scene in euroc_scenes:
        imagedir_left = os.path.join(args.eurocdir, scene, "mav0/cam0/data")
        imagedir_rigt = os.path.join(args.eurocdir, scene, "mav0/cam1/data")
        groundtruth = "datasets/euroc_groundtruth/{}.txt".format(scene) 

        scene_results = []
        for num_patch in range(96, 90, -16):
          for num_frame in range(22, 20, -6):
            cfg['PATCHES_PER_FRAME'] = num_patch
            cfg['REMOVAL_WINDOW'] = num_frame
            for i in range(args.trials):
                traj_est, timestamps, dynamic_slam_logger = run(cfg, args.network, imagedir_left, imagedir_rigt, "calib/euroc.txt", args.stride, args.viz)

                images_list = sorted(glob.glob(os.path.join(imagedir_left, "*.png")))[::args.stride]
                tstamps = [float(x.split('/')[-1][:-4]) for x in images_list]

                tstamps = tstamps[:len(traj_est)]

                dynamic_slam_logger['original_est'] = traj_est

                traj_est = PoseTrajectory3D(
                    positions_xyz=traj_est[:,:3],
                    orientations_quat_wxyz=traj_est[:,3:],
                    timestamps=np.array(tstamps))

                traj_ref = file_interface.read_tum_trajectory_file(groundtruth)
                traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est)

                result_log_file = open("stereo_log.txt3", 'a')
                for test_idx in range(0, len(traj_est.positions_xyz)):
                  dis_est = traj_est.positions_xyz[test_idx] - traj_est.positions_xyz[0]
                  dis_est = dis_est[0] ** 2 + dis_est[1]**2 + dis_est[2]**2
                  dis_est = math.sqrt(dis_est)

                  dis_ref = traj_ref.positions_xyz[test_idx] - traj_ref.positions_xyz[0]
                  dis_ref = dis_ref[0] ** 2 + dis_ref[1]**2 + dis_ref[2]**2
                  dis_ref = math.sqrt(dis_ref)

                  result_log_file.write("idx: " + str(test_idx) + "dis_est: " + str(dis_est) + "  dis_ref: " + str(dis_ref) + '\n')

                result = main_ape.ape(traj_ref, traj_est, est_name='traj', 
                    pose_relation=PoseRelation.translation_part, align=True, correct_scale=True)
                ate_score = result.stats["rmse"]

                for test_idx in range(0, len(traj_est.positions_xyz)):
                  dis_est = traj_est.positions_xyz[test_idx] - traj_est.positions_xyz[0]
                  dis_est = dis_est[0] ** 2 + dis_est[1]**2 + dis_est[2]**2
                  dis_est = math.sqrt(dis_est)

                  dis_ref = traj_ref.positions_xyz[test_idx] - traj_ref.positions_xyz[0]
                  dis_ref = dis_ref[0] ** 2 + dis_ref[1]**2 + dis_ref[2]**2
                  dis_ref = math.sqrt(dis_ref)

                  result_log_file.write("***idx: " + str(test_idx) + "dis_est: " + str(dis_est) + "  dis_ref: " + str(dis_ref) + '\n')

                
                result_log_file.write(scene + " result: " + str(ate_score) + '\n')
                result_log_file.close()

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
                dynamic_log_picke_file = open("dynamic_slam_log/logs/dynamic_slam_log_stereo_no_correct_scale"+scene+"_"+str(num_patch)+"_patches_"+str(num_frame)+"_frames_trials_"+str(i)+".pickle", "wb")
                pickle.dump(dynamic_slam_logger, dynamic_log_picke_file)

        results[scene] = np.median(scene_results)
        print(scene, sorted(scene_results))

    
    xs = []
    for scene in results:
        print(scene, results[scene])
        xs.append(results[scene])

    print("AVG: ", np.mean(xs))

    

    
