import cv2
import glob
import os
import datetime
import numpy as np
import os.path as osp
from pathlib import Path

import torch
from dpvo.adaptive_model_select_config_dpvo import DPVO
from dpvo.utils import Timer
from dpvo.config import cfg

from dpvo.data_readers.tartan import test_split as val_split
from dpvo.plot_utils import plot_trajectory, save_trajectory_tum_format

import evo
from evo.core.trajectory import PoseTrajectory3D
from evo.tools import file_interface
from evo.core import sync
import evo.main_ape as main_ape
from evo.core.metrics import PoseRelation
import pickle
import math

test_split = \
    ["MH%03d"%i for i in range(8)] + \
    ["ME%03d"%i for i in range(8)]

STRIDE = 1
fx, fy, cx, cy = [320, 320, 320, 240]


def difficulty_gt_eval(traj_est, tstamps, traj_ref, slam):
      
    # try:
      traj_est = PoseTrajectory3D(
        positions_xyz=traj_est[:,:3],
        orientations_quat_wxyz=traj_est[:,3:],
        timestamps=np.array(tstamps, dtype = np.float64))
      
      traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est)

      result = main_ape.ape(traj_ref, traj_est, est_name='traj', 
          pose_relation=PoseRelation.translation_part, align=True, correct_scale=True)
      
      #gt diffc compute
      est_xyz           = result.trajectories['traj'].positions_xyz
      ref_xyz           = result.trajectories['reference'].positions_xyz


      error_vec2 = est_xyz[-2] - ref_xyz[-2]
      error_vec1 = est_xyz[-1] - ref_xyz[-1]


      change_of_error_vec = error_vec1 - error_vec2

      change_of_error = math.sqrt(change_of_error_vec[0]**2 + change_of_error_vec[1]**2 + change_of_error_vec[2]**2)*50

      print("gt diffc: " + str(change_of_error))

      #compute movement
      est_pose          = result.trajectories['traj'].poses_se3[-101:]
      est_xyz           = result.trajectories['traj'].positions_xyz[-101:]
    
      rotation_log = []
      translation_log = []
      for idx in range(0, len(est_pose)):
          if idx == 0:
            rotation_log.append(0)
            translation_log.append(0)
            continue


          #translation
          last_coor = est_xyz[idx-1]
          curr_coor = est_xyz[idx]

          diff_coor = last_coor - curr_coor

          translation_movement = math.sqrt(diff_coor[0]**2 + diff_coor[1]**2 + diff_coor[2]**2)

          translation_log.append(translation_movement)

          #rotation

          last_rotation = np.array(est_pose[idx - 1][:3, :3])
          current_totation = np.array(est_pose[idx][:3, :3])

          R_relative = np.dot(last_rotation.T, current_totation)

          trace = np.trace(R_relative)

          cos_theta = (trace - 1) / 2
          cos_theta = np.clip(cos_theta, -1, 1)
          theta = np.arccos(cos_theta)

          rotation_log.append(theta)

      #scaling
      confidence_scale_factor       = 800
      translation_scale_factor      = 0.8
      rotation_scale_factor         = 0.5

      confidence = slam.dynamic_debug_confidence_average_log[-100:]

      confidence            = [x / confidence_scale_factor for x in confidence]
      rotation_log          = [x / rotation_scale_factor for x in rotation_log]
      translation_log       = [x / translation_scale_factor for x in translation_log]

      # print("max confidence: " + str(max(confidence)))
      # print("max rotation: " + str(max(rotation_log)))
      # print("max translation: " + str(max(translation_log)))

      change_of_error = slam.difficulty_estimation(confidence, rotation_log, translation_log)
      print("difficulty level:" + str(change_of_error))
      return change_of_error, result.np_arrays['error_array'] #scalling factor
    # except:
    #     #cannot associate/scale the traj, return a high difficulty
    #     return 1


def show_image(image, t=0):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey(t)

def video_iterator(imagedir, ext=".png", preload=True):
    imfiles = glob.glob(osp.join(imagedir, "*{}".format(ext)))

    data_list = []
    for imfile in sorted(imfiles)[::STRIDE]:
        image = torch.from_numpy(cv2.imread(imfile)).permute(2,0,1)
        intrinsics = torch.as_tensor([fx, fy, cx, cy])
        data_list.append((image, intrinsics))

    for (image, intrinsics) in data_list:
        yield image.cuda(), intrinsics.cuda()

@torch.no_grad()
def run(imagedir, cfg, network, adaptive_network, traj_ref, viz=False):
    slam = DPVO(cfg, network, adaptive_network, ht=480, wd=640, viz=viz)

    timestamps = []
    for i in range(0, len(traj_ref)):
        timestamps.append(i)
        
    traj_ref = PoseTrajectory3D(
      positions_xyz=traj_ref[:,:3],
      orientations_quat_wxyz=traj_ref[:,3:],
      timestamps=np.array(timestamps, dtype = np.float64))

    adaptive_num_patch = 96
    difficulty_level = 10
    un_opt_err = 1

    recent_difficulty = [1] * 10
    for t, (image, intrinsics) in enumerate(video_iterator(imagedir)):
        if viz: 
            show_image(image, 1)
        
        with Timer("SLAM", enabled=False):
            print("num patches:" + str(adaptive_num_patch))
            traj_est, tstamps ,_ = slam(t, image, intrinsics, t, adaptive_num_patch, difficulty_level, un_opt_err)

        #adaptive config selection
        if t > cfg.ADAPTIVE_INIT_LEN:
            difficulty_level, un_opt_err = difficulty_gt_eval(traj_est, tstamps, traj_ref, slam)
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

    return slam.terminate()


def ate(traj_ref, traj_est, timestamps):
    import evo
    import evo.main_ape as main_ape
    from evo.core.trajectory import PoseTrajectory3D
    from evo.core.metrics import PoseRelation

    traj_est = PoseTrajectory3D(
        positions_xyz=traj_est[:,:3],
        orientations_quat_wxyz=traj_est[:,3:],
        timestamps=np.array(timestamps, dtype = np.float64))

    traj_ref = PoseTrajectory3D(
        positions_xyz=traj_ref[:,:3],
        orientations_quat_wxyz=traj_ref[:,3:],
        timestamps=np.array(timestamps, dtype = np.float64))
    
    traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est)
    
    result = main_ape.ape(traj_ref, traj_est, est_name='traj', 
        pose_relation=PoseRelation.translation_part, align=True, correct_scale=True)

    return result.stats["rmse"], result


@torch.no_grad()
def evaluate(config, net, adaptive_network, split="validation", trials=1, plot=False, save=False):

    if config is None:
        config = cfg
        config.merge_from_file("config/default.yaml")

    if not os.path.isdir("TartanAirResults"):
        os.mkdir("TartanAirResults")

    scenes = test_split if split=="test" else val_split

    results = {}
    all_results = []
    for i, scene in enumerate(scenes):

        results[scene] = []
        for j in range(trials):

            # estimated trajectory
            if split == 'test':
                scene_path = os.path.join("datasets/mono", scene)
                traj_ref = osp.join("datasets/mono", "mono_gt", scene + ".txt")
            
            elif split == 'validation':
                scene_path = os.path.join("datasets/TartanAir", scene, "image_left")
                traj_ref = osp.join("datasets/TartanAir", scene, "pose_left.txt")

            PERM = [1, 2, 0, 4, 5, 3, 6] # ned -> xyz
            traj_ref = np.loadtxt(traj_ref, delimiter=" ")[::STRIDE, PERM]

            # run the slam system
            traj_est, tstamps, dynamic_slam_logger = run(scene_path, config, net, adaptive_network, traj_ref)

            # do evaluation
            ate_score, result_trail = ate(traj_ref, traj_est, tstamps)
            all_results.append(ate_score)
            results[scene].append(ate_score)

            dynamic_slam_logger['result'] = result_trail

            leagal_scene_name = scene.replace("/", "_")
            dynamic_log_picke_file = open("dynamic_slam_log/logs/dynamic_slam_log_model_est_config_10_window_"+leagal_scene_name+"_trials_"+str(j)+".pickle", "wb")
            pickle.dump(dynamic_slam_logger, dynamic_log_picke_file)

            # if plot:
            #     scene_name = '_'.join(scene.split('/')[1:]).title()
            #     Path("trajectory_plots").mkdir(exist_ok=True)
            #     plot_trajectory((traj_est, tstamps), (traj_ref, tstamps), f"TartanAir {scene_name.replace('_', ' ')} Trial #{j+1} (ATE: {ate_score:.03f})",
            #                     f"trajectory_plots/TartanAir_{scene_name}_Trial{j+1:02d}.pdf", align=True, correct_scale=True)

            # if save:
            #     Path("saved_trajectories").mkdir(exist_ok=True)
            #     save_trajectory_tum_format((traj_est, tstamps), f"saved_trajectories/TartanAir_{scene_name}_Trial{j+1:02d}.txt")

        print(scene, sorted(results[scene]))

    results_dict = dict([("Tartan/{}".format(k), np.median(v)) for (k, v) in results.items()])

    # write output to file with timestamp
    with open(osp.join("TartanAirResults", datetime.datetime.now().strftime('%m-%d-%I%p.txt')), "w") as f:
        f.write(','.join([str(x) for x in all_results]))

    xs = []
    for scene in results:
        x = np.median(results[scene])
        xs.append(x)

    ates = list(all_results)
    results_dict["AUC"] = np.maximum(1 - np.array(ates), 0).mean()
    results_dict["AVG"] = np.mean(xs)

    return results_dict


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--viz', action="store_true")
    parser.add_argument('--adaptive_network', type=str, default='/pool0/piaodeng/dynamic_dpvo_selector/confidence_to_difficulty_model_new.pth150')
    parser.add_argument('--id', type=int, default=-1)
    parser.add_argument('--weights', default="dpvo.pth")
    parser.add_argument('--config', default="config/adaptive.yaml")
    parser.add_argument('--split', default="validation")
    parser.add_argument('--trials', type=int, default=1)
    parser.add_argument('--plot', action="store_true")
    parser.add_argument('--save_trajectory', action="store_true")
    args = parser.parse_args()

    cfg.merge_from_file(args.config)

    print("Running with config...")
    print(cfg)

    torch.manual_seed(1234)

    if args.id >= 0:
        scene_path = os.path.join("datasets/mono", test_split[args.id])
        traj_est, tstamps = run(scene_path, cfg, args.weights, viz=args.viz)

        traj_ref = osp.join("datasets/mono", "mono_gt", test_split[args.id] + ".txt")
        traj_ref = np.loadtxt(traj_ref, delimiter=" ")[::STRIDE,[1, 2, 0, 4, 5, 3, 6]]

        # do evaluation
        print(ate(traj_ref, traj_est, tstamps))

    else:
        results = evaluate(cfg, args.weights, args.adaptive_network, split=args.split, trials=args.trials, plot=args.plot, save=args.save_trajectory)
        for k in results:
            print(k, results[k])
