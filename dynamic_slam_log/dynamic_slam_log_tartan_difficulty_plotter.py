
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from skimage.transform import rescale
import math
import csv
import multiprocessing

single_plot = 1

gt = 0

dataset_path = '/pool0/piaodeng/adaptive_DPVO/datasets/TartanAir/'

tartan_scenes = [
  "abandonedfactory/abandonedfactory/Easy/P000",
  "abandonedfactory/abandonedfactory/Easy/P001",
  "abandonedfactory/abandonedfactory/Easy/P002",
  "abandonedfactory/abandonedfactory/Easy/P004",
  "abandonedfactory/abandonedfactory/Easy/P005",
  "abandonedfactory/abandonedfactory/Easy/P006",
  "abandonedfactory/abandonedfactory/Easy/P008",
  "abandonedfactory/abandonedfactory/Easy/P009",
  "abandonedfactory/abandonedfactory/Easy/P010",
  "abandonedfactory/abandonedfactory/Easy/P011",
  "abandonedfactory/abandonedfactory/Hard/P000",
  "abandonedfactory/abandonedfactory/Hard/P001",
  "abandonedfactory/abandonedfactory/Hard/P002",
  "abandonedfactory/abandonedfactory/Hard/P003",
  "abandonedfactory/abandonedfactory/Hard/P004",
  "abandonedfactory/abandonedfactory/Hard/P005",
  "abandonedfactory/abandonedfactory/Hard/P006",
  "abandonedfactory/abandonedfactory/Hard/P007",
  "abandonedfactory/abandonedfactory/Hard/P008",
  "abandonedfactory/abandonedfactory/Hard/P009",
  "abandonedfactory/abandonedfactory/Hard/P010",
  "abandonedfactory/abandonedfactory/Hard/P011",
  "abandonedfactory_night/abandonedfactory_night/Easy/P001",
  "abandonedfactory_night/abandonedfactory_night/Easy/P002",
  "abandonedfactory_night/abandonedfactory_night/Easy/P003",
  "abandonedfactory_night/abandonedfactory_night/Easy/P004",
  "abandonedfactory_night/abandonedfactory_night/Easy/P005",
  "abandonedfactory_night/abandonedfactory_night/Easy/P006",
  "abandonedfactory_night/abandonedfactory_night/Easy/P007",
  "abandonedfactory_night/abandonedfactory_night/Easy/P008",
  "abandonedfactory_night/abandonedfactory_night/Easy/P009",
  "abandonedfactory_night/abandonedfactory_night/Easy/P010",
  "abandonedfactory_night/abandonedfactory_night/Easy/P011",
  "abandonedfactory_night/abandonedfactory_night/Easy/P012",
  "abandonedfactory_night/abandonedfactory_night/Easy/P013",
  "abandonedfactory_night/abandonedfactory_night/Hard/P000",
  "abandonedfactory_night/abandonedfactory_night/Hard/P001",
  "abandonedfactory_night/abandonedfactory_night/Hard/P002",
  "abandonedfactory_night/abandonedfactory_night/Hard/P003",
  "abandonedfactory_night/abandonedfactory_night/Hard/P005",
  "abandonedfactory_night/abandonedfactory_night/Hard/P006",
  "abandonedfactory_night/abandonedfactory_night/Hard/P007",
  "abandonedfactory_night/abandonedfactory_night/Hard/P008",
  "abandonedfactory_night/abandonedfactory_night/Hard/P009",
  "abandonedfactory_night/abandonedfactory_night/Hard/P010",
  "abandonedfactory_night/abandonedfactory_night/Hard/P011",
  "abandonedfactory_night/abandonedfactory_night/Hard/P012",
  "abandonedfactory_night/abandonedfactory_night/Hard/P013",
  "abandonedfactory_night/abandonedfactory_night/Hard/P014",
  "amusement/amusement/Easy/P001",
  "amusement/amusement/Easy/P002",
  "amusement/amusement/Easy/P003",
  "amusement/amusement/Easy/P004",
  "amusement/amusement/Easy/P006",
  "amusement/amusement/Easy/P007",
  "amusement/amusement/Easy/P008",
  "amusement/amusement/Hard/P000",
  "amusement/amusement/Hard/P001",
  "amusement/amusement/Hard/P002",
  "amusement/amusement/Hard/P003",
  "amusement/amusement/Hard/P004",
  "amusement/amusement/Hard/P005",
  "amusement/amusement/Hard/P006",
  "amusement/amusement/Hard/P007",
  "carwelding/carwelding/Easy/P001",
  "carwelding/carwelding/Easy/P002",
  "carwelding/carwelding/Easy/P004",
  "carwelding/carwelding/Easy/P005",
  "carwelding/carwelding/Easy/P006",
  "carwelding/carwelding/Easy/P007",
  "carwelding/carwelding/Hard/P000",
  "carwelding/carwelding/Hard/P001",
  "carwelding/carwelding/Hard/P002",
  "carwelding/carwelding/Hard/P003",
  "endofworld/endofworld/Easy/P000",
  "endofworld/endofworld/Easy/P001",
  "endofworld/endofworld/Easy/P002",
  "endofworld/endofworld/Easy/P003",
  "endofworld/endofworld/Easy/P004",
  "endofworld/endofworld/Easy/P005",
  "endofworld/endofworld/Easy/P006",
  "endofworld/endofworld/Easy/P007",
  "endofworld/endofworld/Easy/P008",
  "endofworld/endofworld/Easy/P009",
  "endofworld/endofworld/Hard/P000",
  "endofworld/endofworld/Hard/P001",
  "endofworld/endofworld/Hard/P002",
  "endofworld/endofworld/Hard/P005",
  "endofworld/endofworld/Hard/P006",
  "gascola/gascola/Easy/P001",
  "gascola/gascola/Easy/P003",
  "gascola/gascola/Easy/P004",
  "gascola/gascola/Easy/P005",
  "gascola/gascola/Easy/P006",
  "gascola/gascola/Easy/P007",
  "gascola/gascola/Easy/P008",
  "gascola/gascola/Hard/P000",
  "gascola/gascola/Hard/P001",
  "gascola/gascola/Hard/P002",
  "gascola/gascola/Hard/P003",
  "gascola/gascola/Hard/P004",
  "gascola/gascola/Hard/P005",
  "gascola/gascola/Hard/P006",
  "gascola/gascola/Hard/P007",
  "gascola/gascola/Hard/P008",
  "gascola/gascola/Hard/P009",
  "hospital/hospital/Easy/P000",
  "hospital/hospital/Easy/P001",
  "hospital/hospital/Easy/P002",
  "hospital/hospital/Easy/P003",
  "hospital/hospital/Easy/P004",
  "hospital/hospital/Easy/P005",
  "hospital/hospital/Easy/P006",
  "hospital/hospital/Easy/P007",
  "hospital/hospital/Easy/P008",
  "hospital/hospital/Easy/P009",
  "hospital/hospital/Easy/P010",
  "hospital/hospital/Easy/P011",
  "hospital/hospital/Easy/P012",
  "hospital/hospital/Easy/P013",
  "hospital/hospital/Easy/P014",
  "hospital/hospital/Easy/P015",
  "hospital/hospital/Easy/P016",
  "hospital/hospital/Easy/P017",
  "hospital/hospital/Easy/P018",
  "hospital/hospital/Easy/P019",
  "hospital/hospital/Easy/P020",
  "hospital/hospital/Easy/P021",
  "hospital/hospital/Easy/P022",
  "hospital/hospital/Easy/P023",
  "hospital/hospital/Easy/P024",
  "hospital/hospital/Easy/P025",
  "hospital/hospital/Easy/P026",
  "hospital/hospital/Easy/P027",
  "hospital/hospital/Easy/P028",
  "hospital/hospital/Easy/P029",
  "hospital/hospital/Easy/P030",
  "hospital/hospital/Easy/P031",
  "hospital/hospital/Easy/P032",
  "hospital/hospital/Easy/P033",
  "hospital/hospital/Easy/P034",
  "hospital/hospital/Easy/P035",
  "hospital/hospital/Easy/P036",
  "hospital/hospital/Hard/P037",
  "hospital/hospital/Hard/P038",
  "hospital/hospital/Hard/P039",
  "hospital/hospital/Hard/P040",
  "hospital/hospital/Hard/P041",
  "hospital/hospital/Hard/P042",
  "hospital/hospital/Hard/P043",
  "hospital/hospital/Hard/P044",
  "hospital/hospital/Hard/P045",
  "hospital/hospital/Hard/P046",
  "hospital/hospital/Hard/P047",
  "hospital/hospital/Hard/P048",
  "hospital/hospital/Hard/P049",
  "japanesealley/japanesealley/Easy/P001",
  "japanesealley/japanesealley/Easy/P002",
  "japanesealley/japanesealley/Easy/P003",
  "japanesealley/japanesealley/Easy/P004",
  "japanesealley/japanesealley/Easy/P005",
  "japanesealley/japanesealley/Easy/P007",
  "japanesealley/japanesealley/Hard/P000",
  "japanesealley/japanesealley/Hard/P001",
  "japanesealley/japanesealley/Hard/P002",
  "japanesealley/japanesealley/Hard/P003",
  "japanesealley/japanesealley/Hard/P004",
  "japanesealley/japanesealley/Hard/P005",
  "neighborhood/neighborhood/Easy/P000",
  "neighborhood/neighborhood/Easy/P001",
  "neighborhood/neighborhood/Easy/P002",
  "neighborhood/neighborhood/Easy/P003",
  "neighborhood/neighborhood/Easy/P004",
  "neighborhood/neighborhood/Easy/P005",
  "neighborhood/neighborhood/Easy/P007",
  "neighborhood/neighborhood/Easy/P008",
  "neighborhood/neighborhood/Easy/P009",
  "neighborhood/neighborhood/Easy/P010",
  "neighborhood/neighborhood/Easy/P012",
  "neighborhood/neighborhood/Easy/P013",
  "neighborhood/neighborhood/Easy/P014",
  "neighborhood/neighborhood/Easy/P015",
  "neighborhood/neighborhood/Easy/P016",
  "neighborhood/neighborhood/Easy/P017",
  "neighborhood/neighborhood/Easy/P018",
  "neighborhood/neighborhood/Easy/P019",
  "neighborhood/neighborhood/Easy/P020",
  "neighborhood/neighborhood/Easy/P021",
  "neighborhood/neighborhood/Hard/P000",
  "neighborhood/neighborhood/Hard/P001",
  "neighborhood/neighborhood/Hard/P002",
  "neighborhood/neighborhood/Hard/P003",
  "neighborhood/neighborhood/Hard/P004",
  "neighborhood/neighborhood/Hard/P005",
  "neighborhood/neighborhood/Hard/P006",
  "neighborhood/neighborhood/Hard/P007",
  "neighborhood/neighborhood/Hard/P008",
  "neighborhood/neighborhood/Hard/P009",
  "neighborhood/neighborhood/Hard/P010",
  "neighborhood/neighborhood/Hard/P011",
  "neighborhood/neighborhood/Hard/P012",
  "neighborhood/neighborhood/Hard/P013",
  "neighborhood/neighborhood/Hard/P014",
  "neighborhood/neighborhood/Hard/P015",
  "neighborhood/neighborhood/Hard/P016",
  "neighborhood/neighborhood/Hard/P017",
  "ocean/ocean/Easy/P000",
  "ocean/ocean/Easy/P001",
  "ocean/ocean/Easy/P002",
  "ocean/ocean/Easy/P004",
  "ocean/ocean/Easy/P005",
  "ocean/ocean/Easy/P006",
  "ocean/ocean/Easy/P008",
  "ocean/ocean/Easy/P009",
  "ocean/ocean/Easy/P010",
  "ocean/ocean/Easy/P011",
  "ocean/ocean/Easy/P012",
  "ocean/ocean/Easy/P013",
  "ocean/ocean/Hard/P000",
  "ocean/ocean/Hard/P001",
  "ocean/ocean/Hard/P002",
  "ocean/ocean/Hard/P003",
  "ocean/ocean/Hard/P004",
  "ocean/ocean/Hard/P005",
  "ocean/ocean/Hard/P006",
  "ocean/ocean/Hard/P007",
  "ocean/ocean/Hard/P008",
  "ocean/ocean/Hard/P009",
  "office/office/Easy/P000",
  "office/office/Easy/P001",
  "office/office/Easy/P002",
  "office/office/Easy/P003",
  "office/office/Easy/P004",
  "office/office/Easy/P005",
  "office/office/Easy/P006",
  "office/office/Hard/P000",
  "office/office/Hard/P001",
  "office/office/Hard/P002",
  "office/office/Hard/P003",
  "office/office/Hard/P004",
  "office/office/Hard/P005",
  "office/office/Hard/P006",
  "office/office/Hard/P007",
  "office2/office2/Easy/P000",
  "office2/office2/Easy/P003",
  "office2/office2/Easy/P004",
  "office2/office2/Easy/P005",
  "office2/office2/Easy/P006",
  "office2/office2/Easy/P007",
  "office2/office2/Easy/P008",
  "office2/office2/Easy/P009",
  "office2/office2/Easy/P010",
  "office2/office2/Easy/P011",
  "office2/office2/Hard/P000",
  "office2/office2/Hard/P001",
  "office2/office2/Hard/P002",
  "office2/office2/Hard/P003",
  "office2/office2/Hard/P004",
  "office2/office2/Hard/P005",
  "office2/office2/Hard/P006",
  "office2/office2/Hard/P007",
  "office2/office2/Hard/P008",
  "office2/office2/Hard/P009",
  "office2/office2/Hard/P010",
  "oldtown/oldtown/Easy/P000",
  "oldtown/oldtown/Easy/P001",
  "oldtown/oldtown/Easy/P002",
  "oldtown/oldtown/Easy/P004",
  "oldtown/oldtown/Easy/P005",
  "oldtown/oldtown/Easy/P007",
  "oldtown/oldtown/Hard/P000",
  "oldtown/oldtown/Hard/P001",
  "oldtown/oldtown/Hard/P002",
  "oldtown/oldtown/Hard/P003",
  "oldtown/oldtown/Hard/P004",
  "oldtown/oldtown/Hard/P005",
  "oldtown/oldtown/Hard/P006",
  "oldtown/oldtown/Hard/P007",
  "oldtown/oldtown/Hard/P008",
  "seasidetown/seasidetown/Easy/P000",
  "seasidetown/seasidetown/Easy/P001",
  "seasidetown/seasidetown/Easy/P002",
  "seasidetown/seasidetown/Easy/P003",
  "seasidetown/seasidetown/Easy/P004",
  "seasidetown/seasidetown/Easy/P005",
  "seasidetown/seasidetown/Easy/P006",
  "seasidetown/seasidetown/Easy/P007",
  "seasidetown/seasidetown/Easy/P008",
  "seasidetown/seasidetown/Easy/P009",
  "seasidetown/seasidetown/Hard/P000",
  "seasidetown/seasidetown/Hard/P001",
  "seasidetown/seasidetown/Hard/P002",
  "seasidetown/seasidetown/Hard/P003",
  "seasidetown/seasidetown/Hard/P004",
  "seasonsforest/seasonsforest/Easy/P001",
  "seasonsforest/seasonsforest/Easy/P002",
  "seasonsforest/seasonsforest/Easy/P003",
  "seasonsforest/seasonsforest/Easy/P004",
  "seasonsforest/seasonsforest/Easy/P005",
  "seasonsforest/seasonsforest/Easy/P007",
  "seasonsforest/seasonsforest/Easy/P008",
  "seasonsforest/seasonsforest/Easy/P009",
  "seasonsforest/seasonsforest/Easy/P010",
  "seasonsforest/seasonsforest/Easy/P011",
  "seasonsforest/seasonsforest/Hard/P001",
  "seasonsforest/seasonsforest/Hard/P002",
  "seasonsforest/seasonsforest/Hard/P004",
  "seasonsforest/seasonsforest/Hard/P005",
  "seasonsforest/seasonsforest/Hard/P006",
  "seasonsforest_winter/seasonsforest_winter/Easy/P000",
  "seasonsforest_winter/seasonsforest_winter/Easy/P001",
  "seasonsforest_winter/seasonsforest_winter/Easy/P002",
  "seasonsforest_winter/seasonsforest_winter/Easy/P003",
  "seasonsforest_winter/seasonsforest_winter/Easy/P004",
  "seasonsforest_winter/seasonsforest_winter/Easy/P005",
  "seasonsforest_winter/seasonsforest_winter/Easy/P006",
  "seasonsforest_winter/seasonsforest_winter/Easy/P007",
  "seasonsforest_winter/seasonsforest_winter/Easy/P008",
  "seasonsforest_winter/seasonsforest_winter/Easy/P009",
  "seasonsforest_winter/seasonsforest_winter/Hard/P010",
  "seasonsforest_winter/seasonsforest_winter/Hard/P011",
  "seasonsforest_winter/seasonsforest_winter/Hard/P012",
  "seasonsforest_winter/seasonsforest_winter/Hard/P013",
  "seasonsforest_winter/seasonsforest_winter/Hard/P014",
  "seasonsforest_winter/seasonsforest_winter/Hard/P015",
  "seasonsforest_winter/seasonsforest_winter/Hard/P016",
  "seasonsforest_winter/seasonsforest_winter/Hard/P017",
  "seasonsforest_winter/seasonsforest_winter/Hard/P018",
  "soulcity/soulcity/Easy/P000",
  "soulcity/soulcity/Easy/P001",
  "soulcity/soulcity/Easy/P002",
  "soulcity/soulcity/Easy/P003",
  "soulcity/soulcity/Easy/P004",
  "soulcity/soulcity/Easy/P005",
  "soulcity/soulcity/Easy/P006",
  "soulcity/soulcity/Easy/P007",
  "soulcity/soulcity/Easy/P008",
  "soulcity/soulcity/Easy/P009",
  "soulcity/soulcity/Easy/P010",
  "soulcity/soulcity/Easy/P011",
  "soulcity/soulcity/Easy/P012",
  "soulcity/soulcity/Hard/P000",
  "soulcity/soulcity/Hard/P001",
  "soulcity/soulcity/Hard/P002",
  "soulcity/soulcity/Hard/P003",
  "soulcity/soulcity/Hard/P004",
  "soulcity/soulcity/Hard/P005",
  "soulcity/soulcity/Hard/P008",
  "soulcity/soulcity/Hard/P009",
  "westerndesert/westerndesert/Easy/P001",
  "westerndesert/westerndesert/Easy/P002",
  "westerndesert/westerndesert/Easy/P004",
  "westerndesert/westerndesert/Easy/P005",
  "westerndesert/westerndesert/Easy/P006",
  "westerndesert/westerndesert/Easy/P007",
  "westerndesert/westerndesert/Easy/P008",
  "westerndesert/westerndesert/Easy/P009",
  "westerndesert/westerndesert/Easy/P010",
  "westerndesert/westerndesert/Easy/P011",
  "westerndesert/westerndesert/Easy/P012",
  "westerndesert/westerndesert/Easy/P013",
  "westerndesert/westerndesert/Hard/P000",
  "westerndesert/westerndesert/Hard/P001",
  "westerndesert/westerndesert/Hard/P002",
  "westerndesert/westerndesert/Hard/P003",
  "westerndesert/westerndesert/Hard/P004",
  "westerndesert/westerndesert/Hard/P005",
  "westerndesert/westerndesert/Hard/P006",
  "westerndesert/westerndesert/Hard/P007"
]
rmse_log = []

def plot(scene):

      for trail_idx in range(0, 3):
        leagal_scene_name = scene.replace("/", "_")
        if gt == 1:
          dynamic_log_picke_file = open("dynamic_slam_log/logs/dynamic_slam_log_gt_difficulty_validate_"+leagal_scene_name+"_trials_"+str(trail_idx)+".pickle", "rb")
        else:
          dynamic_log_picke_file = open("dynamic_slam_log/logs/dynamic_slam_log_min_config_comp_"+leagal_scene_name+"_trials_"+str(trail_idx)+".pickle", "rb")
        logged_data = pickle.load(dynamic_log_picke_file)

        errory_array    = logged_data['result'].np_arrays['error_array']
        error_tstamp    = logged_data['result'].np_arrays['timestamps']
        est_xyz          = logged_data['result'].trajectories['traj'].positions_xyz
        ref_xyz          = logged_data['result'].trajectories['reference'].positions_xyz
        confidence      = logged_data['confidence']
        traslation      = logged_data['translation']
        tstamp          = logged_data['tstamp']
        if gt == 1:
          patches_log     = logged_data['num_patches']
          real_time_diffc = logged_data['diffculty_log']
        else:
          patches_log     = [16]*len(confidence)
          real_time_diffc = [1]*len(confidence)

        print(scene+"_trials_"+str(trail_idx) + " mean error: " + str(logged_data['result'].stats['mean']))


        for start_idx in range(0, len(tstamp)):
          if confidence[start_idx] == None:
            continue
          else:
            tstamp          = tstamp[start_idx:]
            confidence      = confidence[start_idx:]
            patches_log     = patches_log[start_idx:]
            real_time_diffc = real_time_diffc[start_idx:]
            traslation      = traslation[start_idx:]
            break

        if tstamp[0] < error_tstamp[0]:
          #tstamp start earlier
          for start_idx in range(0, len(tstamp)):
            if tstamp[start_idx] == error_tstamp[0]:
              tstamp         = tstamp[start_idx:]
              confidence     = confidence[start_idx:]
              patches_log    = patches_log[start_idx:]
              real_time_diffc= real_time_diffc[start_idx:]
              traslation     = traslation[start_idx:]
              break
        else:
          #ref start earlier
          for start_idx in range(0, len(error_tstamp)):
            if tstamp[0] == error_tstamp[start_idx]:
              errory_array = errory_array[start_idx:]
              error_tstamp = error_tstamp[start_idx:]
              est_xyz      = est_xyz[start_idx:]
              ref_xyz      = ref_xyz[start_idx:]
              break

        if tstamp[-1] > error_tstamp[-1]:
          #tstamp end later
          for end_idx in range(len(tstamp) - 1, 0, -1):
            if tstamp[end_idx] == error_tstamp[-1]:
              tstamp         = tstamp[:end_idx+1]
              confidence     = confidence[:end_idx+1]
              real_time_diffc= real_time_diffc[:end_idx+1]
              patches_log    = patches_log[:end_idx+1]
              traslation     = traslation[:end_idx+1]
              break
        else:
          #ref ends later
          for end_idx in range(len(error_tstamp) - 1, 0 ,-1):
            if tstamp[-1] == error_tstamp[end_idx]:
              errory_array = errory_array[:end_idx+1]
              error_tstamp = error_tstamp[:end_idx+1]
              est_xyz      = est_xyz[:end_idx+1]
              ref_xyz      = ref_xyz[:end_idx+1]
              break

        #clip initialization
        initialization_len = 180
        errory_array  = errory_array[initialization_len:]
        error_tstamp  = error_tstamp[initialization_len:]
        est_xyz       = est_xyz[initialization_len:]
        ref_xyz       = ref_xyz[initialization_len:]

        tstamp        = tstamp[initialization_len:]
        confidence    = confidence[initialization_len:]
        real_time_diffc = real_time_diffc[initialization_len:]
        patches_log   = patches_log[initialization_len:]
        traslation    = traslation[initialization_len:]

        rmse_log.append([scene+"_trials_"+str(trail_idx), sum(errory_array)/len(errory_array), sum(patches_log)/len(patches_log), len(patches_log)])

        # #compute change of drift
        # change_of_drift = []
        # error_vec_log = []

        # for idx in range(0, len(est_xyz)):

        #   error_vec = est_xyz[idx] - ref_xyz[idx]

        #   error_vec_log.append(error_vec)

        #   if idx == 0:
        #     change_of_drift.append(0)
        #     continue

        #   change_of_error_vec = error_vec - error_vec_log[idx - 1]

        #   change_of_error = math.sqrt(change_of_error_vec[0]**2 + change_of_error_vec[1]**2 + change_of_error_vec[2]**2)

        #   change_of_drift.append(change_of_error)


        # #smooth drift by finding the max value over past 10 frame
        # smooth_len = 10
        # smoothed_drift = []
        # for idx in range(0, len(change_of_drift)):
        #   if idx < smooth_len:
        #     smoothed_drift.append(0)
        #     continue

        #   smoothed_drift.append(max(change_of_drift[idx-smooth_len:idx]))

        # if gt == 1:
        #   output_folder = "dynamic_slam_log/outputs/dynamic_slam_log_gt_difficulty_validate_"+scene+"_trail_"+str(trail_idx)+"/"
        # else:
        #   output_folder = "dynamic_slam_log/outputs/dynamic_slam_log_full_setup_"+scene+"_trail_"+str(trail_idx)+"/"
        # os.makedirs(output_folder, exist_ok=True)

        # for tstamp_idx in range(0, len(error_tstamp)):
        #   if single_plot == 1:
        #     tstamp_idx = len(error_tstamp) - 1

        #   print(tstamp_idx)

        #   #fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(40, 8))
        #   fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 16))
          
          
          
        #   ax1.plot(tstamp[0:tstamp_idx], errory_array[0:tstamp_idx], 'c-', label='ERROR', linewidth=7)
        #   ax1.set_ylabel('ERROR ', color='c')
        #   ax1.set_xlabel('time stamp')
        #   ax1.set_xlim(min(tstamp), max(tstamp))
        #   #ax1.set_ylim(0, 1)
        #   ax1.tick_params(axis='y', labelcolor='c')
          
        #   #confidence = [x / num_patch for x in confidence]
        #   ax3 = ax1.twinx()
        #   ax3.plot(tstamp[0:tstamp_idx], smoothed_drift[0:tstamp_idx], 'b-', label='smoothed_drift')
        #   ax3.set_ylabel('smoothed_drift', color='b')
        #   ax3.set_xlim(min(tstamp), max(tstamp))
        #   #ax3.set_ylim(min(distance), max(distance_f_S))
        #   ax3.tick_params(axis='y', labelcolor='b')

        #   ax4 = ax1.twinx()
        #   ax4.spines['right'].set_position(('outward', 60))  # Move it 60 points outward
        #   ax4.plot(tstamp[0:tstamp_idx], real_time_diffc[0:tstamp_idx], 'r-', label='real_time_diffc')
        #   ax4.set_ylabel('real_time_diffc', color='r')
        #   ax4.set_xlim(min(tstamp), max(tstamp))
        #   #ax4.set_ylim(0, 1)
        #   ax4.tick_params(axis='y', labelcolor='r')

        #   ax5 = ax1.twinx()
        #   ax5.plot(tstamp[0:tstamp_idx], patches_log[0:tstamp_idx], 'g-', label='patches_log')
        #   ax5.set_ylabel('patches_log', color='g')
        #   ax5.set_xlim(min(tstamp), max(tstamp))
        #   #ax5.set_ylim(min(depth), max(depth))
        #   ax5.spines['right'].set_position(('outward', 120))  
        #   ax5.tick_params(axis='y', labelcolor='g')

        #   # ax6 = ax1.twinx()
        #   # ax6.plot(tstamp[0:tstamp_idx], traslation[0:tstamp_idx], 'y-', label='depth')
        #   # ax6.set_ylabel('movement axis', color='y')
        #   # ax6.set_xlim(min(tstamp), max(tstamp))
        #   # ax6.set_ylim(0, 5)
        #   # ax6.spines['right'].set_position(('outward', 180))  
        #   # ax6.tick_params(axis='y', labelcolor='y')

        #   # Load and display a PNG image in the right subplot
        #   image = mpimg.imread(dataset_path+scene+'/image_left/'+str(int(error_tstamp[tstamp_idx])).zfill(6)+'_left.png')
        #   low_res_image = rescale(image, scale=0.25, anti_aliasing=True)
        #   ax2.imshow(low_res_image, cmap='gray')
        #   ax2.axis('off')  # Turn off the axis for the image
        #   ax2.set_title('dataset')

        #   # Optionally add a title
        #   plt.title('')

        #   if os.path.exists(output_folder+"dynamic"+str(int(error_tstamp[tstamp_idx]))+".png"):
        #     os.remove(output_folder+"dynamic"+str(int(error_tstamp[tstamp_idx]))+".png")

        #   # Save the plot to a file
        #   plt.savefig(output_folder+"dynamic"+str(int(error_tstamp[tstamp_idx]))+".png", dpi=100, bbox_inches='tight')
        #   plt.close('all')

        #   if single_plot == 1:
        #     break

def log_error():

  rmse_cvs_file = "dynamic_slam_log/outputs/error.csv"

  with open(rmse_cvs_file, mode='w', newline='') as file:
      writer = csv.writer(file)
      writer.writerows(rmse_log)


if __name__ == "__main__":
  # with multiprocessing.Pool(processes=16) as pool:
  #     # Map the function to the list of numbers
  #     pool.map(plot, tartan_scenes)
  for scene in tartan_scenes:
    plot(scene)

  log_error()