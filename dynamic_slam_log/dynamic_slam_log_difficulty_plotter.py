
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from skimage.transform import rescale
import math
import csv

single_plot = 1

dataset_path = '/pool0/piaodeng/DROID-SLAM/datasets/EuRoC/'

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
    # "V2_03_difficult"
]
rmse_log = []
for scene in euroc_scenes:

    for trail_idx in range(0, 3):
      dynamic_log_picke_file = open("dynamic_slam_log/logs/dynamic_slam_log_gt_difficulty_validate_"+scene+"_trials_"+str(trail_idx)+".pickle", "rb")
      logged_data = pickle.load(dynamic_log_picke_file)

      errory_array    = logged_data['result'].np_arrays['error_array']
      error_tstamp    = logged_data['result'].np_arrays['timestamps']
      est_xyz          = logged_data['result'].trajectories['traj'].positions_xyz
      ref_xyz          = logged_data['result'].trajectories['reference'].positions_xyz
      confidence      = logged_data['confidence']
      traslation      = logged_data['translation']
      tstamp          = logged_data['tstamp']
      patches_log     = logged_data['num_patches']
      real_time_diffc = logged_data['diffculty_log']

      print(scene+"_trials_"+str(trail_idx) + " mean error: " + str(logged_data['result'].stats['mean']))

      rmse_log.append([scene+"_trials_"+str(trail_idx), logged_data['result'].stats['mean'], sum(patches_log)/len(patches_log), len(patches_log)])

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

      #compute change of drift
      change_of_drift = []
      error_vec_log = []

      for idx in range(0, len(est_xyz)):

        error_vec = est_xyz[idx] - ref_xyz[idx]

        error_vec_log.append(error_vec)

        if idx == 0:
          change_of_drift.append(0)
          continue

        change_of_error_vec = error_vec - error_vec_log[idx - 1]

        change_of_error = math.sqrt(change_of_error_vec[0]**2 + change_of_error_vec[1]**2 + change_of_error_vec[2]**2)

        change_of_drift.append(change_of_error)


      #smooth drift by finding the max value over past 10 frame
      smooth_len = 10
      smoothed_drift = []
      for idx in range(0, len(change_of_drift)):
        if idx < smooth_len:
          smoothed_drift.append(0)
          continue

        smoothed_drift.append(max(change_of_drift[idx-smooth_len:idx]))

      output_folder = "dynamic_slam_log/outputs/dynamic_slam_log_gt_difficulty_validate_"+scene+"_trail_"+str(trail_idx)+"/"
      os.makedirs(output_folder, exist_ok=True)

      for tstamp_idx in range(0, len(error_tstamp)):
        if single_plot == 1:
          tstamp_idx = len(error_tstamp) - 1

        print(tstamp_idx)

        #fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(40, 8))
        fig, (ax1) = plt.subplots(1, 1, figsize=(10, 8))
        
        
        
        ax1.plot(tstamp[0:tstamp_idx], errory_array[0:tstamp_idx], 'c-', label='ERROR', linewidth=7)
        ax1.set_ylabel('ERROR ', color='c')
        ax1.set_xlabel('time stamp')
        ax1.set_xlim(min(tstamp), max(tstamp))
        #ax1.set_ylim(0, 1)
        ax1.tick_params(axis='y', labelcolor='c')
        
        #confidence = [x / num_patch for x in confidence]
        ax3 = ax1.twinx()
        ax3.plot(tstamp[0:tstamp_idx], smoothed_drift[0:tstamp_idx], 'b-', label='distance_f_S')
        ax3.set_ylabel('distance', color='b')
        ax3.set_xlim(min(tstamp), max(tstamp))
        #ax3.set_ylim(min(distance), max(distance_f_S))
        ax3.tick_params(axis='y', labelcolor='b')

        ax4 = ax1.twinx()
        ax4.spines['right'].set_position(('outward', 60))  # Move it 60 points outward
        ax4.plot(tstamp[0:tstamp_idx], real_time_diffc[0:tstamp_idx], 'r-', label='real_time_diffc')
        ax4.set_ylabel('real_time_diffc', color='r')
        ax4.set_xlim(min(tstamp), max(tstamp))
        #ax4.set_ylim(0, 1)
        ax4.tick_params(axis='y', labelcolor='r')

        ax5 = ax1.twinx()
        ax5.plot(tstamp[0:tstamp_idx], patches_log[0:tstamp_idx], 'g-', label='patches_log')
        ax5.set_ylabel('patches_log', color='g')
        ax5.set_xlim(min(tstamp), max(tstamp))
        #ax5.set_ylim(min(depth), max(depth))
        ax5.spines['right'].set_position(('outward', 120))  
        ax5.tick_params(axis='y', labelcolor='g')

        # ax6 = ax1.twinx()
        # ax6.plot(tstamp[0:tstamp_idx], traslation[0:tstamp_idx], 'y-', label='depth')
        # ax6.set_ylabel('movement axis', color='y')
        # ax6.set_xlim(min(tstamp), max(tstamp))
        # ax6.set_ylim(0, 5)
        # ax6.spines['right'].set_position(('outward', 180))  
        # ax6.tick_params(axis='y', labelcolor='y')

        # Load and display a PNG image in the right subplot
        # image = mpimg.imread(dataset_path+scene+'/mav0/cam0/data/'+str(int(error_tstamp[tstamp_idx]))+'.png')
        # low_res_image = rescale(image, scale=0.25, anti_aliasing=True)
        # ax2.imshow(low_res_image, cmap='gray')
        # ax2.axis('off')  # Turn off the axis for the image
        # ax2.set_title('dataset')

        # Optionally add a title
        plt.title('')

        # Save the plot to a file
        plt.savefig(output_folder+"dynamic"+str(int(error_tstamp[tstamp_idx]))+".png", dpi=100, bbox_inches='tight')
        plt.close('all')

        if single_plot == 1:
          break


rmse_cvs_file = "dynamic_slam_log/outputs/error.csv"

with open(rmse_cvs_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(rmse_log)
