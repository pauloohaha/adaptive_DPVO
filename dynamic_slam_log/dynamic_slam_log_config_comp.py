
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from skimage.transform import rescale

#compare the error of different setups

dataset_path = '/pool0/piaodeng/DROID-SLAM/datasets/EuRoC/'

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
for scene in euroc_scenes:
  for num_patch in range(96, 0, -16):
    for num_frame in range(22, 0, -6):

      dynamic_log_picke_file = open("dynamic_slam_log/logs/dynamic_slam_log_"+scene+"_"+str(num_patch)+"_patches_"+str(num_frame)+"_frames_trials+"+str(0)+".pickle", "rb")
      logged_data = pickle.load(dynamic_log_picke_file)


      error_tstamp    = logged_data[0]['result'].np_arrays['timestamps']
      errory_array_0  = logged_data[0]['result'].np_arrays['error_array']
      errory_array_1  = logged_data[1]['result'].np_arrays['error_array']
      errory_array_2  = logged_data[2]['result'].np_arrays['error_array']
      errory_array_3  = logged_data[3]['result'].np_arrays['error_array']
      errory_array_4  = logged_data[4]['result'].np_arrays['error_array']

      max_error_array = {
        max(errory_array_0),
        max(errory_array_1),
        max(errory_array_2),
        max(errory_array_3),
        max(errory_array_4)
      }

      max_error = max(max_error_array)

      min_error_array = {
        min(errory_array_0),
        min(errory_array_1),
        min(errory_array_2),
        min(errory_array_3),
        min(errory_array_4)
      }

      min_error = min(min_error_array)

      output_folder = "dynamic_slam_log/outputs/dynamic_slam_log_"+scene+"_trails/"
      os.makedirs(output_folder, exist_ok=True)

      #for tstamp_idx in range(0, len(error_tstamp)):
      tstamp_idx = len(error_tstamp)-1
      print(tstamp_idx)

      fig, (ax1, ax7) = plt.subplots(1, 2, figsize=(40, 8))
      ax2 = ax1.twinx()
      ax3 = ax1.twinx()
      ax4 = ax1.twinx()
      ax5 = ax1.twinx()
      
      ax1.plot(error_tstamp[0:tstamp_idx], errory_array_0[0:tstamp_idx], 'c-', label='ERROR', linewidth=7)
      ax1.set_ylabel('trail 0', color='c')
      ax1.set_xlabel('time stamp')
      ax1.set_xlim(min(error_tstamp), max(error_tstamp))
      ax1.set_ylim(min_error, max_error)
      ax1.tick_params(axis='y', labelcolor='c')
      
      ax2.plot(error_tstamp[0:tstamp_idx], errory_array_1[0:tstamp_idx], 'b-', label='confidence')
      ax2.set_ylabel('trail 1', color='b')
      ax2.set_xlim(min(error_tstamp), max(error_tstamp))
      ax2.set_ylim(min_error, max_error)
      ax2.tick_params(axis='y', labelcolor='b')

      ax3.spines['right'].set_position(('outward', 60))  # Move it 60 points outward
      ax3.plot(error_tstamp[0:tstamp_idx], errory_array_2[0:tstamp_idx], 'r-', label='delta')
      ax3.set_ylabel('trail 2', color='r')
      ax3.set_xlim(min(error_tstamp), max(error_tstamp))
      ax3.set_ylim(min_error, max_error)
      ax3.tick_params(axis='y', labelcolor='r')

      ax4.plot(error_tstamp[0:tstamp_idx], errory_array_3[0:tstamp_idx], 'g-', label='depth')
      ax4.set_ylabel('trail 3', color='g')
      ax4.set_xlim(min(error_tstamp), max(error_tstamp))
      ax4.set_ylim(min_error, max_error)
      ax4.spines['right'].set_position(('outward', 120))  
      ax4.tick_params(axis='y', labelcolor='g')

      ax5.plot(error_tstamp[0:tstamp_idx], errory_array_4[0:tstamp_idx], 'm-', label='depth')
      ax5.set_ylabel('trail 4', color='m')
      ax5.set_xlim(min(error_tstamp), max(error_tstamp))
      ax5.set_ylim(min_error, max_error)
      ax5.spines['right'].set_position(('outward', 180))  
      ax5.tick_params(axis='y', labelcolor='m')


      # Load and display a PNG image in the right subplot
      image = mpimg.imread(dataset_path+scene+'/mav0/cam0/data/'+str(int(error_tstamp[tstamp_idx]))+'.png')
      low_res_image = rescale(image, scale=0.25, anti_aliasing=True)
      ax7.imshow(low_res_image, cmap='gray')
      ax7.axis('off')  # Turn off the axis for the image
      ax7.set_title('dataset')

      # Optionally add a title
      plt.title('')

      # Save the plot to a file
      plt.savefig(output_folder+"dynamic"+str(int(error_tstamp[tstamp_idx]))+".png", dpi=100, bbox_inches='tight')
      plt.close('all')

