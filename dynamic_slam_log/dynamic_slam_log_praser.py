
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from skimage.transform import rescale

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
    # "V1_03_difficult",
    # "V2_01_easy",
    # "V2_02_medium",
    "V2_03_difficult"
]
for scene in euroc_scenes:
  min_confidence = []
  ate = []
  for num_patch in range(96, 0, -16):
    for num_frame in range(22, 4, -6):
      for trail_idx in range(0, 3):
        dynamic_log_picke_file = open("dynamic_slam_log/logs/dynamic_slam_log_movement_outbound_"+scene+"_"+str(num_patch)+"_patches_"+str(num_frame)+"_frames_trials_"+str(trail_idx)+".pickle", "rb")
        logged_data = pickle.load(dynamic_log_picke_file)

        errory_array    = logged_data['result'].np_arrays['error_array']
        error_tstamp    = logged_data['result'].np_arrays['timestamps']
        inverted_depth  = logged_data['depth']
        confidence      = logged_data['confidence']
        delta           = logged_data['delta']
        traslation      = logged_data['translation']
        outlog          = logged_data['outlog']
        tstamp          = logged_data['tstamp']

        for start_idx in range(0, len(tstamp)):
          if tstamp[start_idx] == None:
            continue
          else:
            tstamp          = tstamp[start_idx:]
            inverted_depth  = inverted_depth[start_idx:]
            confidence      = confidence[start_idx:]
            delta           = delta[start_idx:]
            traslation      = traslation[start_idx:]
            outlog          = outlog[start_idx:]
            break

        if tstamp[0] < error_tstamp[0]:
          #tstamp start earlier
          for start_idx in range(0, len(tstamp)):
            if tstamp[start_idx] == error_tstamp[0]:
              tstamp         = tstamp[start_idx:]
              inverted_depth = inverted_depth[start_idx:]
              confidence     = confidence[start_idx:]
              delta          = delta[start_idx:]
              traslation     = traslation[start_idx:]
              outlog         = outlog[start_idx:]
              break
        else:
          #ref start earlier
          for start_idx in range(0, len(error_tstamp)):
            if tstamp[0] == error_tstamp[start_idx]:
              errory_array = errory_array[start_idx:]
              error_tstamp = error_tstamp[start_idx:]
              break

        if tstamp[-1] > error_tstamp[-1]:
          #tstamp end later
          for end_idx in range(len(tstamp) - 1, 0, -1):
            if tstamp[end_idx] == error_tstamp[-1]:
              tstamp         = tstamp[:end_idx+1]
              inverted_depth = inverted_depth[:end_idx+1]
              confidence     = confidence[:end_idx+1]
              delta          = delta[:end_idx+1]
              traslation     = traslation[:end_idx+1]
              outlog         = outlog[:end_idx+1]
              break
        else:
          #ref ends later
          for end_idx in range(len(error_tstamp) - 1, 0 ,-1):
            if tstamp[-1] == error_tstamp[end_idx]:
              errory_array = errory_array[:end_idx+1]
              error_tstamp = error_tstamp[:end_idx+1]
              break

        depth = [1/x for x in inverted_depth]

        min_confidence.append(min(confidence))
        ate.append(logged_data['result'].stats["rmse"])


        error_change = []
        for i in range(0, len(errory_array)):
          if i < 50:
            error_change.append(0)
            continue
          else:
            error_change.append(abs(errory_array[i] - errory_array[i-50]))

        output_folder = "dynamic_slam_log/outputs/dynamic_slam_log_movement_outbound_"+scene+"_"+str(num_patch)+"_patches_"+str(num_frame)+"_frames_trail_"+str(trail_idx)+"/"
        os.makedirs(output_folder, exist_ok=True)

        for tstamp_idx in range(0, len(error_tstamp)):
          if single_plot == 1:
            tstamp_idx = len(error_tstamp) - 1

          print(tstamp_idx)

          fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(40, 8))
          ax3 = ax1.twinx()
          ax4 = ax1.twinx()
          
          ax6 = ax1.twinx()
          
          ax1.plot(tstamp[0:tstamp_idx], errory_array[0:tstamp_idx], 'c-', label='ERROR', linewidth=7)
          ax1.set_ylabel('ERROR change', color='c')
          ax1.set_xlabel('time stamp')
          ax1.set_xlim(min(tstamp), max(tstamp))
          ax1.set_ylim(0, 1)
          ax1.tick_params(axis='y', labelcolor='c')
          
          #confidence = [x / num_patch for x in confidence]
          ax3.plot(tstamp[0:tstamp_idx], confidence[0:tstamp_idx], 'b-', label='confidence')
          ax3.set_ylabel('confidence', color='b')
          ax3.set_xlim(min(tstamp), max(tstamp))
          ax3.set_ylim(min(confidence), max(confidence))
          ax3.tick_params(axis='y', labelcolor='b')

          ax4.spines['right'].set_position(('outward', 60))  # Move it 60 points outward
          ax4.plot(tstamp[0:tstamp_idx], outlog[0:tstamp_idx], 'r-', label='outlog')
          ax4.set_ylabel('outlog', color='r')
          ax4.set_xlim(min(tstamp), max(tstamp))
          ax4.set_ylim(0, 1)
          ax4.tick_params(axis='y', labelcolor='r')

          # ax5 = ax1.twinx()
          # ax5.plot(tstamp[0:tstamp_idx], depth[0:tstamp_idx], 'g-', label='depth')
          # ax5.set_ylabel('depth axis', color='g')
          # ax5.set_xlim(min(tstamp), max(tstamp))
          # ax5.set_ylim(min(depth), max(depth))
          # ax5.spines['right'].set_position(('outward', 120))  
          # ax5.tick_params(axis='y', labelcolor='g')

          ax6.plot(tstamp[0:tstamp_idx], traslation[0:tstamp_idx], 'y-', label='depth')
          ax6.set_ylabel('movement axis', color='y')
          ax6.set_xlim(min(tstamp), max(tstamp))
          ax6.set_ylim(0, 5)
          ax6.spines['right'].set_position(('outward', 180))  
          ax6.tick_params(axis='y', labelcolor='y')

          # Load and display a PNG image in the right subplot
          image = mpimg.imread(dataset_path+scene+'/mav0/cam0/data/'+str(int(error_tstamp[tstamp_idx]))+'.png')
          low_res_image = rescale(image, scale=0.25, anti_aliasing=True)
          ax2.imshow(low_res_image, cmap='gray')
          ax2.axis('off')  # Turn off the axis for the image
          ax2.set_title('dataset')

          # Optionally add a title
          plt.title('')

          # Save the plot to a file
          plt.savefig(output_folder+"dynamic"+str(int(error_tstamp[tstamp_idx]))+".png", dpi=100, bbox_inches='tight')
          plt.close('all')

          if single_plot == 1:
            break

  fig, ax1 = plt.subplots(1, 1, figsize=(10, 8))
  test_id = []
  for i in range(0, len(min_confidence)):
    test_id.append(i)

  ax1.plot(test_id, min_confidence, 'r-', label='min confidence')
  ax1.set_ylabel("min confidence", color='r')
  ax1.set_xlabel('test id')
  ax1.set_ylim(0, 90)
  ax1.tick_params(axis='y', labelcolor='r')

  ax2 = ax1.twinx()
  ax2.plot(test_id, ate, 'g-', label='ate')
  ax2.set_ylabel("ate", color='g')
  ax2.set_ylim(0, 0.7)
  ax2.tick_params(axis='y', labelcolor='g')
  plt.savefig("dynamic_slam_log/outputs/dynamic_slam_log_movement_outbound_"+scene+"_min_confidence_ate_comp"+".png", dpi=100, bbox_inches='tight')
  plt.close('all')

