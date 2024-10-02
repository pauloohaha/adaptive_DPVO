import cv2
import os

def play_and_save_images_as_video(image_folder, output_video_path, fps=30):
    # Get all the image files from the directory
    images = [img for img in sorted(os.listdir(image_folder)) if img.endswith((".png", ".jpg", ".jpeg"))]
    
    if not images:
        print("No images found in the folder!")
        return
    
    # Load the first image to get the size (assumes all images are the same size)
    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    height, width, _ = frame.shape
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for .avi file (use 'mp4v' for .mp4)
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    # Create a named window for displaying the video
    cv2.namedWindow("Image Video", cv2.WINDOW_NORMAL)
    
    # Play and save images as video
    for image_file in images:
        image_path = os.path.join(image_folder, image_file)
        frame = cv2.imread(image_path)
        
        if frame is None:
            print(f"Error loading image: {image_file}")
            continue
        
        # Display the frame
        cv2.imshow("Image Video", frame)
        
        # Write the frame to the video file
        out.write(frame)
        
        # Wait for the specified time depending on fps (1000ms / fps)
        if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
            break

    # Release the VideoWriter object and close windows
    out.release()
    cv2.destroyAllWindows()


# Usage example: Pass the folder containing images, output video path, and specify FPS
image_folder = '/pool0/piaodeng/distributed_dpvo/dynamic_slam_log/outputs/dynamic_slam_log_movement_MH_03_medium_96_patches_22_frames_trail_0'
output_video_path = image_folder+'.avi'  # You can change this to output.mp4 if using 'mp4v' codec
play_and_save_images_as_video(image_folder, output_video_path, fps=30)