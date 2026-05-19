import cv2
import os

def avi_to_tiff(avi_file, output_folder):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the video file
    cap = cv2.VideoCapture(avi_file)

    frame_count = 0
    while True:
        # Read frame by frame
        ret, frame = cap.read()

        # If no frame is returned, we've reached the end of the video
        if not ret:
            break

        # Save each frame as a TIFF image
        output_file = os.path.join(output_folder, f"frame_{frame_count:04d}.tiff")
        cv2.imwrite(output_file, frame)

        frame_count += 1

    # Release the video capture object
    cap.release()
    print(f"Conversion complete. Total frames: {frame_count}")


