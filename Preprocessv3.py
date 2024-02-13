import cv2
import face_recognition
import os
import random

def extract_faces(video_path, output_folder):
    # Load the video
    video_capture = cv2.VideoCapture(video_path)

    # Get the frames per second (fps) of the input video
    fps = video_capture.get(cv2.CAP_PROP_FPS)

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Counter for extracted frames
    frame_counter = 0

    # Limit the number of frames per video to 10
    max_frames_per_video = 20

    # List to store frame indexes for random selection
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indexes = random.sample(range(total_frames), min(total_frames, max_frames_per_video))

    # Iterate over selected frame indexes
    for frame_index in frame_indexes:
        # Set frame position
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

        # Read the current frame
        ret, frame = video_capture.read()

        if not ret:
            break

        # Find all face locations in the current frame
        face_locations = face_recognition.face_locations(frame)

        # Extract faces and save them as images
        for face_location in face_locations:
            top, right, bottom, left = face_location
            face_image = frame[top:bottom, left:right]

            # Resize the face image for consistency
            face_image = cv2.resize(face_image, (224, 224))

            # Save the face image to the output folder
            face_filename = os.path.join(
                output_folder, f"{os.path.basename(video_path)}_frame_{frame_index}_face.jpg")
            cv2.imwrite(face_filename, face_image)

    # Release the video capture object
    video_capture.release()


if __name__ == "__main__":
    # Specify the path to the dataset folder containing videos
    dataset_folder = "Celeb-DF/YouTube-real"

    # Specify the output folder for extracted face frames
    output_folder = "OP-Real"

    # Loop through all video files in the dataset folder
    for filename in os.listdir(dataset_folder):
        if filename.endswith(".mp4") or filename.endswith(".avi"):
            video_path = os.path.join(dataset_folder, filename)
            extract_faces(video_path, output_folder)

    print("Face extraction completed.")
