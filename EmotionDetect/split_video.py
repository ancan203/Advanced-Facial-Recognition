import os
import shutil
import random
import cv2

def split_videos_to_frames(video_root, output_root, split_ratio=0.8):
    """
    Splits videos into frames and divides them into train and validation sets with labeling.

    Args:
        video_root (str): Path to the `Video` folder containing subfolders of gestures.
        output_root (str): Path to the root folder for train/validation datasets.
        split_ratio (float): Ratio of train frames (default: 0.8).
    """
    train_root = os.path.join(output_root, 'Train')
    val_root = os.path.join(output_root, 'Validation')

    # Ensure train and validation directories exist
    os.makedirs(train_root, exist_ok=True)
    os.makedirs(val_root, exist_ok=True)

    # Debug logging
    debug_log = os.path.join(output_root, "frame_extraction_log.txt")
    with open(debug_log, "w") as log:
        log.write("Frame Extraction Log\n")
        log.write("====================\n")

    # Iterate through gesture subfolders
    for gesture_class in os.listdir(video_root):
        if gesture_class == '.ipynb_checkpoints':  # Skip .ipynb_checkpoints folder
            continue

        class_folder = os.path.join(video_root, gesture_class)

        if not os.path.isdir(class_folder):
            continue

        # Create gesture-specific train and validation folders
        train_class_folder = os.path.join(train_root, gesture_class)
        val_class_folder = os.path.join(val_root, gesture_class)
        os.makedirs(train_class_folder, exist_ok=True)
        os.makedirs(val_class_folder, exist_ok=True)

        total_frames = 0
        train_count = 0
        val_count = 0

        # Process videos in the class folder
        for video_file in os.listdir(class_folder):
            video_path = os.path.join(class_folder, video_file)

            if not video_file.endswith(('.mp4', '.avi', '.MOV')):  # Check valid video formats
                continue

            # Extract frames from video
            vidcap = cv2.VideoCapture(video_path)
            success, frame = vidcap.read()
            frames = []
            frame_count = 0

            # Use gesture class name and video name as part of the frame identifier
            video_id = os.path.splitext(video_file)[0]
            while success:
                frame_name = f"{gesture_class}_{video_id}_frame_{frame_count}.jpg"
                temp_frame_path = os.path.join(output_root, "temp", frame_name)
                frames.append(temp_frame_path)

                # Save frame temporarily
                os.makedirs(os.path.join(output_root, "temp"), exist_ok=True)
                cv2.imwrite(temp_frame_path, frame)
                success, frame = vidcap.read()
                frame_count += 1

            vidcap.release()
            total_frames += frame_count

            # Shuffle frames and split into train/validation
            random.shuffle(frames)
            train_split = int(len(frames) * split_ratio)
            train_frames = frames[:train_split]
            val_frames = frames[train_split:]

            # Move frames to respective folders
            for frame in train_frames:
                dest_path = os.path.join(train_class_folder, os.path.basename(frame))
                try:
                    shutil.move(frame, dest_path)
                    train_count += 1
                except Exception as e:
                    print(f"Error moving frame {frame} to {dest_path}: {e}")

            for frame in val_frames:
                dest_path = os.path.join(val_class_folder, os.path.basename(frame))
                try:
                    shutil.move(frame, dest_path)
                    val_count += 1
                except Exception as e:
                    print(f"Error moving frame {frame} to {dest_path}: {e}")

        # Log frame information for the current class
        actual_train_count = len(os.listdir(train_class_folder))
        actual_val_count = len(os.listdir(val_class_folder))

        with open(debug_log, "a") as log:
            log.write(f"Class: {gesture_class}\n")
            log.write(f"  Total frames extracted: {total_frames}\n")
            log.write(f"  Train frames (expected): {train_count}, Actual: {actual_train_count}\n")
            log.write(f"  Validation frames (expected): {val_count}, Actual: {actual_val_count}\n")
            log.write("\n")
        print(f"Class: {gesture_class}, Total: {total_frames}, Train: {train_count} (Actual: {actual_train_count}), Validation: {val_count} (Actual: {actual_val_count})")

    # Clean up temp folder
    temp_folder = os.path.join(output_root, "temp")
    if os.path.exists(temp_folder):
        shutil.rmtree(temp_folder)

video_folder = "Dataset/Videos"
output_folder = "Dataset"
split_videos_to_frames(video_folder, output_folder)
