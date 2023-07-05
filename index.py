import cv2
import joblib
import numpy as np
# from folder_name.file_name import class_name
from helpers.converter import Converter

pre_processor = Converter("./config.json")

# Load the classifier
clf = joblib.load(pre_processor.config['model_file'])

# Create a video capture object
camera = cv2.VideoCapture(pre_processor.config['camera_number'])

# Initialize variables for fps calculation
frame_count = 0
start_time = cv2.getTickCount()

# Define button functions


def start_processing():
    global camera, frame_count, start_time
    if not camera.isOpened():
        raise ("Error opening Camera")

    # Start the video processing loop
    while camera.isOpened():
        ret, frame = camera.read()

        if ret:
            frame = cv2.flip(frame, 1)
            try:
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                cropped_frame, (x, y, w, h) = pre_processor.crop_and_resize(
                    frame_gray)
                extracted_f = pre_processor.extract_facial_features(
                    cropped_frame, "db1", 3)

                fx = clf.predict([np.array(extracted_f).flatten()])

                # Draw rectangle around the detected face
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                for key, item in enumerate(pre_processor.config['members']):
                    if (key+1) == fx:
                        print(f"[+] - {item.capitalize()} Detected")
                        cv2.putText(frame, item.capitalize(),  # Capitalize the first letter
                                    (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            except:
                print("[+] - No face detected")

            finally:
                # Calculate frames per second (fps)
                frame_count += 1
                elapsed_time = (cv2.getTickCount() -
                                start_time) / cv2.getTickFrequency()
                fps = int(frame_count / elapsed_time)

                # Display fps in red color at the top right corner
                fps_text = "FPS: {}".format(fps)
                cv2.putText(frame, fps_text, (10, 40),
                            cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 0, 255), 2)

                cv2.imshow('Video', frame)

                key = cv2.waitKey(25) & 0xFF
                if key == 27:  # "Esc" key
                    break
        else:
            return


def stop_processing():
    global camera
    # Release the video capture object
    camera.release()


if __name__ == "__main__":
    start_processing()
    stop_processing()
