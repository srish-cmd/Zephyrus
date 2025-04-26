import cv2
import numpy as np
import math
import time
import serial
import threading
import pyttsx3
from ultralytics import YOLO
import cvzone
from ipcam import cam  # Assuming ipcam.py is available in the same directory

# --- TTS Setup ---
engine = pyttsx3.init()
tts_lock = threading.Lock()


def play_sound(text):
    """Function to convert text to speech using pyttsx3."""
    with tts_lock:  # Ensure that only one thread can access the TTS engine at a time
        engine.say(text)
        engine.runAndWait()


def play_sound_async(text):
    """Run play_sound in a separate thread to avoid blocking."""
    thread = threading.Thread(target=play_sound, args=(text,))
    thread.start()


# --- Configuration ---
boundary_marker_ids_map = {1: 'tl', 2: 'tr', 3: 'br', 4: 'bl'}
boundary_marker_ids = set(boundary_marker_ids_map.keys())
target_row = 2
target_col = 2
grid_rows = 5
grid_cols = 5
flat_grid_width = 500
flat_grid_height = 500

# ArUco configuration for boundary markers
aruco_dict_type = cv2.aruco.DICT_4X4_50

# YOLO configuration
model_path = "yolo11s.pt"  # Path to your YOLO model
coco_classes_path = "coco.txt"  # Path to COCO class names
target_object_class = "person"  # Change this to your target object class

# Serial configuration
serial_port = 'COM5'
baud_rate = 9600

# --- Pre-calculated servo angles ---
target_cell_pick_angles = {
    "approach_pick": [70, 50, 77, 36, 49, 64],
    "pick": [70, 50, 77, 36, 49, 64],
    "close_gripper": [70, 50, 77, 36, 49, 0],
    "lift": [65, 72, 77, 36, 49, 0],
}

common_angles = {
    "home": [62, 118, 153, 36, 49, 10],
    "approach_drop": [3, 85, 100, 36, 49, 0],
    "drop": [3, 108, 135, 36, 49, 0],
    "open_gripper": [3, 108, 135, 36, 49, 45],
}

# --- Serial Communication Functions ---
ser = None


def initialize_robot_serial():
    global ser
    try:
        ser = serial.Serial(serial_port, baud_rate, timeout=1)
        print(f"[serial] Attempting connection to {serial_port}...")
        time.sleep(2)
        if ser.is_open:
            print(f"[serial] Connected successfully to {serial_port} at {baud_rate} baud.")
            return True
        else:
            print(f"[serial] Failed to open port {serial_port}, but no exception.")
            return False
    except serial.SerialException as e:
        print(f"[serial] Error connecting to {serial_port}: {e}")
        ser = None
        return False
    except Exception as e:
        print(f"[serial] Non-serial error during connection: {e}")
        ser = None
        return False


def close_robot_serial():
    global ser
    if ser and ser.is_open:
        ser.close()
        print("[serial] Connection closed.")
        ser = None


def send_servo_angles(angle_list):
    global ser
    if not (ser and ser.is_open):
        print("[serial] Error: Not connected.")
        return False
    if len(angle_list) != 6:
        print(f"[serial] Error: Expected 6 angles, got {len(angle_list)}")
        return False
    try:
        cmd = ("s1{:03d}s2{:03d}s3{:03d}s4{:03d}s5{:03d}s6{:03d}".format(*angle_list))
        print(f"[serial] Sending: {cmd}")
        ser.write((cmd + "\n").encode('utf-8'))
        time.sleep(1.5)
        return True
    except serial.SerialException as e:
        print(f"[serial] Error sending data: {e}")
        return False
    except Exception as e:
        print(f"[serial] Unexpected error during send: {e}")
        return False


# --- Setup ---
# Load COCO class names
try:
    with open(coco_classes_path, "r") as f:
        class_names = f.read().splitlines()
    print(f"[info] Loaded {len(class_names)} class names from {coco_classes_path}")
except Exception as e:
    print(f"[error] Failed to load COCO class names: {e}")
    class_names = []
    exit()

# Load YOLO model
try:
    model = YOLO(model_path)
    print(f"[info] YOLO model loaded from {model_path}")
except Exception as e:
    print(f"[error] Failed to load YOLO model: {e}")
    exit()

# Initialize ArUco for boundary detection
aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
try:
    aruco_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    use_detector_object = True
except AttributeError:
    aruco_params = cv2.aruco.DetectorParameters_create()
    detector = None
    use_detector_object = False

# Initialize robot
if not initialize_robot_serial():
    print("error: cannot connect to robot controller. please check port and connection. exiting.")
    exit()

# Perspective transformation setup
dst_pts = np.array([[0, 0], [flat_grid_width - 1, 0],
                    [flat_grid_width - 1, flat_grid_height - 1],
                    [0, flat_grid_height - 1]], dtype='float32')

# Initialize state variables
object_picked_up = False
spoken_ids = set()  # To track already announced objects

# Move robot to home position
print("[robot] Moving to home position...")
if not send_servo_angles(common_angles["home"]):
    print("[robot] Error sending home command. Check connection.")
print("[info] Robot homed (or command sent). Starting detection loop.")

# --- Main Loop ---
try:
    while True:
        # Capture frame
        frame = cam()  # Using the IP camera function
        if frame is None:
            print("error: failed to capture frame.")
            continue

        frame = cv2.resize(frame, (1020, 500))

        # Detect boundary markers with ArUco
        boundary_corners_dict = {}
        if use_detector_object:
            corners, ids, rejected = detector.detectMarkers(frame)
        else:
            corners, ids, rejected = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=aruco_params)

        if ids is not None:
            flat_ids = ids.flatten()
            for i, marker_id in enumerate(flat_ids):
                if marker_id in boundary_marker_ids:
                    boundary_corners_dict[marker_id] = corners[i]

            # Draw ArUco markers
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        # Detect objects with YOLO
        results = model.track(frame, persist=True)
        object_marker_center = None
        object_row, object_col = -1, -1

        # Check if there are any objects detected by YOLO
        if results[0].boxes is not None and results[0].boxes.id is not None:
            # Get boxes, class IDs, track IDs
            boxes = results[0].boxes.xyxy.int().cpu().tolist()
            class_ids = results[0].boxes.cls.int().cpu().tolist()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            confidences = results[0].boxes.conf.cpu().tolist()

            # Process detected objects
            for box, class_id, track_id, conf in zip(boxes, class_ids, track_ids, confidences):
                class_name = class_names[class_id]
                x1, y1, x2, y2 = box

                # Draw detection on frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cvzone.putTextRect(frame, f'{track_id}', (x1, y2), 1, 1)
                cvzone.putTextRect(frame, f'{class_name}', (x1, y1), 1, 1)

                # Announce new detections (optional)
                if track_id not in spoken_ids:
                    spoken_ids.add(track_id)
                    play_sound_async(f"Detected {class_name}")

                # If this is our target object class, use it for robot control
                if class_name == target_object_class:
                    # Calculate center of the bounding box
                    object_marker_center = ((x1 + x2) // 2, (y1 + y2) // 2)

        # If all boundary markers are detected, calculate perspective transform
        if len(boundary_corners_dict) == len(boundary_marker_ids):
            src_pts_list = [None] * 4
            valid_corners = True

            for marker_id, corner_role in boundary_marker_ids_map.items():
                if marker_id not in boundary_corners_dict:
                    valid_corners = False
                    break

                marker_corners = boundary_corners_dict[marker_id].reshape((4, 2))
                if corner_role == 'tl':
                    src_pts_list[0] = marker_corners[0]
                elif corner_role == 'tr':
                    src_pts_list[1] = marker_corners[1]
                elif corner_role == 'br':
                    src_pts_list[2] = marker_corners[2]
                elif corner_role == 'bl':
                    src_pts_list[3] = marker_corners[3]

            if valid_corners and all(pt is not None for pt in src_pts_list):
                src_pts = np.array(src_pts_list, dtype='float32')
                cv2.polylines(frame, [np.int32(src_pts)], isClosed=True, color=(255, 0, 0), thickness=2)

                # Calculate perspective transform
                matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)

                # If object is detected, map it to grid coordinates
                if object_marker_center is not None:
                    obj_center_np = np.array([[object_marker_center]], dtype='float32')
                    transformed_point = cv2.perspectiveTransform(obj_center_np, matrix)

                    if transformed_point is not None:
                        tx, ty = transformed_point[0][0]
                        cell_width = flat_grid_width / grid_cols
                        cell_height = flat_grid_height / grid_rows

                        col_index = max(0, min(math.floor(tx / cell_width), grid_cols - 1))
                        row_index = max(0, min(math.floor(ty / cell_height), grid_rows - 1))

                        object_row, object_col = row_index, col_index

                        # Display grid position
                        text = f"grid: ({object_row}, {object_col})"
                        cv2.putText(frame, text, (object_marker_center[0] + 10, object_marker_center[1]),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                        # Check if object is at target position
                        if object_row == target_row and object_col == target_col:
                            cv2.putText(frame, f"target ({object_row}, {object_col})", (10, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                            if not object_picked_up:
                                print(
                                    f"\n>>> object detected at target ({object_row}, {object_col}). starting sequence <<<")
                                play_sound_async("Object at target location. Starting pick sequence.")
                                object_picked_up = True

                                print("[robot] moving to approach pick")
                                success = send_servo_angles(target_cell_pick_angles["approach_pick"])

                                if success:
                                    print("[robot] moving to pick height")
                                    success = send_servo_angles(target_cell_pick_angles["pick"])

                                if success:
                                    print("[robot] closing gripper")
                                    success = send_servo_angles(target_cell_pick_angles["close_gripper"])

                                if success:
                                    print("[robot] lifting object")
                                    success = send_servo_angles(target_cell_pick_angles["lift"])

                                if success:
                                    print("[robot] moving to approach drop")
                                    success = send_servo_angles(common_angles["approach_drop"])

                                if success:
                                    print("[robot] moving to drop height")
                                    success = send_servo_angles(common_angles["drop"])

                                if success:
                                    print("[robot] opening gripper")
                                    success = send_servo_angles(common_angles["open_gripper"])

                                print("[robot] returning to home")
                                if not send_servo_angles(common_angles["home"]):
                                    print("[robot] warning: failed to send final home command.")

                                if success:
                                    print("--- sequence complete successfully ---")
                                    play_sound_async("Sequence completed successfully")
                                else:
                                    print("--- sequence interrupted due to send error ---")
                                    play_sound_async("Sequence interrupted due to error")

        # Reset state if the object is no longer visible
        if object_marker_center is None and object_picked_up:
            print("object marker lost. ready for next pick.")
            object_picked_up = False

        # Display the frame
        cv2.imshow('YOLO Detection and Robot Control', frame)

        # Check for quit key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("quit key pressed. exiting...")
            break

except Exception as e:
    print(f"Unexpected error: {e}")

finally:
    print("[info] cleaning up...")

    # Send robot to home position
    if ser and ser.is_open:
        print("[robot] sending final home command before exit.")
        send_servo_angles(common_angles["home"])
        time.sleep(2)

    # Close resources
    close_robot_serial()
    cv2.destroyAllWindows()
    print("program terminated.")
