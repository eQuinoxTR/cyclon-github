import os
import sys
import argparse
import glob
import time
import serial

import cv2
import numpy as np
from ultralytics import YOLO
from threading import Thread # --- MODIFICATION --- Added Thread import

# --- MODIFICATION START ---
# A dedicated class to handle video streams in a separate thread to prevent buffer lag.
class VideoStream:
    """A class to read frames from a camera in a dedicated thread."""
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        # Start the thread to read frames from the video stream
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        # Keep looping infinitely until the thread is stopped
        while True:
            if self.stopped:
                self.stream.release()
                return
            # Otherwise, read the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        # Return the frame most recently read
        return self.frame

    def stop(self):
        # Indicate that the thread should be stopped
        self.stopped = True
# --- MODIFICATION END ---


try:
    # Abre el puerto serial
    SERIAL_PORT = '/dev/ttyUSB0'
    BAUD_RATE = 9600 # Debe coincidir con la configuración de Arduino
    esp32 = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    time.sleep(2) 
    print("se estan mandando datos desde python!")

except serial.SerialException as e:
    print(f"Error al abrir o comunicarse con el puerto serial: {e}")

# Define and parse user input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', help='Path to YOLO model file (example: "runs/detect/train/weights/best.pt")',
                    required=True)
parser.add_argument('--source', help='Image source, can be image file ("test.jpg"), \
                    image folder ("test_dir"), video file ("testvid.mp4"), or index of USB camera ("usb0")', 
                    required=True)
parser.add_argument('--thresh', help='Minimum confidence threshold for displaying detected objects (example: "0.4")',
                    default=0.5)
parser.add_argument('--resolution', help='Resolution in WxH to display inference results at (example: "640x480"), \
                    otherwise, match source resolution',
                    default=None)
parser.add_argument('--record', help='Record results from video or webcam and save it as "demo1.avi". Must specify --resolution argument to record.',
                    action='store_true')

args = parser.parse_args()


# Parse user inputs
model_path = args.model
img_source = args.source
min_thresh = float(args.thresh) # --- MODIFICATION --- Converted thresh to float
user_res = args.resolution
record = args.record

# Check if model file exists and is valid
if (not os.path.exists(model_path)):
    print('ERROR: Model path is invalid or model was not found. Make sure the model filename was entered correctly.')
    sys.exit(0)

# Load the model into memory and get labemap
model = YOLO(model_path, task='detect')
labels = model.names

# Parse input to determine if image source is a file, folder, video, or USB camera
img_ext_list = ['.jpg','.JPG','.jpeg','.JPEG','.png','.PNG','.bmp','.BMP']
vid_ext_list = ['.avi','.mov','.mp4','.mkv','.wmv']

if os.path.isdir(img_source):
    source_type = 'folder'
elif os.path.isfile(img_source):
    _, ext = os.path.splitext(img_source)
    if ext in img_ext_list:
        source_type = 'image'
    elif ext in vid_ext_list:
        source_type = 'video'
    else:
        print(f'File extension {ext} is not supported.')
        sys.exit(0)
elif 'usb' in img_source:
    source_type = 'usb'
    usb_idx = int(img_source[3:])
elif 'picamera' in img_source:
    source_type = 'picamera'
elif img_source.startswith(('http://', 'https://', 'rtsp://')):
    source_type = 'ip_camera'
else:
    print(f'Input {img_source} is invalid. Please try again.')
    sys.exit(0)

# Parse user-specified display resolution
resize = False
if user_res:
    resize = True
    resW, resH = int(user_res.split('x')[0]), int(user_res.split('x')[1])

# Check if recording is valid and set up recording
if record:
    if source_type not in ['video','usb', 'ip_camera']:
        print('Recording only works for video and camera sources. Please try again.')
        sys.exit(0)
    if not user_res:
        print('Please specify resolution to record video at.')
        sys.exit(0)
    
    # Set up recording
    record_name = 'demo1.avi'
    record_fps = 30
    recorder = cv2.VideoWriter(record_name, cv2.VideoWriter_fourcc(*'MJPG'), record_fps, (resW,resH))

# Load or initialize image source
if source_type == 'image':
    imgs_list = [img_source]
elif source_type == 'folder':
    imgs_list = []
    filelist = glob.glob(img_source + '/*')
    for file in filelist:
        _, file_ext = os.path.splitext(file)
        if file_ext in img_ext_list:
            imgs_list.append(file)
            
# --- MODIFICATION START ---
# Use the new threaded VideoStream for live video sources
elif source_type in ['video', 'usb', 'ip_camera']:
    if source_type == 'video': cap_arg = img_source
    elif source_type == 'usb': cap_arg = usb_idx
    else: cap_arg = img_source # For 'ip_camera'
    
    print("[INFO] Starting threaded video stream...")
    cap = VideoStream(src=cap_arg).start()
    # Note: Setting resolution on a threaded stream is more complex and often handled by the stream URL or camera settings.
    # The resizing done in the main loop will enforce the display resolution.
# --- MODIFICATION END ---

elif source_type == 'picamera':
    from picamera2 import Picamera2
    cap = Picamera2()
    config = cap.create_video_configuration(main={"format": 'XRGB8888', "size": (resW, resH)})
    cap.configure(config)
    cap.start()

# Set bounding box colors
bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106), 
              (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184)]

# Initialize control and status variables
avg_frame_rate = 0
frame_rate_buffer = []
fps_avg_len = 200
img_count = 0

# Allow camera to warm up
time.sleep(1.0)

# Begin inference loop
while True:

    t_start = time.perf_counter()

    # --- MODIFICATION START ---
    # Load frame from the appropriate source
    if source_type in ['image', 'folder']:
        if img_count >= len(imgs_list):
            break
        frame = cv2.imread(imgs_list[img_count])
        img_count += 1
    elif source_type == 'picamera':
        frame = cap.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    else: # For video, usb, ip_camera
        frame = cap.read()
    
    # Check if the frame is valid
    if frame is None:
        print("Frame could not be read. Stream may have ended.")
        break
    # --- MODIFICATION END ---

    # Resize frame to desired display resolution
    if resize:
        frame = cv2.resize(frame,(resW,resH))

    # Run inference on frame
    results = model(frame, verbose=False)

    # Extract results
    detections = results[0].boxes

    # Initialize variable for basic object counting example
    object_count = 0

    # Go through each detection and get bbox coords, confidence, and class
    for i in range(len(detections)):
        xyxy_tensor = detections[i].xyxy.cpu()
        xyxy = xyxy_tensor.numpy().squeeze()
        xmin, ymin, xmax, ymax = xyxy.astype(int)

        classidx = int(detections[i].cls.item())
        classname = labels[classidx]
        conf = detections[i].conf.item()

        if conf > min_thresh:
            try:
                transformar_string_a_bytes = (classname + '\n').encode('utf-8')
                esp32.write(transformar_string_a_bytes)
                print(f"string enviado desde python a esp32: {classname}")
            except NameError: # Handle case where esp32 is not defined
                pass 
            except serial.SerialException as e:
                print(f"Serial communication error: {e}")


            color = bbox_colors[classidx % 10]
            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), color, 2)
            label = f'{classname}: {int(conf*100)}%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_ymin = max(ymin, labelSize[1] + 10)
            cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), color, cv2.FILLED)
            cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            object_count += 1

    # Calculate and draw framerate
    if source_type not in ['image', 'folder']:
        cv2.putText(frame, f'FPS: {avg_frame_rate:0.2f}', (10,20), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2)
    
    # Display detection results
    cv2.putText(frame, f'Number of objects: {object_count}', (10,40), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2)
    cv2.imshow('YOLO detection results',frame)
    if record: recorder.write(frame)

    # Handle key presses
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        cv2.waitKey(0)
    elif key == ord('p'):
        cv2.imwrite('capture.png',frame)
    
    # Calculate FPS
    t_stop = time.perf_counter()
    frame_rate_calc = 1 / (t_stop - t_start)
    frame_rate_buffer.append(frame_rate_calc)
    if len(frame_rate_buffer) > fps_avg_len:
        frame_rate_buffer.pop(0)
    avg_frame_rate = np.mean(frame_rate_buffer)

# Clean up
print(f'Average pipeline FPS: {avg_frame_rate:.2f}')
if source_type in ['video', 'usb', 'ip_camera']:
    cap.stop() # --- MODIFICATION --- Stop the threaded stream
elif source_type == 'picamera':
    cap.stop()
if record: recorder.release()
cv2.destroyAllWindows()