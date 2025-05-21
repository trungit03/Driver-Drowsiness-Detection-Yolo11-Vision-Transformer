import cv2
import torch
import pygame
import numpy as np
from ultralytics import YOLO
from collections import deque
import datetime
import time
import os
import argparse  # Added for proper argument handling


# Define constants
FPS = 30  # Frames per second
WARNING_DURATION = 2  # Duration to show warning (in seconds)
QUEUE_DURATION = 2  # Duration to store data in queue (in seconds)
YAWN_THRESHOLD_FRAMES = int(FPS * 1)  # Yawn detection threshold -> changed to around 1 second for demo
DROWSY_THRESHOLD_FRAMES = int(FPS * 0.8)  # Drowsy threshold -> define all sleep as 'drowsy'
HEAD_THRESHOLD_FRAMES = int(FPS * 0.8)  # Head movement detection threshold

def play_alarm(sound_file, duration):
    pygame.mixer.init()
    alarm_sound = pygame.mixer.Sound(sound_file)
    alarm_sound.play(loops=0, maxtime=duration) 

def trigger_alarm(trigger, sound_file, duration, play_audio=True):
    if trigger and play_audio:
        print("Alarm is triggered!")
        play_alarm(sound_file, duration)
    else:
        print("Alarm is not triggered.")

def get_source_fps(source):
    """Get FPS from video source or default to 30 for images/webcam"""
    if source == 0 or isinstance(source, str) and source.isdigit():  # Webcam
        cap = cv2.VideoCapture(int(source))
        if not cap.isOpened():
            print("Cannot access webcam.")
            return 30
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        return fps if fps > 0 else 30
    elif isinstance(source, str) and os.path.isfile(source):  # Video file
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"Cannot open video file: {source}")
            return 30
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        return fps if fps > 0 else 30
    else:  # Image or other
        return 30

def load_model(model_path):
    model = YOLO(model_path)
    return model

def detect(model, source, output_path=None, show_display=True, progress_callback=None,
           frame_callback=None, stop_flag=None, play_audio=True):
        
    # Determine source type
    is_image = False
    is_webcam=False
    if source == 0 or (isinstance(source, str) and source.isdigit()):
        source = int(source)
        is_webcam = True
    elif isinstance(source, str):
        if os.path.isfile(source):
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
            if os.path.splitext(source)[1].lower() in image_extensions:
                is_image = True
        else:
            print(f"File not found: {source}")
            return {"error": "File not found"}
    
    # Get FPS for the source
    fps = get_source_fps(source)
    
    # Initialize detection queues and variables
    queue_length = int(fps * QUEUE_DURATION)
    drowsy_threshold_frames = int(fps * 0.8)  
    yawn_threshold_frames = int(fps * 1)
    head_threshold_frames = int(fps * 0.8)
    
    # Special handling for images
    if is_image:
        # Set threshold to 1 for images
        drowsy_threshold_frames = 1
        yawn_threshold_frames = 1
        head_threshold_frames = 1
    
    eye_closed_queue = deque(maxlen=queue_length)
    yawn_queue = deque(maxlen=queue_length)
    head_queue = deque(maxlen=queue_length)
    head_warning_time = None
    yawn_warning_time = None
    drowsy_warning_time = None
    alarm_end_time = None
    
    # Track new events for callbacks
    new_drowsy = False
    new_yawn = False
    new_head = False

    frame_times = deque(maxlen=30)  # Store last 30 frame times
    
    # Statistics to return
    stats = {
        "drowsy_detections": 0,
        "yawn_detections": 0,
        "head_movement_detections": 0,
        "total_frames": 0,
        "avg_fps": 0,
        "output_path": output_path,
        "drowsy_timestamps": [],
        "yawn_timestamps": [],
        "head_timestamps": []
    }
    
    # Setup video writer if output path is provided
    video_writer = None
    
    # Get total frames for progress calculation
    total_frames = 1  # Default for images
    if not is_image:
        cap_check = cv2.VideoCapture(source)
        if cap_check.isOpened():
            total_frames = int(cap_check.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:  # Some video formats don't report frame count
                total_frames = 1000  # Use a default estimate
            cap_check.release()
    
    # Open video capture for video or webcam
    if not is_image:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"Cannot open source: {source}")
            return {"error": "Cannot open source"}
        
        # Create video writer if output path is provided
        if output_path:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps_out = cap.get(cv2.CAP_PROP_FPS)
            if fps_out <= 0:
                fps_out = 30
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_path, fourcc, fps_out, (width, height))
    
    prev_time = time.time()
    frame_count = 0
    current_fps = 0
    avg_fps = 0
    # Process frames
    while True:
        if stop_flag and stop_flag():
            break
        
        if is_image:
            # Process a single image
            frame = cv2.imread(source)
            if frame is None:
                print(f"Failed to read image: {source}")
                return {"error": "Failed to read image"}
            ret = True
        else:
            # Process video or webcam
            ret, frame = cap.read()
        
        if not ret:
            # End of video or error
            break
        
        frame_count += 1
        stats["total_frames"] += 1
        
        # Report progress
        if progress_callback and total_frames > 0:
            progress = min(100, int((frame_count / total_frames) * 100))
            progress_callback(progress)
        
        if not is_image:
            # Calculate FPS
            current_time = time.time()
            frame_time = current_time - prev_time
            frame_times.append(frame_time)
            current_fps = 1 / frame_time if frame_time > 0 else 0
            avg_fps = len(frame_times) / sum(frame_times) if sum(frame_times) > 0 else 0
            prev_time = current_time
        
        # Display FPS on frame
        if (show_display or output_path or frame_callback is not None) and not is_image:
            cv2.putText(frame, f"Current FPS: {int(current_fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Average FPS: {int(avg_fps)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Preprocess image
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        
        # Get model prediction results
        results = model.predict(source=[img], save=False)[0]
        
        # Reset event flags for this frame
        new_drowsy = False
        new_yawn = False
        new_head = False
        
        # Visualize results and extract object information
        detected_event_list = []  # Initialize list for detected events
        current_eye_closed = False
        current_yawn = False
        current_head_event = False
        
        for result in results:  # Iterate over detected objects
            boxes = result.boxes
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            classes = boxes.cls.cpu().numpy()
            
            for i in range(len(xyxy)):
                xmin, ymin, xmax, ymax = map(int, xyxy[i])
                confidence = confs[i]
                label = int(classes[i])
                
                if confidence > 0.5:  # Only show objects with confidence > 0.5
                    label_text = f"{model.names[label]} {confidence:.2f}"
                    
                    # Default color: green
                    color = (0, 255, 0)
                    
                    # Check eye-closed status (assume labels 0, 1, 2 are eye closed)
                    if label in [0, 1, 2]:
                        current_eye_closed = True
                    
                    # Check head up/down status (labels 4, 5 are head states)
                    if label in [4, 5]:
                        color = (0, 255, 255)  # Set yellow
                        current_head_event = True
                    
                    # Check yawn status (label 8)
                    if label == 8:
                        color = (0, 255, 255)  # Set yellow
                        current_yawn = True
                    
                    if show_display or output_path or frame_callback is not None:
                        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                        cv2.putText(frame, label_text, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Append current states to queues
        eye_closed_queue.append(current_eye_closed)
        yawn_queue.append(current_yawn)
        head_queue.append(current_head_event)
        
        # For images, fill queue with current state
        if is_image:
            # Ensure queue is filled with current state for single images
            for _ in range(queue_length - 1):
                eye_closed_queue.append(current_eye_closed)
                yawn_queue.append(current_yawn)
                head_queue.append(current_head_event)
        
        # Determine drowsiness state based on recent eye closure
        eye_closed_count = sum(eye_closed_queue)
        if eye_closed_count >= drowsy_threshold_frames:
            detected_event_list.append('drowsy')
            drowsy_warning_time = datetime.datetime.now()
            
            # Only increment counter if this is a new event (not already in progress)
            if 'drowsy' not in detected_event_list[:-1]:
                stats["drowsy_detections"] += 1
                new_drowsy = True
                stats["drowsy_timestamps"].append(frame_count / fps)
            
            if alarm_end_time is None or datetime.datetime.now() >= alarm_end_time:
                trigger_alarm(True, 'alarm.wav', 3000, play_audio=play_audio)  # 3 seconds alarm
                alarm_end_time = datetime.datetime.now() + datetime.timedelta(seconds=3)
        
        # Trigger warning if yawning occurs frequently
        yawn_count = sum(yawn_queue)
        if yawn_count >= yawn_threshold_frames:
            detected_event_list.append('yawn')
            yawn_warning_time = datetime.datetime.now()
            
            # Only increment counter for new yawn events
            if 'yawn' not in detected_event_list[:-1]:
                stats["yawn_detections"] += 1
                new_yawn = True
                stats["yawn_timestamps"].append(frame_count / fps)
            
            if alarm_end_time is None or datetime.datetime.now() >= alarm_end_time:
                trigger_alarm(True, 'alarm.wav', 1000, play_audio=play_audio)  # 1 second alarm
                alarm_end_time = datetime.datetime.now() + datetime.timedelta(seconds=1)
            yawn_queue.clear()
        
        # Trigger warning if head movement occurs frequently
        head_event_count = sum(head_queue)
        if head_event_count >= head_threshold_frames:
            detected_event_list.append('head_movement')
            head_warning_time = datetime.datetime.now()
            
            # Only increment counter for new head movement events
            if 'head_movement' not in detected_event_list[:-1]:
                stats["head_movement_detections"] += 1
                new_head = True
                stats["head_timestamps"].append(frame_count / fps)
            
            if alarm_end_time is None or datetime.datetime.now() >= alarm_end_time:
                trigger_alarm(True, 'alarm.wav', 1000, play_audio=play_audio)  # 1 second alarm
                alarm_end_time = datetime.datetime.now() + datetime.timedelta(seconds=1)
            head_queue.clear()
        
        # Reset alarm if no events detected
        if eye_closed_count < drowsy_threshold_frames and yawn_count < yawn_threshold_frames and head_event_count < head_threshold_frames:
            alarm_end_time = None
        
        # Current time
        current_time = datetime.datetime.now()
        
        # Change bounding box color based on drowsiness state
        for result in results:
            boxes = result.boxes
            xyxy = boxes.xyxy.cpu().numpy()
            classes = boxes.cls.cpu().numpy()
            
            for i in range(len(xyxy)):
                xmin, ymin, xmax, ymax = map(int, xyxy[i])
                label = int(classes[i])
                if label in [0, 1, 2]:  # Only change color for eye closed state
                    if 'drowsy' in detected_event_list:
                        color = (0, 0, 255)  # Red
                    else:
                        color = (0, 255, 0)  # Green
                    
                    if show_display or output_path or frame_callback is not None:
                        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                        cv2.putText(frame, model.names[label], (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Display warning messages
        font_scale = 0.75
        font_thickness = 2
        
        if show_display or output_path or frame_callback is not None:
            if drowsy_warning_time and (current_time - drowsy_warning_time).total_seconds() < WARNING_DURATION:
                cv2.putText(frame, 'Warning: Drowsy Detected!', (50, 150), cv2.FONT_ITALIC, font_scale, (0, 0, 255), font_thickness)
            if yawn_warning_time and (current_time - yawn_warning_time).total_seconds() < WARNING_DURATION:
                cv2.putText(frame, 'Warning: Yawning Detected!', (50, 50), cv2.FONT_ITALIC, font_scale, (0, 255, 255), font_thickness)
            if head_warning_time and (current_time - head_warning_time).total_seconds() < WARNING_DURATION:
                cv2.putText(frame, 'Warning: Head Up/Down Detected!', (50, 100), cv2.FONT_ITALIC, font_scale, (0, 255, 255), font_thickness)
        
        # Write frame to output video if requested
        if output_path and video_writer is not None:
            video_writer.write(frame)
        
        # For image, save the processed image if output path is provided
        if is_image and output_path:
            cv2.imwrite(output_path, frame)
            
        if frame_callback:
            # Pass frame and event flags to callback
            event_stats = {
                'new_drowsy': new_drowsy,
                'new_yawn': new_yawn,
                'new_head': new_head
            }
            frame_callback(frame.copy(), event_stats)
            
        # Display the frame
        if is_image:
            if show_display:
                cv2.imshow('YOLOv8 Object Detection', frame)
                print("Press any key to close image display...")
                cv2.waitKey(0)  # Wait indefinitely until key is pressed
                cv2.destroyAllWindows()
            break
        else:
            # For video/webcam
            if show_display:
                cv2.imshow('YOLOv8 Object Detection', frame)
                if cv2.waitKey(1) == ord('q'):  # Press 'q' to quit
                    break
                

        # For image, we only process one frame
        
    
    # Clean up
    if not is_image:
        cap.release()
    if video_writer is not None:
        video_writer.release()
    if show_display:
        cv2.destroyAllWindows()
    
    if not is_image:
    # Update statistics
        stats["avg_fps"] = avg_fps
    
    return stats

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Drowsiness Detection using YOLOv8')
    parser.add_argument('--source', type=str, default="0", help='Source for detection: 0 for webcam, path for video/image')
    parser.add_argument('--output', type=str, default=None, help='Output file path')
    parser.add_argument('--no-display', action='store_true', help='Do not display output')
    parser.add_argument('--model', type=str, default='models/best.pt', help='Path to YOLOv8 model')
    
    args = parser.parse_args()
    
    model = load_model(args.model)
    
    # Run detection
    detect(model, args.source, output_path=args.output, show_display=not args.no_display)