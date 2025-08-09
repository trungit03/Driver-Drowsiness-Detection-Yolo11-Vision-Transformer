# optimize.py - FIXED VERSION với logic detection được sửa
import cv2
import torch
import pygame
import numpy as np
from ultralytics import YOLO
from collections import deque
import datetime
import time
import os
import argparse
import threading

# Define constants
FPS = 30
WARNING_DURATION = 2
QUEUE_DURATION = 2
YAWN_THRESHOLD_FRAMES = int(FPS * 1)      # 30 frames = 1 second
DROWSY_THRESHOLD_FRAMES = int(FPS * 0.8)  # 24 frames = 0.8 second  
HEAD_THRESHOLD_FRAMES = int(FPS * 0.8)    # 24 frames = 0.8 second

# Global lock for alarm management
alarm_lock = threading.Lock()
current_alarm_thread = None
alarm_active = False

def play_alarm(sound_file, duration):
    global alarm_active
    try:
        pygame.mixer.init()
        alarm_sound = pygame.mixer.Sound(sound_file)
        
        with alarm_lock:
            alarm_active = True
        
        alarm_sound.play(loops=0, maxtime=duration)
        
        # Wait for the alarm to finish
        time.sleep(duration / 1000.0)  # Convert ms to seconds
        
        with alarm_lock:
            alarm_active = False
            
    except Exception as e:
        print(f"Error playing alarm: {str(e)}")
        with alarm_lock:
            alarm_active = False

def trigger_alarm(trigger, sound_file, duration, play_audio=True):
    global current_alarm_thread, alarm_active
    
    if trigger and play_audio:
        with alarm_lock:
            # Only start new alarm if no alarm is currently playing
            if not alarm_active:
                print(f"Alarm triggered for {duration}ms!")
                current_alarm_thread = threading.Thread(
                    target=play_alarm, 
                    args=(sound_file, duration), 
                    daemon=True
                )
                current_alarm_thread.start()
                return True
            else:
                print("Alarm already playing, skipping...")
                return False
    return False

def get_source_fps(source):
    """Get FPS from video source or default to 30 for images/webcam"""
    if source == 0 or (isinstance(source, str) and source.isdigit()):  # Webcam
        cap = cv2.VideoCapture(int(source))
        if not cap.isOpened():
            return 30
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        return fps if fps > 0 else 30
    elif isinstance(source, str) and os.path.isfile(source):  # Video file
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            return 30
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        return fps if fps > 0 else 30
    else:  # Image or other
        return 30

def load_model(model_path):
    model = YOLO(model_path)
    
    # Optimize for GPU if available
    if torch.cuda.is_available():
        model = model.cuda()
        torch.backends.cudnn.benchmark = True  # Enable cuDNN benchmark
        print("Using GPU acceleration")
    else:
        print("Using CPU")
    
    # Warm up the model
    dummy_input = torch.randn(1, 3, 640, 640).to(model.device)
    if torch.cuda.is_available():
        dummy_input = dummy_input.half()  # Use half precision for GPU
        
    for _ in range(3):  # Warm up 3 times
        model.predict(dummy_input, verbose=False)
    
    return model

def detect(model, source, output_path=None, show_display=True, progress_callback=None,
           frame_callback=None, stop_flag=None, play_audio=True):
    
    # Determine source type
    is_image = False
    is_webcam = False
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
    drowsy_threshold_frames = int(fps * 0.8)  # 24 frames
    yawn_threshold_frames = int(fps * 1)      # 30 frames
    head_threshold_frames = int(fps * 0.8)    # 24 frames
    
    print(f"FPS: {fps}, Thresholds - Drowsy: {drowsy_threshold_frames}, Yawn: {yawn_threshold_frames}, Head: {head_threshold_frames}")
    
    # Special handling for images
    if is_image:
        drowsy_threshold_frames = 1
        yawn_threshold_frames = 1
        head_threshold_frames = 1
    
    eye_closed_queue = deque(maxlen=queue_length)
    yawn_queue = deque(maxlen=queue_length)
    head_queue = deque(maxlen=queue_length)
    
    # Alarm timing 
    drowsy_alarm_counter = 0
    yawn_alarm_counter = 0  
    head_alarm_counter = 0
    
    # Alarm cycle
    DROWSY_ALARM_CYCLE = drowsy_threshold_frames  # Play back every 24 frames when drowsy
    YAWN_ALARM_CYCLE = yawn_threshold_frames      # Play back every 30 frames when yawn
    HEAD_ALARM_CYCLE = head_threshold_frames      # Play back every 24 frames when head movement
    
    # Warning time tracking
    drowsy_warning_time = None
    yawn_warning_time = None
    head_warning_time = None
    
    # Track new events for callbacks
    new_drowsy = False
    new_yawn = False
    new_head = False
    
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
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"Cannot open source: {source}")
            return {"error": "Cannot open source"}
        
        # Get video properties for writer
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_out = cap.get(cv2.CAP_PROP_FPS)
        if fps_out <= 0:
            fps_out = 30
        
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_path, fourcc, fps_out, (width, height))
        
        # Get total frames
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:  # Some video formats don't report frame count
            total_frames = 1000  # Use a default estimate
    else:
        # For images
        frame = cv2.imread(source)
        if frame is None:
            print(f"Failed to read image: {source}")
            return {"error": "Failed to read image"}
    
    # FPS calculation variables
    frame_times = deque(maxlen=30)
    frame_count = 0
    detection_time_total = 0
    avg_fps = 0
    
    # Use optimized settings
    img_size = 480  # Balanced size for performance and accuracy
    use_half = torch.cuda.is_available()
    
    # Video timing control
    start_time = time.time()
    
    # Main processing loop
    try:
        while True:
            if stop_flag and stop_flag():
                break
            
            # Start timing for FPS calculation
            loop_start_time = time.time()
            
            # Read frame
            if is_image:
                ret = True
                # Only process once
                if frame_count > 0:
                    break
            else:
                ret, frame = cap.read()
                if not ret:
                    break
            
            frame_count += 1
            stats["total_frames"] += 1
            
            # Calculate current time in video (for proper timing)
            current_video_time = frame_count / fps
            current_real_time = time.time()
            
            # Report progress
            if progress_callback and total_frames > 0:
                progress = min(100, int((frame_count / total_frames) * 100))
                progress_callback(progress)
            
            # Preprocess image
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            
            # Get model prediction with optimized settings
            det_start_time = time.time()
            
            results = model.predict(
                source=[img],  
                save=False, 
                verbose=False, 
                imgsz=img_size, 
                half=use_half,
                conf=0.5,
                stream=False
            )
            
           
            current_eye_closed = False
            current_yawn = False
            current_head_event = False
            boxes_to_draw = []
            
            for result in results:
                boxes = result.boxes
                if boxes is None:
                    continue
                
                if boxes.xyxy.numel() > 0:
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
                            
                            # Store box info for later drawing
                            boxes_to_draw.append((xmin, ymin, xmax, ymax, label, confidence, color))
            
            det_end_time = time.time()
            detection_time_total += (det_end_time - det_start_time)
            
            # Update queues
            eye_closed_queue.append(current_eye_closed)
            yawn_queue.append(current_yawn)
            head_queue.append(current_head_event)
            
            # For single images, fill the queue
            if is_image and frame_count == 1:
                for _ in range(queue_length - 1):
                    eye_closed_queue.append(current_eye_closed)
                    yawn_queue.append(current_yawn)
                    head_queue.append(current_head_event)
            
            # Reset event flags for this frame
            new_drowsy = False
            new_yawn = False
            new_head = False
            
            # Calculate event counts
            eye_closed_count = sum(eye_closed_queue)
            yawn_count = sum(yawn_queue)
            head_event_count = sum(head_queue)
            
            current_time = datetime.datetime.now()
            
            # Handle drowsiness detection (eye closed events)
            if eye_closed_count >= drowsy_threshold_frames:
                drowsy_alarm_counter += 1
                drowsy_warning_time = current_time
                
                # Generate alarm every DROWSY_ALARM_CYCLE frames cycle
                if drowsy_alarm_counter >= DROWSY_ALARM_CYCLE:
                    stats["drowsy_detections"] += 1
                    new_drowsy = True
                    stats["drowsy_timestamps"].append(current_video_time)
                    
                    trigger_alarm(True, 'alarm.wav', 3000, play_audio=play_audio)
                    drowsy_alarm_counter = 0  # Reset counter
                    print(f"DROWSY ALARM at {current_video_time:.2f}s (cycle: {DROWSY_ALARM_CYCLE} frames)")
            else:
                # Reset counter 
                drowsy_alarm_counter = 0
            
            # Handle yawning detection
            if yawn_count >= yawn_threshold_frames:
                yawn_alarm_counter += 1
                yawn_warning_time = current_time
                
                # Generate alarm every YAWN_ALARM_CYCLE frames cycle
                if yawn_alarm_counter >= YAWN_ALARM_CYCLE:
                    stats["yawn_detections"] += 1
                    new_yawn = True
                    stats["yawn_timestamps"].append(current_video_time)
                    
                    trigger_alarm(True, 'alarm.wav', 1000, play_audio=play_audio)
                    yawn_alarm_counter = 0  # Reset counter
                    print(f"YAWN ALARM at {current_video_time:.2f}s (cycle: {YAWN_ALARM_CYCLE} frames)")
            else:
                # Reset counter 
                yawn_alarm_counter = 0
            
            # Handle head movement detection  
            if head_event_count >= head_threshold_frames:
                head_alarm_counter += 1
                head_warning_time = current_time
                
                # Generate alarm every HEAD_ALARM_CYCLE frames cycle
                if head_alarm_counter >= HEAD_ALARM_CYCLE:
                    stats["head_movement_detections"] += 1
                    new_head = True
                    stats["head_timestamps"].append(current_video_time)
                    
                    trigger_alarm(True, 'alarm.wav', 1000, play_audio=play_audio)
                    head_alarm_counter = 0  # Reset counter
                    print(f"HEAD MOVEMENT ALARM at {current_video_time:.2f}s (cycle: {HEAD_ALARM_CYCLE} frames)")
            else:
                # Reset counter 
                head_alarm_counter = 0
            
            # Determine detected events for display purposes
            detected_event_list = []
            if eye_closed_count >= drowsy_threshold_frames:
                detected_event_list.append('drowsy')
            if yawn_count >= yawn_threshold_frames:
                detected_event_list.append('yawn')
            if head_event_count >= head_threshold_frames:
                detected_event_list.append('head_movement')
            
            # Change bounding box color based on drowsiness state
            for result in results:
                boxes = result.boxes
                if boxes is None:
                    continue
                    
                if boxes.xyxy.numel() > 0:
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
                                cv2.putText(frame, model.names[label], (xmin, ymin - 10), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw other boxes (non-eye-closed)
            for (xmin, ymin, xmax, ymax, label, confidence, color) in boxes_to_draw:
                if label not in [0, 1, 2]:  # Skip eye-closed boxes (already drawn above)
                    if show_display or output_path or frame_callback:
                        label_text = f"{model.names.get(label, 'unknown')} {confidence:.2f}"
                        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                        cv2.putText(frame, label_text, (xmin, ymin - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Add warning messages và counters
            if show_display or output_path or frame_callback:
                font = cv2.FONT_ITALIC
                font_scale = 0.75
                font_thickness = 2
                
                # Show warnings with proper timing
                if drowsy_warning_time and (current_time - drowsy_warning_time).total_seconds() < WARNING_DURATION:
                    cv2.putText(frame, f'Warning: Drowsy Detected! ', 
                               (50, 150), font, font_scale, (0, 0, 255), font_thickness)
                if yawn_warning_time and (current_time - yawn_warning_time).total_seconds() < WARNING_DURATION:
                    cv2.putText(frame, f'Warning: Yawning Detected! ', 
                               (50, 50), font, font_scale, (0, 255, 255), font_thickness)
                if head_warning_time and (current_time - head_warning_time).total_seconds() < WARNING_DURATION:
                    cv2.putText(frame, f'Warning: Head Up/Down Detected! ', 
                               (50, 100), font, font_scale, (0, 255, 255), font_thickness)
                
            
            # Calculate FPS
            loop_end_time = time.time()
            frame_time = loop_end_time - loop_start_time
            frame_times.append(frame_time)
            
            
            # Calculate current and average FPS
            current_fps = 1.0 / frame_time if frame_time > 0 else 0
            avg_fps = sum([1.0 / t for t in frame_times]) / len(frame_times) if frame_times else 0
            
            # Add FPS and timing info to frame
            if show_display or output_path or frame_callback:
                cv2.putText(frame, f"FPS: {int(current_fps)}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"AVG FPS: {int(avg_fps)}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Save or display frame
            if output_path:
                if is_image:
                    cv2.imwrite(output_path, frame)
                else:
                    video_writer.write(frame)
            
            if frame_callback:
                frame_callback(frame.copy(), {
                    'new_drowsy': new_drowsy,
                    'new_yawn': new_yawn,
                    'new_head': new_head
                })
            
            # Display if needed
            if show_display:
                cv2.imshow('YOLOv8 Object Detection', frame)
                key = cv2.waitKey(1)
                if key == 27 or key == ord('q'):  # ESC or 'q' to quit
                    break
            
            # For single image, only process once
            if is_image:
                if show_display:
                    print("Press any key to close image display...")
                    cv2.waitKey(0)  # Wait indefinitely until key is pressed
                break
            
            # ADDED: Frame rate control for video files to prevent too fast processing
            if not is_webcam and not is_image:
                expected_time = frame_count / fps
                actual_time = time.time() - start_time
                if actual_time < expected_time:
                    time.sleep(expected_time - actual_time)
    
    except Exception as e:
        print(f"Detection error: {str(e)}")
        return {"error": str(e)}
    
    finally:
        # Wait for any remaining alarms to finish
        global current_alarm_thread
        if current_alarm_thread and current_alarm_thread.is_alive():
            current_alarm_thread.join(timeout=5)  # Wait up to 5 seconds
        
        # Clean up resources
        if not is_image and 'cap' in locals() and cap.isOpened():
            cap.release()
        if video_writer is not None:
            video_writer.release()
        if show_display:
            cv2.destroyAllWindows()
        
        # Stop pygame mixer
        try:
            pygame.mixer.quit()
        except:
            pass
    
    # Calculate average detection time
    if stats["total_frames"] > 0:
        avg_detection_time = detection_time_total / stats["total_frames"]
        print(f"Average detection time per frame: {avg_detection_time:.4f}s")
    
    # Update stats with average FPS
    stats["avg_fps"] = avg_fps if avg_fps > 0 else current_fps
    
    print(f"Final stats: Drowsy: {stats['drowsy_detections']}, Yawn: {stats['yawn_detections']}, Head: {stats['head_movement_detections']}")
    
    return stats

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Drowsiness Detection using YOLOv8')
    parser.add_argument('--source', type=str, default="0", help='Source: 0 for webcam, path for video/image')
    parser.add_argument('--output', type=str, default=None, help='Output file path')
    parser.add_argument('--no-display', action='store_true', help='Disable display')
    parser.add_argument('--model', type=str, default='models/best.pt', help='Model path')
    
    args = parser.parse_args()
    
    model = load_model(args.model)
    detect(model, args.source, output_path=args.output, show_display=not args.no_display)