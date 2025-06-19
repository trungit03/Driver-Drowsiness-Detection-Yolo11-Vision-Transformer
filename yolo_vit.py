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
from PIL import Image
import torchvision.transforms as transforms

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

# ViT label mapping
VIT_LABELS = {0: 'Drowsy', 1: 'Nondrowsy', 2: 'Yawning'}
YOLO_LABELS = {
  0: 'eyes_closed', 1: 'eyes_closed_head_right', 2: 'eyes_closed_head_left', 
  3: 'focused', 4: 'head_down', 5: 'head_up', 
  6: 'seeing_right', 7: 'seeing_left', 8: 'yawning'
}

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

def load_model(model_path, vit_model_path=None):
  """Load YOLO model and optionally ViT model"""
  # Load YOLO model
  yolo_model = YOLO(model_path)
  
  # Optimize for GPU if available
  if torch.cuda.is_available():
      yolo_model = yolo_model.cuda()
      torch.backends.cudnn.benchmark = True  # Enable cuDNN benchmark
      print("Using GPU acceleration")
  else:
      print("Using CPU")
  
  # Load ViT model if path is provided
  vit_model = None
  vit_transform = None
  
  if vit_model_path and os.path.exists(vit_model_path):
      try:
          print(f"Loading ViT model from: {vit_model_path}")
          vit_model = torch.load(vit_model_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
          vit_model.eval()
          
          # Define ViT preprocessing transform
          vit_transform = transforms.Compose([
              transforms.Resize((224, 224)),
              transforms.ToTensor(),
              transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
          ])
          
          print("ViT model loaded successfully")
      except Exception as e:
          print(f"Error loading ViT model: {str(e)}")
          vit_model = None
          vit_transform = None
  
  # Warm up the YOLO model
  dummy_input = torch.randn(1, 3, 640, 640).to(yolo_model.device)
  if torch.cuda.is_available():
      dummy_input = dummy_input.half()  # Use half precision for GPU
      
  for _ in range(3):  # Warm up 3 times
      yolo_model.predict(dummy_input, verbose=False)
  
  return yolo_model, vit_model, vit_transform

def predict_vit(vit_model, vit_transform, image):
  """Predict using ViT model"""
  if vit_model is None or vit_transform is None:
      return None, 0.0
  
  try:
      # Convert BGR to RGB
      if len(image.shape) == 3 and image.shape[2] == 3:
          image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      else:
          image_rgb = image
      
      # Convert to PIL Image
      pil_image = Image.fromarray(image_rgb)
      
      # Apply transforms
      input_tensor = vit_transform(pil_image).unsqueeze(0)
      
      # Move to device
      device = next(vit_model.parameters()).device
      input_tensor = input_tensor.to(device)
      
      # Predict
      with torch.no_grad():
          outputs = vit_model(input_tensor)
          probabilities = torch.softmax(outputs, dim=1)
          predicted_class = torch.argmax(probabilities, dim=1).item()
          confidence = probabilities[0][predicted_class].item()
      
      return predicted_class, confidence
  
  except Exception as e:
      print(f"Error in ViT prediction: {str(e)}")
      return None, 0.0

def combine_predictions(yolo_detections, vit_prediction, vit_confidence):
  """Combine YOLO and ViT predictions for better accuracy"""
  # Initialize combined results
  combined_result = {
      'eye_closed': False,
      'yawn': False,
      'head_movement': False,
      'confidence_scores': {}
  }
  
  # Process YOLO detections
  yolo_eye_closed = False
  yolo_yawn = False
  yolo_head_movement = False
  
  for detection in yolo_detections:
      label = detection.get('label', '')
      confidence = detection.get('confidence', 0.0)
      
      if label in ['eyes_closed', 'eyes_closed_head_right', 'eyes_closed_head_left']:
          yolo_eye_closed = True
          combined_result['confidence_scores']['yolo_drowsy'] = max(
              combined_result['confidence_scores'].get('yolo_drowsy', 0), confidence
          )
      elif label == 'yawning':
          yolo_yawn = True
          combined_result['confidence_scores']['yolo_yawn'] = max(
              combined_result['confidence_scores'].get('yolo_yawn', 0), confidence
          )
      elif label in ['head_down', 'head_up']:  # Only head_down and head_up trigger warnings
          yolo_head_movement = True
          combined_result['confidence_scores']['yolo_head'] = max(
              combined_result['confidence_scores'].get('yolo_head', 0), confidence
          )
  
  # Process ViT prediction
  vit_drowsy = False
  vit_yawn = False
  
  if vit_prediction is not None and vit_confidence > 0.5:  # Confidence threshold
      if vit_prediction == 0:  # Drowsy
          vit_drowsy = True
          combined_result['confidence_scores']['vit_drowsy'] = vit_confidence
      elif vit_prediction == 2:  # Yawning
          vit_yawn = True
          combined_result['confidence_scores']['vit_yawn'] = vit_confidence
  
  # Combine predictions with weighted voting
  # ViT gets higher weight for overall state, YOLO for specific features
  
  # Eye closed detection: Combine YOLO and ViT drowsy predictions
  if yolo_eye_closed and vit_drowsy:
      combined_result['eye_closed'] = True  # Both agree - high confidence
  elif yolo_eye_closed or vit_drowsy:
      # One model detects - use confidence threshold
      yolo_conf = combined_result['confidence_scores'].get('yolo_drowsy', 0)
      vit_conf = combined_result['confidence_scores'].get('vit_drowsy', 0)
      if max(yolo_conf, vit_conf) > 0.6:
          combined_result['eye_closed'] = True
  
  # Yawn detection: Combine YOLO and ViT yawn predictions
  if yolo_yawn and vit_yawn:
      combined_result['yawn'] = True  # Both agree - high confidence
  elif yolo_yawn or vit_yawn:
      # One model detects - use confidence threshold
      yolo_conf = combined_result['confidence_scores'].get('yolo_yawn', 0)
      vit_conf = combined_result['confidence_scores'].get('vit_yawn', 0)
      if max(yolo_conf, vit_conf) > 0.6:
          combined_result['yawn'] = True
  
  # Head movement: Primarily from YOLO (ViT doesn't detect this specifically)
  combined_result['head_movement'] = yolo_head_movement
  
  return combined_result

def get_detection_color(label, drowsy_detected, yawn_detected, head_detected):
  """Get color for bounding box and text based on detection state"""
  # Default color is green
  default_color = (0, 255, 0)
  
  # Check if this detection is related to current alerts
  if label in ['eyes_closed', 'eyes_closed_head_right', 'eyes_closed_head_left'] and drowsy_detected:
      return (0, 0, 255)  # Red for drowsy
  elif label == 'yawning' and yawn_detected:
      return (0, 165, 255)  # Orange for yawn
  elif label in ['head_down', 'head_up'] and head_detected:
      return (0, 255, 255)  # Yellow for head movement
  else:
      return default_color  # Green for normal/no alert

def detect(model, source, output_path=None, show_display=True, progress_callback=None,
         frame_callback=None, stop_flag=None, play_audio=True, vit_model_path=None):
  
  # Load models (YOLO + ViT)
  if isinstance(model, tuple):
      yolo_model, vit_model, vit_transform = model
  else:
      # Backward compatibility - if only YOLO model is passed
      yolo_model, vit_model, vit_transform = load_model(model, vit_model_path)
  
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
      "head_timestamps": [],
      "vit_predictions": []  # New: Store ViT predictions for analysis
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
  img_size = 640  # BACK TO ORIGINAL SIZE
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
          
          # ORIGINAL YOLO PREDICTION - NO PREPROCESSING
          det_start_time = time.time()
          
          # Use original YOLO prediction method
          yolo_results = yolo_model.predict(
              source=frame,  # Pass frame directly like original
              save=False, 
              verbose=False, 
              imgsz=img_size, 
              half=use_half,
              conf=0.3
          )
          
          # Get ViT prediction
          vit_prediction, vit_confidence = predict_vit(vit_model, vit_transform, frame)
          
          # Store ViT prediction for analysis
          if vit_prediction is not None:
              stats["vit_predictions"].append({
                  "frame": frame_count,
                  "prediction": VIT_LABELS.get(vit_prediction, "Unknown"),
                  "confidence": vit_confidence,
                  "timestamp": current_video_time
              })
          
          # Process YOLO detections - ORIGINAL LOGIC
          yolo_detections = []
          boxes_to_draw = []
          
          for result in yolo_results:
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
                      label_idx = int(classes[i])
                      
                      # Use original confidence threshold
                      if confidence > 0.3:  # ORIGINAL THRESHOLD
                          label_text = f"{yolo_model.names[label_idx]} {confidence:.2f}"
                          
                          # Store detection for combination
                          yolo_detections.append({
                              'label': yolo_model.names[label_idx],
                              'confidence': confidence,
                              'bbox': (xmin, ymin, xmax, ymax)
                          })
                          
                          boxes_to_draw.append({
                              'bbox': (xmin, ymin, xmax, ymax),
                              'label': label_text,
                              'label_name': yolo_model.names[label_idx],
                              'confidence': confidence
                          })
          
          # Combine YOLO and ViT predictions
          combined_result = combine_predictions(yolo_detections, vit_prediction, vit_confidence)
          
          # Extract combined results
          current_eye_closed = combined_result['eye_closed']
          current_yawn = combined_result['yawn']
          current_head_event = combined_result['head_movement']
          
          # Add to queues
          eye_closed_queue.append(current_eye_closed)
          yawn_queue.append(current_yawn)
          head_queue.append(current_head_event)
          
          # Count consecutive detections
          eye_closed_count = sum(eye_closed_queue)
          yawn_count = sum(yawn_queue)
          head_count = sum(head_queue)
          
          # Check thresholds and trigger alarms
          drowsy_detected = eye_closed_count >= drowsy_threshold_frames
          yawn_detected = yawn_count >= yawn_threshold_frames
          head_detected = head_count >= head_threshold_frames
          
          # Handle drowsy detection
          if drowsy_detected:
              if drowsy_warning_time is None:
                  drowsy_warning_time = current_real_time
                  stats["drowsy_detections"] += 1
                  stats["drowsy_timestamps"].append(current_video_time)
                  new_drowsy = True
                  
                  # Trigger alarm
                  trigger_alarm(True, "static/alarm.wav", 3000, play_audio)
              
              drowsy_alarm_counter += 1
              if drowsy_alarm_counter >= DROWSY_ALARM_CYCLE:
                  trigger_alarm(True, "static/alarm.wav", 3000, play_audio)
                  drowsy_alarm_counter = 0
          else:
              if drowsy_warning_time and (current_real_time - drowsy_warning_time) > WARNING_DURATION:
                  drowsy_warning_time = None
                  new_drowsy = False
              drowsy_alarm_counter = 0
          
          # Handle yawn detection
          if yawn_detected:
              if yawn_warning_time is None:
                  yawn_warning_time = current_real_time
                  stats["yawn_detections"] += 1
                  stats["yawn_timestamps"].append(current_video_time)
                  new_yawn = True
                  
                  # Trigger alarm
                  trigger_alarm(True, "static/alarm.wav", 1000, play_audio)
              
              yawn_alarm_counter += 1
              if yawn_alarm_counter >= YAWN_ALARM_CYCLE:
                  trigger_alarm(True, "static/alarm.wav", 1000, play_audio)
                  yawn_alarm_counter = 0
          else:
              if yawn_warning_time and (current_real_time - yawn_warning_time) > WARNING_DURATION:
                  yawn_warning_time = None
                  new_yawn = False
              yawn_alarm_counter = 0
          
          # Handle head movement detection
          if head_detected:
              if head_warning_time is None:
                  head_warning_time = current_real_time
                  stats["head_movement_detections"] += 1
                  stats["head_timestamps"].append(current_video_time)
                  new_head = True
                  
                  # Trigger alarm
                  trigger_alarm(True, "static/alarm.wav", 1000, play_audio)
              
              head_alarm_counter += 1
              if head_alarm_counter >= HEAD_ALARM_CYCLE:
                  trigger_alarm(True, "static/alarm.wav", 1000, play_audio)
                  head_alarm_counter = 0
          else:
              if head_warning_time and (current_real_time - head_warning_time) > WARNING_DURATION:
                  head_warning_time = None
                  new_head = False
              head_alarm_counter = 0
          
          # Draw results on frame with COLOR-CODED detection states
          result_frame = frame.copy()
          
          # Draw YOLO bounding boxes with COLOR-CODED styling based on detection state
          for box_info in boxes_to_draw:
              xmin, ymin, xmax, ymax = box_info['bbox']
              label = box_info['label']
              label_name = box_info['label_name']
              
              # Get color based on detection state
              color = get_detection_color(label_name, drowsy_detected, yawn_detected, head_detected)
              
              # Draw bounding box with state-based color
              cv2.rectangle(result_frame, (xmin, ymin), (xmax, ymax), color, 2)
              
              # Draw label text with same color (NO BACKGROUND)
              cv2.putText(result_frame, label, (xmin, ymin - 5), 
                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
          
          # Draw ViT prediction
          if vit_prediction is not None:
              vit_label = f"ViT: {VIT_LABELS[vit_prediction]} ({vit_confidence:.2f})"
              cv2.putText(result_frame, vit_label, (10, 30), 
                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
          
          # Draw FPS and average FPS - RESTORED
          if len(frame_times) > 0:
              current_fps = 1.0 / frame_times[-1] if frame_times[-1] > 0 else 0
              cv2.putText(result_frame, f"FPS: {current_fps:.1f}", (10, 30), 
                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
              cv2.putText(result_frame, f"AVG FPS: {avg_fps:.1f}", (10, 60), 
                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
          
          # Draw warning messages with original styling
          warning_y = 70
          if drowsy_detected:
              cv2.putText(result_frame, "Warning: DROWSY DETECTED!", (10, warning_y), 
                         cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 3)
              warning_y += 40
          
          if yawn_detected:
              cv2.putText(result_frame, "Warning: YAWN DETECTED!", (10, warning_y), 
                         cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 165, 255), 3)
              warning_y += 40
          
          if head_detected:
              cv2.putText(result_frame, "Warning: HEAD MOVEMENT!", (10, warning_y), 
                         cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 3)
          
          # Show display if requested
          if show_display:
              cv2.imshow('Drowsiness Detection', result_frame)
              if cv2.waitKey(1) & 0xFF == ord('q'):
                  break
          
          # Write frame to output video
          if video_writer:
              video_writer.write(result_frame)
          
          # Call frame callback if provided
          if frame_callback:
              frame_callback(result_frame, {
                  'drowsy': new_drowsy,
                  'yawn': new_yawn,
                  'head': new_head,
                  'vit_prediction': VIT_LABELS.get(vit_prediction, "Unknown") if vit_prediction is not None else None,
                  'vit_confidence': vit_confidence
              })
          
          # Calculate FPS
          detection_time = time.time() - det_start_time
          detection_time_total += detection_time
          
          frame_times.append(time.time() - loop_start_time)
          if len(frame_times) > 0:
              avg_fps = 1.0 / (sum(frame_times) / len(frame_times))
          
          # Control frame rate for video files
          if not is_webcam and not is_image:
              elapsed_time = time.time() - start_time
              expected_time = frame_count / fps
              if elapsed_time < expected_time:
                  time.sleep(expected_time - elapsed_time)
  
  except KeyboardInterrupt:
      print("Detection interrupted by user")
  except Exception as e:
      print(f"Error during detection: {str(e)}")
      stats["error"] = str(e)
  
  finally:
      # Cleanup
      if not is_image:
          cap.release()
      if video_writer:
          video_writer.release()
      if show_display:
          cv2.destroyAllWindows()
  
  # Calculate final statistics
  stats["avg_fps"] = avg_fps
  if frame_count > 0:
      stats["avg_detection_time"] = detection_time_total / frame_count
  
  print(f"Detection completed. Processed {frame_count} frames.")
  print(f"Drowsy detections: {stats['drowsy_detections']}")
  print(f"Yawn detections: {stats['yawn_detections']}")
  print(f"Head movement detections: {stats['head_movement_detections']}")
  print(f"Average FPS: {avg_fps:.2f}")
  
  return stats

# Backward compatibility function
def load_yolo_model(model_path):
  """Backward compatibility - load only YOLO model"""
  return load_model(model_path)[0]

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Drowsiness Detection')
  parser.add_argument('--model', type=str, required=True, help='Path to YOLO model')
  parser.add_argument('--vit-model', type=str, help='Path to ViT model (optional)')
  parser.add_argument('--source', type=str, default='0', help='Source (webcam=0, video file path, image path)')
  parser.add_argument('--output', type=str, help='Output video path')
  parser.add_argument('--no-display', action='store_true', help='Disable display')
  parser.add_argument('--no-audio', action='store_true', help='Disable audio alerts')
  
  args = parser.parse_args()
  
  # Convert source to int if it's a digit
  source = int(args.source) if args.source.isdigit() else args.source
  
  # Load models
  models = load_model(args.model, args.vit_model)
  
  # Run detection
  stats = detect(
      models,
      source,
      output_path=args.output,
      show_display=not args.no_display,
      play_audio=not args.no_audio
  )
  
  print(f"Final statistics: {stats}")