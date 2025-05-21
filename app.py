import os
import shutil
import zipfile
import uuid
import datetime
import base64
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify, send_file, Response
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import sqlite3
import cv2
from detector import load_model, detect
import tempfile
import threading
import base64
import numpy as np
import time
import queue
import json
from pydub import AudioSegment
from moviepy.editor import VideoFileClip, AudioFileClip

# Initialize Flask app
app = Flask(__name__)
app.config.from_object('config.Config')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///instance/users.db'

# Ensure upload directories exist
os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'images'), exist_ok=True)
os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'videos'), exist_ok=True)
os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'results'), exist_ok=True)

# Initialize database
db = SQLAlchemy(app)

# Define User model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    private_key = db.Column(db.String(36), unique=True, nullable=False)
    
    def __init__(self, username, password):
        self.username = username
        self.password_hash = generate_password_hash(password)
        self.private_key = str(uuid.uuid4())

# Define Detection model to store detection results
class Detection(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    filename = db.Column(db.String(255), nullable=False)
    result_path = db.Column(db.String(255), nullable=True)
    detection_type = db.Column(db.String(20), nullable=False)  # 'image', 'video', or 'webcam'
    drowsy_count = db.Column(db.Integer, default=0)
    yawn_count = db.Column(db.Integer, default=0)
    head_movement_count = db.Column(db.Integer, default=0)
    avg_fps = db.Column(db.Float, default=0)
    total_frames = db.Column(db.Integer, default=0)
    timestamp = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    status = db.Column(db.String(20), default='pending')
    error_message = db.Column(db.Text, nullable=True)
    progress = db.Column(db.Integer, default=0)
    drowsy_timestamps = db.Column(db.Text, default='[]')
    yawn_timestamps = db.Column(db.Text, default='[]')
    head_timestamps = db.Column(db.Text, default='[]')

# Create database tables
with app.app_context():
    db.create_all()
    
    # Add default users if they don't exist
    if not User.query.filter_by(username='admin').first():
        admin = User('admin', 'admin123')
        db.session.add(admin)
    
    if not User.query.filter_by(username='user').first():
        user = User('user', 'user123')
        db.session.add(user)
    
    db.session.commit()

# Load YOLO model
model = load_model(app.config['MODEL_PATH'])

# Helper functions
def allowed_file(filename, allowed_extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

def get_file_type(filename):
    extension = filename.rsplit('.', 1)[1].lower()
    if extension in app.config['ALLOWED_IMAGE_EXTENSIONS']:
        return 'image'
    elif extension in app.config['ALLOWED_VIDEO_EXTENSIONS']:
        return 'video'
    return None

def update_progress(detection_id, progress):
    """Update the progress of a detection task"""
    with app.app_context():
        detection = Detection.query.get(detection_id)
        if detection:
            detection.progress = progress
            db.session.commit()

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = User.query.filter_by(username=username).first()
        
        if user and check_password_hash(user.password_hash, password):
            session['user_id'] = user.id
            session['username'] = user.username
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password', 'danger')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        if password != confirm_password:
            flash('Passwords do not match', 'danger')
            return render_template('register.html')
        
        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            flash('Username already exists', 'danger')
            return render_template('register.html')
        
        new_user = User(username, password)
        db.session.add(new_user)
        db.session.commit()
        
        flash('Registration successful! Please login.', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    session.pop('username', None)
    flash('You have been logged out', 'info')
    return redirect(url_for('index'))

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        flash('Please login first', 'warning')
        return redirect(url_for('login'))
    
    # Model information to display on dashboard
    model_info = {
        'name': 'YOLOv8 Drowsiness Detection',
        'description': 'This model detects driver drowsiness based on eye closure, yawning, and head movements.',
        'metrics': {
            'mAP': '0.984',
            'Precision': '0.961',
            'Recall': '0.974'
        },
        'classes': ['eyes_closed', 'eyes_closed_head_right', 'eyes_closed_head_left', 'focused', 'head_down', 'head_up', 'seeing_right', 'seeing_left', 'yawning']
    }
    
    return render_template('dashboard.html', username=session['username'], model_info=model_info)

def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'mp4', 'avi', 'mov', 'wmv'}
    return '.' in filename and \
          filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/file_detection', methods=['GET', 'POST'])
def file_detection():
    if 'user_id' not in session:
        flash('Please login first', 'warning')
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        # Check if a file was uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No selected file'})
        
        if file and allowed_file(file.filename):
            # Save the uploaded file
            filename = secure_filename(file.filename)
            unique_filename = f"{uuid.uuid4()}_{filename}"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(file_path)
            
            # Determine if it's an image or video
            file_ext = os.path.splitext(filename)[1].lower()
            if file_ext in ['.jpg', '.jpeg', '.png', '.gif']:
                detection_type = 'image'
            else:
                detection_type = 'video'
            
            # Generate a unique filename for the result
            result_filename = f"result_{unique_filename}"
            result_path = os.path.join(app.config['UPLOAD_FOLDER'], 'results', result_filename)
            
            try:
                # Create detection record first
                detection = Detection(
                    user_id=session['user_id'],
                    filename=filename,
                    result_path=result_path,
                    detection_type=detection_type,
                    status='processing'
                )
                db.session.add(detection)
                db.session.commit()
                
                # Start processing in a separate thread
                thread = threading.Thread(
                    target=process_detection,
                    args=(model, file_path, result_path, detection.id)
                )
                thread.daemon = True
                thread.start()
                
                return jsonify({
                    'message': 'Processing started',
                    'detection_id': detection.id
                })
            
            except Exception as e:
                app.logger.error(f"Detection error: {str(e)}")
                return jsonify({'error': str(e)})
        else:
            return jsonify({'error': 'File type not allowed'})
    
    return render_template('file_detection.html')
@app.route('/detection_progress/<int:detection_id>')
def detection_progress(detection_id):
    """Get the progress of a detection task"""
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    detection = Detection.query.get(detection_id)
    if not detection:
        return jsonify({'error': 'Detection not found'}), 404
    
    if detection.user_id != session['user_id']:
        return jsonify({'error': 'Unauthorized'}), 403
    
    return jsonify({
        'progress': detection.progress,
        'status': detection.status,
        'error_message': detection.error_message
    })

@app.route('/detection_result/<int:detection_id>')
def detection_result(detection_id):
    """Get the result of a detection task"""
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    detection = Detection.query.get(detection_id)
    if not detection:
        return jsonify({'error': 'Detection not found'}), 404
    
    if detection.user_id != session['user_id']:
        return jsonify({'error': 'Unauthorized'}), 403
    
    # Get the relative path for the result file
    result_url = None
    if detection.result_path:
        # Convert absolute path to URL path
        result_filename = os.path.basename(detection.result_path)
        result_url = url_for('static', filename=f'uploads/results/{result_filename}')
    
    # Parse timestamps from JSON strings
    try:
        drowsy_timestamps = json.loads(detection.drowsy_timestamps)
        yawn_timestamps = json.loads(detection.yawn_timestamps)
        head_timestamps = json.loads(detection.head_timestamps)
    except:
        drowsy_timestamps = []
        yawn_timestamps = []
        head_timestamps = []
    
    return jsonify({
        'id': detection.id,
        'filename': detection.filename,
        'detection_type': detection.detection_type,
        'drowsy_count': detection.drowsy_count,
        'yawn_count': detection.yawn_count,
        'head_movement_count': detection.head_movement_count,
        'avg_fps': detection.avg_fps,
        'total_frames': detection.total_frames,
        'timestamp': detection.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
        'status': detection.status,
        'result_url': result_url,
        'drowsy_timestamps': drowsy_timestamps,
        'yawn_timestamps': yawn_timestamps,
        'head_timestamps': head_timestamps
    })


def process_detection(model, file_path, result_path, detection_id):
    """Process detection in a separate thread"""
    temp_dir = None
    try:
        # [1] Cập nhật trạng thái processing
        with app.app_context():
            detection = Detection.query.get(detection_id)
            detection.status = 'processing'
            db.session.commit()

        # [2] Chạy detection và thu thập dữ liệu
        stats = detect(
            model, 
            file_path,
            output_path=result_path,
            show_display=False,
            progress_callback=lambda p: update_progress(detection_id, p),
            play_audio=False
        )

        # [3] Kiểm tra loại file
        with app.app_context():
            detection = Detection.query.get(detection_id)
            # Chỉ xử lý audio khi là file video, bỏ qua nếu là ảnh
            if detection.detection_type == 'video':
                # [4] Tạo thư mục tạm
                temp_dir = tempfile.mkdtemp()
                temp_audio_path = os.path.join(temp_dir, "alarm_track.wav")
                temp_video_path = os.path.join(temp_dir, "output.mp4")

                # [5] Tạo audio cảnh báo
                # Lấy timestamps từ kết quả detection
                drowsy_ts = stats.get('drowsy_timestamps', [])
                yawn_ts = stats.get('yawn_timestamps', [])
                head_ts = stats.get('head_timestamps', [])

                # Tạo audio track với độ dài video
                total_duration = (stats['total_frames'] / stats['avg_fps']) if stats['avg_fps'] > 0 else 0
                base_audio = AudioSegment.silent(duration=int(total_duration * 1000))
                
                # Tải các file âm thanh cảnh báo khác nhau
                drowsy_alarm = AudioSegment.from_wav(os.path.join(app.root_path, 'static', 'alarm.wav'))
                yawn_alarm = AudioSegment.from_wav(os.path.join(app.root_path, 'static', 'alarm.wav'))
                head_alarm = AudioSegment.from_wav(os.path.join(app.root_path, 'static', 'alarm.wav'))
                
                # Xử lý từng loại cảnh báo riêng biệt
                # Tạo danh sách các khoảng thời gian đã được sử dụng để tránh chồng chéo
                used_time_ranges = []
                
                # Xử lý cảnh báo drowsy (ưu tiên cao nhất)
                for ts in drowsy_ts:
                    start_ms = int(ts * 1000)
                    end_ms = start_ms + 3000  # Drowsy alarm kéo dài 3s
                    
                    # Kiểm tra xem khoảng thời gian này đã được sử dụng chưa
                    if not any(start < end_ms and end > start_ms for start, end in used_time_ranges):
                        if start_ms + 3000 <= len(base_audio):
                            base_audio = base_audio.overlay(drowsy_alarm[:3000], position=start_ms)
                            used_time_ranges.append((start_ms, end_ms))
                
                # Xử lý cảnh báo yawn (ưu tiên thứ hai)
                for ts in yawn_ts:
                    start_ms = int(ts * 1000)
                    end_ms = start_ms + 1000  # Yawn alarm kéo dài 1s
                    
                    # Kiểm tra xem khoảng thời gian này đã được sử dụng chưa
                    if not any(start < end_ms and end > start_ms for start, end in used_time_ranges):
                        if start_ms + 1000 <= len(base_audio):
                            base_audio = base_audio.overlay(yawn_alarm[:1000], position=start_ms)
                            used_time_ranges.append((start_ms, end_ms))
                
                # Xử lý cảnh báo head movement (ưu tiên thấp nhất)
                for ts in head_ts:
                    start_ms = int(ts * 1000)
                    end_ms = start_ms + 1000  # Head alarm kéo dài 1s
                    
                    # Kiểm tra xem khoảng thời gian này đã được sử dụng chưa
                    if not any(start < end_ms and end > start_ms for start, end in used_time_ranges):
                        if start_ms + 1000 <= len(base_audio):
                            base_audio = base_audio.overlay(head_alarm[:1000], position=start_ms)
                            used_time_ranges.append((start_ms, end_ms))

                # Xuất file audio tạm
                base_audio.export(temp_audio_path, format="wav")

                # [6] Merge audio vào video
                with VideoFileClip(result_path) as video:
                    with AudioFileClip(temp_audio_path) as audio:
                        final = video.set_audio(audio)
                        final.write_videofile(
                            temp_video_path,
                            codec='libx264',
                            audio_codec='aac',
                            threads=4,
                            verbose=False,
                            ffmpeg_params=['-movflags', '+faststart']
                        )

                # [7] Thay thế file kết quả
                shutil.move(temp_video_path, result_path)

        # [8] Cập nhật database
        with app.app_context():
            detection = Detection.query.get(detection_id)
            detection.status = 'completed'
            detection.drowsy_count = len(stats.get('drowsy_timestamps', []))
            detection.yawn_count = len(stats.get('yawn_timestamps', []))
            detection.head_movement_count = len(stats.get('head_timestamps', []))
            detection.avg_fps = stats.get('avg_fps', 0)
            detection.total_frames = stats.get('total_frames', 0)
            db.session.commit()

    except Exception as e:
        app.logger.error(f"LỖI XỬ LÝ: {str(e)}")
        with app.app_context():
            detection = Detection.query.get(detection_id)
            detection.status = 'failed'
            detection.error_message = str(e)
            db.session.commit()

    finally:
        # [9] Dọn dẹp tài nguyên
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                app.logger.error(f"Lỗi dọn dẹp: {str(e)}")



@app.route('/webcam_detection')
def webcam_detection():
    if 'user_id' not in session:
        flash('Please login first', 'warning')
        return redirect(url_for('login'))
    
    return render_template('webcam_detection.html')

frame_queue = queue.Queue(maxsize=10)
current_frame = None
stop_threads = False

@app.route('/video_feed/<int:detection_id>')
def video_feed(detection_id):
    """Video streaming route for webcam detection"""
    if 'user_id' not in session:
        return "Unauthorized", 401
    
    # Check if detection exists and belongs to user
    detection = Detection.query.get_or_404(detection_id)
    if detection.user_id != session['user_id']:
        return "Unauthorized", 403
    
    def generate_frames():
        global current_frame, stop_threads
        
        # Reset stop flag
        stop_threads = False
        current_frame = None
        # Initialize counters for this session
        drowsy_count = 0
        yawn_count = 0
        head_count = 0
        frame_count = 0
        
        # Start detection thread
        detection_thread = threading.Thread(
            target=webcam_detection_thread,
            args=(detection_id,)
        )
        detection_thread.daemon = True
        detection_thread.start()
        
        try:
            last_update_time = time.time()
            
            while not stop_threads:
                # If we have a new frame
                if current_frame is not None:
                    # Encode frame as JPEG
                    ret, buffer = cv2.imencode('.jpg', current_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                    frame_bytes = buffer.tobytes()
                    
                    # Update detection counters based on frame content
                    frame_count += 1
                    
                    # Update database periodically (every 2 seconds)
                    current_time = time.time()
                    if current_time - last_update_time > 2:
                        with app.app_context():
                            detection = Detection.query.get(detection_id)
                            if detection:
                                # Update with the latest counts from the global variables
                                detection.drowsy_count = app.config.get('drowsy_count', 0)
                                detection.yawn_count = app.config.get('yawn_count', 0)
                                detection.head_movement_count = app.config.get('head_count', 0)
                                detection.total_frames = frame_count
                                db.session.commit()
                        last_update_time = current_time
                    
                    # Yield frame as bytes
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
                # Sleep to reduce CPU load
                time.sleep(0.01)  # ~30 FPS
                continue
        except GeneratorExit:
            # When client disconnects
            stop_threads = True
            current_frame = None
    
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/start_webcam_detection', methods=['POST'])
def start_webcam_detection():
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    # Tạo record cho session webcam này
    detection = Detection(
        user_id=session['user_id'],
        filename='webcam_session',
        result_path=None,
        detection_type='webcam',
        status='processing'
    )
    db.session.add(detection)
    db.session.commit()
    
    # Tạo output path
    result_filename = f"webcam_{detection.id}.mp4"
    result_path = os.path.join(app.config['UPLOAD_FOLDER'], 'results', result_filename)
    detection.result_path = result_path
    db.session.commit()
    
    return jsonify({
        'message': 'Webcam detection started',
        'detection_id': detection.id
    })

@app.route('/stop_webcam_detection', methods=['POST'])
def stop_webcam_detection():
    """Stop webcam detection"""
    global stop_threads
    
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    # Đặt cờ dừng để kết thúc thread detection
    stop_threads = True
    
    current_frame = None
    
    # Cập nhật detection record nếu có active
    if request.json and 'detection_id' in request.json:
        detection_id = request.json['detection_id']
        detection = Detection.query.get(detection_id)
        if detection and detection.user_id == session['user_id']:
            detection.status = 'completed'
            db.session.commit()
    
    return jsonify({'message': 'Webcam detection stopped'})



# app.py (sửa hàm webcam_detection_thread)
def webcam_detection_thread(detection_id):
    global current_frame, stop_threads
    
    app.config['drowsy_count'] = 0
    app.config['yawn_count'] = 0
    app.config['head_count'] = 0
    
    try:
        def frame_callback(frame, stats=None):
            global current_frame
            current_frame = frame
            if stats:
                if stats.get('new_drowsy', False):
                    app.config['drowsy_count'] += 1
                if stats.get('new_yawn', False):
                    app.config['yawn_count'] += 1
                if stats.get('new_head', False):
                    app.config['head_count'] += 1
        
        # Gọi hàm detect và lưu kết quả trả về
        stats = detect(
            model=model,
            source=0,
            output_path=None,
            show_display=False,
            frame_callback=frame_callback,
            stop_flag=lambda: stop_threads
        )
        
        # Cập nhật thông số vào database
        with app.app_context():
            detection = Detection.query.get(detection_id)
            if detection:
                detection.avg_fps = stats.get('avg_fps', 0)
                detection.drowsy_count = app.config['drowsy_count']
                detection.yawn_count = app.config['yawn_count']
                detection.head_movement_count = app.config['head_count']
                detection.total_frames = stats.get('total_frames', 0)
                detection.drowsy_timestamps = json.dumps(stats.get('drowsy_timestamps', []))
                detection.yawn_timestamps = json.dumps(stats.get('yawn_timestamps', []))
                detection.head_timestamps = json.dumps(stats.get('head_timestamps', []))

                detection.status = 'completed'
                db.session.commit()
                   
    except Exception as e:
        app.logger.error(f"Error in webcam detection thread: {str(e)}")
        with app.app_context():
            detection = Detection.query.get(detection_id)
            if detection:
                detection.status = 'failed'
                detection.error_message = str(e)
                db.session.commit()
    finally:
        cv2.destroyAllWindows()
        stop_threads = True


@app.route('/save_webcam_recording', methods=['POST'])
def save_webcam_recording():
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    data = request.json
    detection_id = data.get('detection_id')
    recording_data = data.get('recording')
    stats = data.get('stats', {})
    
    if not detection_id or not recording_data:
        return jsonify({'error': 'Missing required data'}), 400
    
    try:
        # Lấy bản ghi detection từ database
        detection = Detection.query.get_or_404(detection_id)
        
        # Kiểm tra quyền sở hữu
        if detection.user_id != session['user_id']:
            return jsonify({'error': 'Access denied'}), 403
        
        # Lưu bản ghi video từ dữ liệu base64
        recording_filename = f"webcam_recording_{detection_id}.webm"
        recording_path = os.path.join(app.config['UPLOAD_FOLDER'], 'results', recording_filename)
        os.makedirs(os.path.dirname(recording_path), exist_ok=True)
        
        # Xử lý dữ liệu base64
        if ',' in recording_data:
            recording_data = recording_data.split(',')[1]
        
        with open(recording_path, 'wb') as f:
            f.write(base64.b64decode(recording_data))
        
        # Chuyển đổi sang định dạng MP4
        mp4_path = os.path.join(app.config['UPLOAD_FOLDER'], 'results', f"webcam_recording_{detection_id}.mp4")
        result_path = mp4_path
        
        try:
            import subprocess
            subprocess.run([
                'ffmpeg', '-y', 
                '-i', recording_path,
                '-c:v', 'libx264', 
                '-preset', 'fast',
                '-pix_fmt', 'yuv420p',
                '-movflags', '+faststart',
                mp4_path
            ], check=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE, timeout=60)
        except Exception as e:
            app.logger.error(f"Lỗi chuyển đổi video: {str(e)}")
            result_path = recording_path

        # Xử lý âm thanh cảnh báo
        try:
            from pydub import AudioSegment
            import json

            # Lấy timestamps từ database
            drowsy_ts = json.loads(detection.drowsy_timestamps or '[]')
            yawn_ts = json.loads(detection.yawn_timestamps or '[]')
            head_ts = json.loads(detection.head_timestamps or '[]')

            # Tạo audio track với độ dài video
            total_duration = (detection.total_frames / detection.avg_fps) if detection.avg_fps > 0 else 0
            base_audio = AudioSegment.silent(duration=int(total_duration * 1000))
            
            # Tải các file âm thanh cảnh báo
            drowsy_alarm = AudioSegment.from_wav(os.path.join(app.root_path, 'static', 'alarm.wav'))
            yawn_alarm = AudioSegment.from_wav(os.path.join(app.root_path, 'static', 'alarm.wav'))
            head_alarm = AudioSegment.from_wav(os.path.join(app.root_path, 'static', 'alarm.wav'))
            
            # Danh sách khoảng thời gian đã được sử dụng
            used_time_ranges = []
            
            # Xử lý cảnh báo drowsy (ưu tiên cao nhất)
            for ts in drowsy_ts:
                start_ms = int(ts * 1000)
                end_ms = start_ms + 3000  # Drowsy alarm kéo dài 3s
                
                # Kiểm tra xem khoảng thời gian này đã được sử dụng chưa
                if not any(start < end_ms and end > start_ms for start, end in used_time_ranges):
                    if start_ms + 3000 <= len(base_audio):
                        base_audio = base_audio.overlay(drowsy_alarm[:3000], position=start_ms)
                        used_time_ranges.append((start_ms, end_ms))
            
            # Xử lý cảnh báo yawn (ưu tiên thứ hai)
            for ts in yawn_ts:
                start_ms = int(ts * 1000)
                end_ms = start_ms + 1000  # Yawn alarm kéo dài 1s
                
                # Kiểm tra xem khoảng thời gian này đã được sử dụng chưa
                if not any(start < end_ms and end > start_ms for start, end in used_time_ranges):
                    if start_ms + 1000 <= len(base_audio):
                        base_audio = base_audio.overlay(yawn_alarm[:1000], position=start_ms)
                        used_time_ranges.append((start_ms, end_ms))
            
            # Xử lý cảnh báo head movement (ưu tiên thấp nhất)
            for ts in head_ts:
                start_ms = int(ts * 1000)
                end_ms = start_ms + 1000  # Head alarm kéo dài 1s
                
                # Kiểm tra xem khoảng thời gian này đã được sử dụng chưa
                if not any(start < end_ms and end > start_ms for start, end in used_time_ranges):
                    if start_ms + 1000 <= len(base_audio):
                        base_audio = base_audio.overlay(head_alarm[:1000], position=start_ms)
                        used_time_ranges.append((start_ms, end_ms))

            # Merge audio vào video
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio:
                base_audio.export(tmp_audio.name, format="wav")
                
                final_output = os.path.join(app.config['UPLOAD_FOLDER'], 'results', f"final_{detection_id}.mp4")
                subprocess.run([
                    'ffmpeg', '-y',
                    '-i', result_path,
                    '-i', tmp_audio.name,
                    '-c:v', 'copy',
                    '-c:a', 'aac',
                    '-strict', 'experimental',
                    final_output
                ], check=True)
                
                os.replace(final_output, result_path)

        except Exception as e:
            app.logger.error(f"Lỗi ghép âm thanh: {str(e)}")

        # Cập nhật database
        detection.result_path = result_path
        detection.drowsy_count = stats.get('drowsy_detections', 0)
        detection.yawn_count = stats.get('yawn_detections', 0)
        detection.head_movement_count = stats.get('head_movement_detections', 0)
        detection.status = 'completed'
        db.session.commit()

        # Tạo symlink cho static
        static_dir = os.path.join(app.root_path, 'static', 'results')
        os.makedirs(static_dir, exist_ok=True)
        static_path = os.path.join(static_dir, f"webcam_{detection_id}_{int(time.time())}.mp4")
        shutil.copy2(result_path, static_path)

        return jsonify({
            'message': 'Lưu bản ghi thành công',
            'result_path': result_path
        })

    except Exception as e:
        app.logger.error(f"Lỗi hệ thống: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/check_processing_status/<int:detection_id>', methods=['GET'])
def check_processing_status(detection_id):
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    detection = Detection.query.get_or_404(detection_id)
    
    # Check if the detection belongs to the current user
    if detection.user_id != session['user_id']:
        return jsonify({'error': 'Access denied'}), 403
    
    return jsonify({
        'id': detection.id,
        'status': detection.status,
        'progress': detection.progress,
        'drowsy_count': detection.drowsy_count,
        'yawn_count': detection.yawn_count,
        'head_movement_count': detection.head_movement_count,
        'error_message': detection.error_message
    })

@app.route('/run_webcam_detection')
def run_webcam_detection():
    if 'user_id' not in session:
        flash('Please login first', 'warning')
        return redirect(url_for('login'))
    
    # This route is now just for backward compatibility
    # The actual webcam handling is done in JavaScript
    return redirect(url_for('webcam_detection'))

@app.route('/statistics')
def statistics():
    if 'user_id' not in session:
        flash('Please login first', 'warning')
        return redirect(url_for('login'))
    
    # Get all detections for the current user
    detections = Detection.query.filter_by(user_id=session['user_id']).order_by(Detection.timestamp.desc()).all()
    
    return render_template('statistics.html', detections=detections)

@app.route('/view_result/<int:detection_id>')
def view_result(detection_id):
    if 'user_id' not in session:
        flash('Please login first', 'warning')
        return redirect(url_for('login'))
    
    detection = Detection.query.get_or_404(detection_id)
    
    # Check if the detection belongs to the current user
    if detection.user_id != session['user_id']:
        flash('Access denied', 'danger')
        return redirect(url_for('statistics'))
    
    # Get the relative path for display
    result_path = detection.result_path
    result_url = None
    
    if result_path and os.path.exists(result_path):
        # Tạo thư mục static nếu chưa tồn tại
        static_results_dir = os.path.join(app.root_path, 'static', 'results')
        os.makedirs(static_results_dir, exist_ok=True)
        
        # Tạo tên file duy nhất
        filename = f"{detection.id}_{os.path.basename(result_path)}"
        static_result_path = os.path.join(static_results_dir, filename)
        
        try:
            # Luôn copy file mới mỗi lần xem
            shutil.copy2(result_path, static_result_path)
            
            # Thêm timestamp để tránh cache
            timestamp = int(os.path.getmtime(static_result_path))
            result_url = url_for('static', filename=f'results/{filename}') + f'?v={timestamp}'

        except Exception as e:
            app.logger.error(f"Error copying result file: {e}")
            result_url = None
        
        # If the file exists, generate the URL
        if os.path.exists(static_result_path):
            result_url = url_for('static', filename=f'results/{filename}')
            
            # For videos, ensure proper format
            if detection.detection_type in ['video', 'webcam']:
                try:
                    import subprocess
                    # Luôn convert sang MP4 với định dạng chuẩn cho web
                    mp4_path = os.path.splitext(static_result_path)[0] + '_converted.mp4'
                    if not os.path.exists(mp4_path):
                        subprocess.run([
                            'ffmpeg', '-y', '-i', static_result_path,
                            '-c:v', 'libx264', 
                            '-preset', 'fast',
                            '-profile:v', 'main',
                            '-pix_fmt', 'yuv420p',
                            '-movflags', '+faststart',
                            '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',
                            '-c:a', 'aac',
                            '-b:a', '128k',
                            mp4_path
                        ], check=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
                    
                    # Sử dụng file đã convert
                    result_url = url_for('static', filename=f'results/{os.path.basename(mp4_path)}') + f'?v={timestamp}'
                    
                except Exception as e:
                    app.logger.error(f"Lỗi convert video: {str(e)}")
                    result_url = url_for('static', filename=f'results/{filename}')
                        # try:
                    #     # Try to convert to MP4 using ffmpeg if available
                    #     import subprocess
                    #     mp4_path = os.path.splitext(static_result_path)[0] + '.mp4'
                    #     if not os.path.exists(mp4_path):
                    #         subprocess.run([
                    #             'ffmpeg', '-i', static_result_path, 
                    #             '-c:v', 'libx264', '-preset', 'fast',
                    #             '-c:a', 'aac', '-movflags', '+faststart',
                    #             mp4_path
                    #         ], check=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
                            
                    #         if os.path.exists(mp4_path):
                    #             result_url = url_for('static', filename=f'results/{os.path.basename(mp4_path)}')
                    # except (ImportError, subprocess.SubprocessError, FileNotFoundError) as e:
                    #     app.logger.error(f"Video conversion error: {e}")
    
    return render_template('view_result.html', 
                           detection=detection, 
                           result_url=result_url)



@app.route('/download_result/<int:detection_id>')
def download_result(detection_id):
    if 'user_id' not in session:
        flash('Please login first', 'warning')
        return redirect(url_for('login'))
    
    detection = Detection.query.get_or_404(detection_id)
    
    # Check if the detection belongs to the current user
    if detection.user_id != session['user_id']:
        flash('Access denied', 'danger')
        return redirect(url_for('statistics'))
    
    # Check if the result file exists
    if not detection.result_path or not os.path.exists(detection.result_path):
        flash('Result file not found', 'danger')
        return redirect(url_for('statistics'))
    
    # Create a zip file containing the result
    zip_filename = f"detection_result_{detection.id}.zip"
    zip_path = os.path.join(app.config['UPLOAD_FOLDER'], 'results', zip_filename)
    
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        # Add the result file to the zip
        zipf.write(detection.result_path, os.path.basename(detection.result_path))
        
        # Add a metadata file with detection information
        metadata_content = f"""Detection Information:
        ID: {detection.id}
        Type: {detection.detection_type}
        Timestamp: {detection.timestamp}
        Drowsy Detections: {detection.drowsy_count}
        Yawn Detections: {detection.yawn_count}
        Head Movement Detections: {detection.head_movement_count}
        Average FPS: {detection.avg_fps}
        Total Frames: {detection.total_frames}
        """
        
        zipf.writestr('metadata.txt', metadata_content)
    
    return send_file(zip_path, as_attachment=True, download_name=zip_filename)

@app.route('/download_all_results')
def download_all_results():
    if 'user_id' not in session:
        flash('Please login first', 'warning')
        return redirect(url_for('login'))
    
    # Get all detections for the current user
    detections = Detection.query.filter_by(user_id=session['user_id']).all()
    
    if not detections:
        flash('No detection results found', 'warning')
        return redirect(url_for('statistics'))
    
    # Create a zip file containing all results
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    zip_filename = f"all_detection_results_{timestamp}.zip"
    zip_path = os.path.join(app.config['UPLOAD_FOLDER'], 'results', zip_filename)
    
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        # Add each result file to the zip
        for detection in detections:
            if detection.result_path and os.path.exists(detection.result_path):
                # Create a unique name in the zip to avoid conflicts
                zip_name = f"{detection.detection_type}_{detection.id}_{os.path.basename(detection.result_path)}"
                zipf.write(detection.result_path, zip_name)
        
        # Add a metadata file with summary information
        metadata_content = "Detection Results Summary:\n\n"
        for detection in detections:
            metadata_content += f"""ID: {detection.id}
            Type: {detection.detection_type}
            Filename: {detection.filename}
            Timestamp: {detection.timestamp}
            Drowsy Detections: {detection.drowsy_count}
            Yawn Detections: {detection.yawn_count}
            Head Movement Detections: {detection.head_movement_count}
            Average FPS: {detection.avg_fps}
            Total Frames: {detection.total_frames}
            
            """
        
        zipf.writestr('summary.txt', metadata_content)
    
    return send_file(zip_path, as_attachment=True, download_name=zip_filename)

@app.route('/model_info')
def model_info():
    if 'user_id' not in session:
        flash('Please login first', 'warning')
        return redirect(url_for('login'))
    
    # Model information
    model_info = {
        'name': 'YOLOv8 Drowsiness Detection',
        'description': 'This model is trained to detect driver drowsiness based on eye closure, yawning, and head movements.',
        'architecture': 'YOLOv8',
        'dataset': 'Custom drowsiness detection dataset with annotated images of drivers in various states.',
        'metrics': {
            'mAP': '0.984',
            'Precision': '0.961',
            'Recall': '0.974',
        },
        'classes': [
            {'name': 'Eye Closed', 'id': 0},
            {'name': 'Eye Closed Head Right', 'id': 1},
            {'name': 'Eye Closed Head Left', 'id': 2},
            {'name': 'Focused', 'id': 3},
            {'name': 'Head Down', 'id': 4},
            {'name': 'Head Up', 'id': 5},
            {'name': 'Seeing Right', 'id': 6},
            {'name': 'Seeing Left', 'id': 7},
            {'name': 'Yawning', 'id': 8}
        ],
        'training_details': {
            'epochs': 100,
            'batch_size': 16,
            'optimizer': 'AdamW',
            'learning_rate': '0.000769',
            'augmentation': 'Rotation, brightness, contrast'
        }
    }
    
    return render_template('model_info.html', model_info=model_info)

if __name__ == '__main__':
    app.run(debug=False, threaded=True)