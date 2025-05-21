import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'hard-to-guess-string'
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///users.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    UPLOAD_FOLDER = 'static/uploads'
    ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
    ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'wmv'}
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB max upload
    MODEL_PATH = 'models/best.pt'

