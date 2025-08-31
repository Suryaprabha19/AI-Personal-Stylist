import os
from flask import Flask, request, render_template, redirect, url_for, flash, send_from_directory
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from sklearn.cluster import MiniBatchKMeans
import secrets

app = Flask(__name__)

# --- Flask Configuration ---
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY', secrets.token_hex(32))
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- Mediapipe ---
mp_pose = mp.solutions.pose
mp_face = mp.solutions.face_mesh

# --- Color Classification & Image Helpers (Unchanged) ---
def classify_hair_color(rgb):
    r, g, b = rgb
    brightness = (r + g + b) / 3

    # Black (very dark values across all channels)
    if r < 80 and g < 80 and b < 80:
        return "Black"

    # Brown (medium brightness, red slightly dominant)
    if 60 <= brightness < 160 and r >= g and r >= b:
        return "Brown"

    # Blonde (light/bright, all channels relatively high)
    if brightness >= 160 and abs(r - g) < 60 and b < 180:
        return "Blonde"

    # Red (red clearly dominant over green/blue)
    if r > 120 and (r - g) > 50 and (r - b) > 50:
        return "Red"

    return "Unknown"


def classify_eye_color(rgb):
    r, g, b = rgb
    if 60 <= r <= 140 and 40 <= g <= 110 and 30 <= b <= 110: return "Brown"
    elif 70 <= r <= 130 and 100 <= g <= 170 and 80 <= b <= 150: return "Green"
    elif 60 <= r <= 140 and 100 <= g <= 160 and 140 <= b <= 255: return "Blue"
    else: return "Unknown"

def classify_skin_tone(rgb):
    r, g, b = rgb
    if 245 <= r <= 255 and 220 <= g <= 245 and 200 <= b <= 240: return "Very Fair"
    elif 220 <= r <= 245 and 190 <= g <= 230 and 170 <= b <= 210: return "Fair"
    elif 180 <= r <= 220 and 150 <= g <= 200 and 120 <= b <= 180: return "Medium"
    elif 140 <= r <= 190 and 110 <= g <= 160 and 90 <= b <= 140: return "Olive"
    else: return "Brown"

def detect_undertone(rgb):
    r, g, b = rgb
    if r > b and g > b: return "Warm"
    elif b > r and b > g: return "Cool"
    else: return "Neutral"

def get_dominant_color(roi_pixels, k=1):
    if roi_pixels is None or roi_pixels.size == 0: return None
    pixels = roi_pixels.reshape(-1, 3).astype(np.float32)
    if len(np.unique(pixels, axis=0)) < k: k = 1
    kmeans = MiniBatchKMeans(n_clusters=k, n_init='auto', random_state=42).fit(pixels)
    return tuple(map(int, kmeans.cluster_centers_[0]))

def analyze_wrist_color(wrist_image_path):
    image = cv2.imread(wrist_image_path)
    if image is None: return "Unknown", "Unknown"
    h, w, _ = image.shape
    patch_size = min(w, h, 60)
    center = image[h//2 - patch_size//2 : h//2 + patch_size//2, w//2 - patch_size//2 : w//2 + patch_size//2]
    color = get_dominant_color(cv2.cvtColor(center, cv2.COLOR_BGR2RGB))
    return (classify_skin_tone(color), detect_undertone(color)) if color else ("Unknown", "Unknown")

def extract_face_features(rgb_image):
    eye_label, hair_label = "Unknown", "Unknown"
    with mp_face.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
        results = face_mesh.process(rgb_image)
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        h, w, _ = rgb_image.shape
        eye_indices = [33, 160, 158, 133, 144, 145, 362, 385, 387, 263, 373, 374]
        eye_pixels = np.array([rgb_image[int(landmarks[i].y * h), int(landmarks[i].x * w)] for i in eye_indices])
        eye_color = get_dominant_color(eye_pixels)
        if eye_color: eye_label = classify_eye_color(eye_color)
        face_y_min = min(int(lm.y * h) for lm in landmarks)
        hair_roi = rgb_image[max(0, face_y_min - 50):max(0, face_y_min - 10), w//2-50:w//2+50]
        if hair_roi.size > 0:
            hair_color = get_dominant_color(hair_roi)
            if hair_color: hair_label = classify_hair_color(hair_color)
    return eye_label, hair_label

def get_pixel_coords(landmarks, idx, w, h):
    if landmarks[idx].visibility > 0.5:
        return (int(landmarks[idx].x * w), int(landmarks[idx].y * h))
    return None

# --- MODIFIED: Function now only calculates measurements for torso length ---
def calculate_torso_measurements(landmarks, image_shape, height_cm):
    h, w, _ = image_shape
    cm = {}

    lm_coords = {id.name: get_pixel_coords(landmarks, id.value, w, h) for id in mp_pose.PoseLandmark}
    
    required = ['LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_HIP', 'RIGHT_HIP', 'NOSE', 'LEFT_KNEE', 'RIGHT_KNEE']
    if not all(lm_coords.get(name) for name in required):
        print("Required landmarks for torso calculation not detected.")
        if height_cm and height_cm > 0:
            cm['torso_length_cm'] = round(height_cm * 0.27, 2)
            cm['full_leg_length_estimated_cm'] = round(height_cm * 0.45, 2)
        return cm

    visible_height_px = (lm_coords['LEFT_KNEE'][1] + lm_coords['RIGHT_KNEE'][1]) / 2 - lm_coords['NOSE'][1]
    
    if visible_height_px > 0 and height_cm > 0:
        px_per_cm = visible_height_px / (height_cm * 0.65)
        if px_per_cm > 0:
            shoulder_y = (lm_coords['LEFT_SHOULDER'][1] + lm_coords['RIGHT_SHOULDER'][1]) / 2
            hip_y = (lm_coords['LEFT_HIP'][1] + lm_coords['RIGHT_HIP'][1]) / 2
            knee_y = (lm_coords['LEFT_KNEE'][1] + lm_coords['RIGHT_KNEE'][1]) / 2
            
            cm['torso_length_cm'] = round(abs(hip_y - shoulder_y) / px_per_cm, 2)
            cm['full_leg_length_estimated_cm'] = round((abs(knee_y - hip_y) / px_per_cm) * 1.8, 2)

    if not cm: # Fallback if calculations fail
        cm['torso_length_cm'] = round(height_cm * 0.27, 2)
        cm['full_leg_length_estimated_cm'] = round(height_cm * 0.45, 2)

    return cm


def torso_leg_balance(cm):
    torso = cm.get('torso_length_cm', 0)
    leg = cm.get('full_leg_length_estimated_cm', 0)
    if not all([torso, leg]): return "Not enough data"
    if torso > leg: return "Long Torso"
    elif leg > torso: return "Short Torso"
    else: return "Balanced"

def get_recommendations(features):
    try:
        df = pd.read_csv('dataset/recommendations.csv')
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    except Exception as e:
        return [{"type": "Error", "suggestion": f"Dataset error: {e}"}]

    for key in ['hair_color', 'eye_color', 'skin_tone', 'undertone', 'body_shape', 'torso_length']:
        if features.get(key, "Unknown") in ["Unknown", "Not enough data", "", None]:
            return [{"type": "NoMatch", "suggestion": f"Cannot recommend: '{key}' could not be determined."}]
    
    query_string = " & ".join([f"`{col}` == '{val}'" for col, val in {
        'hair_color': features['hair_color'], 'eye_color': features['eye_color'],
        'skin_tone': features['skin_tone'], 'under_tone': features['undertone'],
        'body_proportion': features['body_shape'], 'torso_length': features['torso_length']
    }.items() if col in df.columns])

    matches = df.query(query_string)
    if not matches.empty:
        return [matches.iloc[0].to_dict()]
    return [{"type": "NoMatch", "suggestion": "No exact recommendation found for your unique features."}]

def process_features(full_body_path, wrist_path, height_cm, gender, body_shape):
    features = {
        'gender': gender, 'height': height_cm, 'body_shape': body_shape,
        'skin_tone': 'Unknown', 'undertone': 'Unknown',
        'eye_color': 'Unknown', 'hair_color': 'Unknown',
        'torso_length': 'Not enough data' # Default value
    }
    
    features['skin_tone'], features['undertone'] = analyze_wrist_color(wrist_path)

    image = cv2.imread(full_body_path)
    if image is None: return features
    
    h, w, _ = image.shape
    resized_image = cv2.resize(image, (600, int(h * 600 / w)))
    rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
    
    features['eye_color'], features['hair_color'] = extract_face_features(rgb_image)

    with mp_pose.Pose(static_image_mode=True, model_complexity=1) as pose:
        results = pose.process(rgb_image)
    
    if results.pose_landmarks:
        cm = calculate_torso_measurements(results.pose_landmarks.landmark, resized_image.shape, height_cm)
        features['torso_length'] = torso_leg_balance(cm)
    
    return features

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'full_body_photo' not in request.files or 'wrist_photo' not in request.files:
            flash('Both photo uploads are required.', 'error')
            return redirect(request.url)

        full_body_file = request.files['full_body_photo']
        wrist_file = request.files['wrist_photo']
        
        if full_body_file.filename == '' or wrist_file.filename == '':
            flash('Please select both files.', 'error')
            return redirect(request.url)

        if allowed_file(full_body_file.filename) and allowed_file(wrist_file.filename):
            try:
                height = float(request.form['height'])
                gender = request.form['gender']
                body_shape = request.form.get('body_shape_result')
                
                if not 50 <= height <= 250:
                    flash('Please enter a realistic height (50-250 cm).', 'error')
                    return redirect(request.url)
                if not body_shape:
                    flash('Please provide your body shape.', 'error')
                    return redirect(request.url)

            except (ValueError, KeyError):
                flash('Invalid or missing height, gender, or body shape.', 'error')
                return redirect(request.url)
            
            fb_filename = secure_filename(f"full_{secrets.token_hex(8)}.jpg")
            w_filename = secure_filename(f"wrist_{secrets.token_hex(8)}.jpg")
            fb_path = os.path.join(app.config['UPLOAD_FOLDER'], fb_filename)
            w_path = os.path.join(app.config['UPLOAD_FOLDER'], w_filename)
            
            full_body_file.save(fb_path)
            wrist_file.save(w_path)
            
            features = process_features(fb_path, w_path, height, gender, body_shape)
            recommendations = get_recommendations(features)
            
            return render_template('result.html', 
                                   features=features, 
                                   recommendations=recommendations, 
                                   full_body_image_url=url_for('uploaded_file', filename=fb_filename),
                                   wrist_image_url=url_for('uploaded_file', filename=w_filename))
        else:
            flash('Invalid file type. Please use png, jpg, or jpeg.', 'error')
            return redirect(request.url)
            
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/updated.html')
def updated():
    return render_template('updated.html')

# Add this new route to your app.py file
@app.route('/manual.html')
def body_analyzer_page():
    return render_template('manual.html')

if __name__ == '__main__':
    app.run(debug=True)
