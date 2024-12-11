from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash
import os
import joblib
import pandas as pd
import json
from ultralytics import YOLO
import cv2
import glob
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Change to a strong secret key

# Load models
model_rf = joblib.load('car_price_model.pkl')
infer = YOLO('best_damage_detection_model.pt')

# Load and preprocess dataset
df = pd.read_csv('Car_price.csv')
df.drop(columns=['name'], inplace=True)
categorical_cols = ['fuel', 'seller_type', 'transmission', 'owner']
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
df['mileage'] = df['mileage'].str.extract('(\d+\.\d+|\d+)').astype(float)
df['engine'] = df['engine'].str.extract('(\d+)').astype(float)
df['max_power'] = df['max_power'].str.extract('(\d+\.\d+|\d+)').astype(float)
df['torque'] = df['torque'].str.extract('(\d+\.\d+|\d+)').astype(float)
df.fillna(df.median(), inplace=True)
X = df.drop(columns=['selling_price'])

# JSON-based database for user data
users_db = 'users.json'

def load_users():
    if os.path.exists(users_db):
        with open(users_db, 'r') as file:
            return json.load(file)
    return {}

def save_user(username, password):
    users = load_users()
    users[username] = password
    with open(users_db, 'w') as file:
        json.dump(users, file)
        
#-------------------------------------------------------------------------------------------
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

def create_annotated_image(image_path, save_path):
    """
    Annotates the input image with bounding boxes for detected damages and labels.

    Args:
    - image_path (str): Path to the input image.
    - save_path (str): Path to save the annotated image.
    - results (list): List of results from the YOLO model containing detected damages.
    """
    # Open the image file
    infer = YOLO("best_damage_detection_model.pt")
    
    # Run inference and specify save=True to save annotated images in save_dir
    results = infer.predict(image_path, save=True)

    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    # Load a font (optional, for displaying labels)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        font = ImageFont.load_default()

    # Iterate over the results from YOLO (bounding boxes and class labels)
    for i, box in enumerate(results[0].boxes):
        # Get the coordinates of the bounding box
        x1, y1, x2, y2 = box.xyxy[0]  # Get the coordinates of the box (xmin, ymin, xmax, ymax)
        label = infer.names[int(box.cls[0].item())]  # Get the label name for the damage

        # Draw the rectangle around the damage
        draw.rectangle(((x1, y1), (x2, y2)), outline="red", width=3)

        # Draw the label text
        draw.text((x1, y1 - 20), label, fill="red", font=font)

    # Save the annotated image
    image.save(save_path)

    # Optional: Display the annotated image
    # plt.imshow(image)
    # plt.axis('off')
    # plt.show()
 
#-------------------------------------------------------------------------------------------

# def create_annotated_image(image_path, save_path):
#     # Open the image file
#     image = Image.open(image_path)
#     draw = ImageDraw.Draw(image)
    
#     # Example: Draw a simple rectangle (you can customize this with damage detection logic)
#     draw.rectangle(((50, 50), (150, 150)), outline="red", width=5)

#     # Save the annotated image
#     image.save(save_path)


def detect_damages(image_path, save_dir="runs/detect"):
    """
    Detect damages from the car image using a trained YOLO model and display the annotated image.
    
    Args:
    - image_path (str): Path to the image file.
    - save_dir (str): Base directory where YOLO saves annotated images.

    Returns:
    - damage_count (int): Number of detected damages.
    """
    
      # Define price reduction factors for each damage type
    damage_costs = {
        'Bodypanel-Dent': 8000,
        'Front-Windscreen-Damage': 15000,
        'Headlight-Damage': 1200,
        'Rear-windscreen-Damage': 10000,
        'RunningBoard-Dent': 6000,
        'Sidemirror-Damage': 500,
        'Signlight-Damage': 400,
        'Taillight-Damage': 700,
        'bonnet-dent': 14000,
        'boot-dent': 9000,
        'doorouter-dent': 5000,
        'fender-dent': 8000,
        'front-bumper-dent': 13000,
        'pillar-dent': 7000,
        'quaterpanel-dent': 1200,
        'rear-bumper-dent':15000,
        'roof-dent': 5000
    }
    # Load the pre-trained YOLO model
    infer = YOLO("best_damage_detection_model.pt")
    
    # Run inference and specify save=True to save annotated images in save_dir
    results = infer.predict(image_path, save=True)
    
    # Check if any directories exist in the save_dir
    save_subdirs = glob.glob(os.path.join(save_dir, '*/'))
    if not save_subdirs:
        print("No directories found in the save directory. No annotated images were saved.")
        return 0  # Return 0 damages if no save directories are found
    
    # Get the latest created subdirectory in the save_dir
    latest_run_dir = max(save_subdirs, key=os.path.getmtime)
    
    # Get the saved annotated image path
    save_path = os.path.join(latest_run_dir, os.path.basename(image_path))
    damage_list = []

    # Iterate over each detection in the results
    for box in results[0].boxes:
        # Access class ID and confidence score
        class_id = int(box.cls)
        damage_name = results[0].names[class_id]  # Get damage name from class ID
        damage_list.append(damage_name)
        

  # Calculate total damage cost
    total_damage_cost = sum(damage_costs.get(damage, 500) for damage in damage_list)  # Default to 500 if not found

    # Count the damages (number of bounding boxes)
    damage_count = len(results[0].boxes)
    
    # Load and display the annotated image if it exists
    annotated_image = cv2.imread(save_path)


    
   # print("New",damage_list)
    if annotated_image is not None:
        plt.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()
    else:
        print("Annotated image could not be loaded.")

    return total_damage_cost


def preprocess_and_predict(input_data):
    df_input = pd.DataFrame([input_data])
    df_input = pd.get_dummies(df_input, columns=['fuel', 'seller_type', 'transmission', 'owner'], drop_first=True)
    df_input['mileage'] = df_input['mileage'].str.extract('(\d+\.\d+|\d+)').astype(float)
    df_input['engine'] = df_input['engine'].str.extract('(\d+)').astype(float)
    df_input['max_power'] = df_input['max_power'].str.extract('(\d+\.\d+|\d+)').astype(float)
    df_input['torque'] = df_input['torque'].str.extract('(\d+\.\d+|\d+)').astype(float)
    df_input.fillna(df_input.median(), inplace=True)
    
    missing_cols = set(X.columns) - set(df_input.columns)
    for col in missing_cols:
        df_input[col] = 0
    df_input = df_input[X.columns]
    return model_rf.predict(df_input)[0]

# Routes
@app.route('/')
def index():
    if 'username' in session:
        return render_template('index.html', username=session['username'])
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        users = load_users()
        if username in users:
            flash('Username already exists. Please log in.', 'warning')
            return redirect(url_for('login'))
        save_user(username, password)
        flash('Registration successful. Please log in.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        users = load_users()
        if username in users and users[username] == password:
            session['username'] = username
            flash('Login successful!', 'success')
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password', 'danger')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    flash('Logged out successfully', 'info')
    return redirect(url_for('login'))

import os

@app.route('/predict', methods=['POST'])
def predict():
    if 'username' not in session:
        flash('Please log in to access this feature.', 'warning')
        return redirect(url_for('login'))

    if 'image' not in request.files:
        flash('No image file found', 'danger')
        return redirect(url_for('index'))

    image_file = request.files['image']
    
    # Ensure the directory exists for saving images
    image_dir = 'static/car_images'
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    # Save the uploaded image
    image_path = os.path.join(image_dir, image_file.filename)
    image_file.save(image_path)

    # Define input data from the form
    input_data = {
        'year': request.form['year'],
        'km_driven': request.form['km_driven'],
        'fuel': request.form['fuel'],
        'seller_type': request.form['seller_type'],
        'transmission': request.form['transmission'],
        'owner': request.form['owner'],
        'mileage': request.form['mileage'],
        'engine': request.form['engine'],
        'max_power': request.form['max_power'],
        'torque': request.form['torque'],
        'seats': request.form['seats'],
    }
    
    # Perform damage detection and price prediction
    damage_count = detect_damages(image_path)
    predicted_price = preprocess_and_predict(input_data)
    final_price_estimate = predicted_price - (damage_count)

    # Create the annotated image (example using OpenCV or PIL)
    annotated_image_path = os.path.join('static/annotated_images', image_file.filename)
    create_annotated_image(image_path, annotated_image_path)
    
    # Return the result and render it on the same page
    return render_template('index.html', 
                           predicted_price=final_price_estimate,
                           damage_count=damage_count,
                          # annotated_image_path=f"static/annotated_images/{image_file.filename}")
                           annotated_image_path=f"static/annotated_images/{image_file.filename}")





if __name__ == "__main__":
    app.run(debug=True)