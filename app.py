import os
import shutil
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from pathlib import Path

BASE_DIR = os.getcwd() + "/dataset"  # Contains 'dog' and 'cat' subfolders
WORKING_DIR = os.getcwd() + "/working_dir"  # Directory for split data
RESULTS_DIR = os.getcwd() + "/results"  # Directory to store predictions
IMG_SIZE = (150, 150)  # Resize images to this size

def predict_and_move_image(classifier, img_path, results_dir, img_size):
    # Ensure the results directory exists
    results_path = Path(results_dir)
    for folder in ["cat", "dog", "not_cat_dog"]:
        (results_path / folder).mkdir(parents=True, exist_ok=True)

    try:
        # Load and preprocess the image
        img = Image.open(img_path).convert("L").resize(img_size)
        img_array = np.array(img).flatten().reshape(1, -1)

        # Predict the class
        prediction = classifier.predict(img_array)[0]
        if prediction == 0:
            target_folder = results_path / "cat"
            print("It's a Cat!")
        elif prediction == 1:
            target_folder = results_path / "dog"
            print("It's a Dog!")
        else:
            target_folder = results_path / "not_cat_dog"
            print("Not a Cat or Dog!")

        # Move the image
        img_name = Path(img_path).name
        shutil.move(img_path, target_folder / img_name)
        print(f"Moved {img_name} to {target_folder}")

    except Exception as e:
        print(f"Error processing image {img_path}: {e}")

print("Start predicting...")
from joblib import load
loaded_classifier = load("model.joblib")
images = [os.getcwd() + "/test/" + image for image in os.listdir("test")]
for image in images:
    predict_and_move_image(loaded_classifier, image, RESULTS_DIR, IMG_SIZE)