import os
import shutil
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from pathlib import Path
from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn.ensemble import GradientBoostingClassifier
from sklearn_extra.cluster import ResNet

PARENT_DIR = os.path.dirname(os.getcwd())
BASE_DIR = os.getcwd() + "/dataset"  # Contains 'dog' and 'cat' subfolders
WORKING_DIR = os.getcwd() + "/working_dir"  # Directory for split data
RESULTS_DIR = os.getcwd() + "/results"  # Directory to store predictions
IMG_SIZE = (150, 150)  # Resize images to this size

def load_and_preprocess_images(base_dir, img_size):
    labels, images = [], []

    for label, folder in enumerate(["Cat", "Dog"]):
        folder_path = Path(base_dir, folder)
        for img_path in folder_path.glob("*.jpg"):
            try:
                img = Image.open(img_path).convert("L")  # Convert to grayscale
                img = img.resize(img_size)
                images.append(np.array(img).flatten())
                labels.append(label)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")

    return np.array(images), np.array(labels)


def split_data(images, labels, split_ratio=0.1):
    return train_test_split(images, labels, test_size=split_ratio, random_state=42)


def prepare_directories(working_dir):
    train_dir = Path(working_dir, "train")
    val_dir = Path(working_dir, "validation")
    for folder in [train_dir, val_dir]:
        folder.mkdir(parents=True, exist_ok=True)
        for sub_folder in ["dog", "cat"]:
            Path(folder, sub_folder).mkdir(parents=True, exist_ok=True)
    return train_dir, val_dir


def extract_resnet_features(images, img_size):
    """Use ResNet to extract features."""
    resnet = ResNet(pretrained=True, input_size=img_size)
    return resnet.fit_transform(images)

def train_classifier(train_images, train_labels):
    train_features = extract_resnet_features(train_images, IMG_SIZE)
    classifier = GradientBoostingClassifier(
        n_estimators=100, 
        learning_rate=0.1, 
        max_depth=6, 
        random_state=42
    )
    classifier.fit(train_features, train_labels)
    print("Model training completed using Gradient Boosting with ResNet features.")
    return classifier


def evaluate_model(classifier, test_images, test_labels):
    predictions = classifier.predict(test_images)
    accuracy = accuracy_score(test_labels, predictions)
    print(f"Accuracy: {accuracy * 100:.2f}%")


if __name__ == "__main__":
    # Step 1: Load and preprocess data
    print("Loading and preprocessing images...")
    images, labels = load_and_preprocess_images(BASE_DIR, IMG_SIZE)

    # Step 2: Split data
    print("Splitting data into training")
    train_images, test_images, train_labels, test_labels = split_data(images, labels)

    # Step 3: Train the classifier
    print("Training the classifier...")
    classifier = train_classifier(train_images, train_labels)

    # Step 4: Evaluate the classifier
    print("Evaluating the classifier...")
    evaluate_model(classifier, test_images, test_labels)

    # Save the model if necessary (optional)
    from joblib import dump
    dump(classifier, "model.joblib")

    os.chdir(PARENT_DIR)
