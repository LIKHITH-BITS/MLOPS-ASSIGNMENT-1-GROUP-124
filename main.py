import onnxruntime as ort
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from joblib import dump, load
from pathlib import Path
from PIL import Image
import shutil

import requests

url = "https://github.com/onnx/models/raw/main/vision/classification/resnet/model/resnet50-v1-12.onnx"
output_path = "resnet50.onnx"

print("Downloading ResNet-50 ONNX model locally...")
response = requests.get(url, stream=True)
if response.status_code == 200:
    with open(output_path, "wb") as f:
        f.write(response.content)
    print(f"Model downloaded and saved as {output_path}")
else:
    print("Failed to download the model. Check the URL or your internet connection.")


# Define constants
IMG_SIZE = (224, 224)
RESULTS_DIR = "results"
ONNX_MODEL_PATH = "resnet50.onnx"

# Load ONNX model
def load_onnx_model(model_path):
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    return session, input_name

# Preprocess a single image
def preprocess_image(img_path, img_size):
    img = Image.open(img_path).resize(img_size).convert("RGB")
    img_array = np.array(img, dtype=np.float32).transpose(2, 0, 1)  # Channels first
    img_array = img_array / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Extract features for a single image
def extract_features(session, input_name, img_array):
    outputs = session.run(None, {input_name: img_array})
    return outputs[0].flatten()  # Flatten the feature vector

# Extract features for the entire dataset
def extract_features_for_dataset(session, input_name, img_paths, img_size):
    features = []
    for img_path in img_paths:
        img_array = preprocess_image(img_path, img_size)
        features.append(extract_features(session, input_name, img_array))
    return np.array(features)

# Train a Scikit-learn classifier
def train_classifier(train_features, train_labels):
    classifier = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42)
    classifier.fit(train_features, train_labels)
    print("Model training completed.")
    return classifier

# Save the trained model
def save_model(classifier, filename="dog_cat_classifier.joblib"):
    dump(classifier, filename)
    print(f"Model saved as {filename}")

# Load a saved model
def load_saved_model(filename="dog_cat_classifier.joblib"):
    classifier = load(filename)
    print(f"Model loaded from {filename}")
    return classifier

# Predict and move an image to the appropriate folder
def predict_and_move_image(classifier, session, input_name, img_path, results_dir, img_size):
    results_path = Path(results_dir)
    for folder in ["cat", "dog", "not_cat_dog"]:
        (results_path / folder).mkdir(parents=True, exist_ok=True)

    try:
        # Preprocess and extract features
        img_array = preprocess_image(img_path, img_size)
        features = extract_features(session, input_name, img_array).reshape(1, -1)

        # Predict class
        prediction = classifier.predict(features)[0]
        if prediction == 0:
            target_folder = results_path / "cat"
            print(f"{img_path} is a Cat.")
        elif prediction == 1:
            target_folder = results_path / "dog"
            print(f"{img_path} is a Dog.")
        else:
            target_folder = results_path / "not_cat_dog"
            print(f"{img_path} is not a Cat or Dog.")

        # Move image
        img_name = Path(img_path).name
        shutil.move(img_path, target_folder / img_name)
        print(f"Moved {img_name} to {target_folder}")

    except Exception as e:
        print(f"Error processing image {img_path}: {e}")

# Main workflow
if __name__ == "__main__":
    # Paths
    cat_dir = "path/to/cat"
    dog_dir = "path/to/dog"
    onnx_model_path = ONNX_MODEL_PATH

    # Load image paths and labels
    cat_images = list(Path(cat_dir).glob("*.jpg"))
    dog_images = list(Path(dog_dir).glob("*.jpg"))
    all_images = cat_images + dog_images
    labels = [0] * len(cat_images) + [1] * len(dog_images)  # 0 for cats, 1 for dogs

    # Split dataset
    train_imgs, test_imgs, train_labels, test_labels = train_test_split(all_images, labels, test_size=0.2, random_state=42)

    # Load ONNX model
    session, input_name = load_onnx_model(onnx_model_path)

    # Extract features for training and testing
    print("Extracting features for training...")
    train_features = extract_features_for_dataset(session, input_name, train_imgs, IMG_SIZE)
    print("Extracting features for testing...")
    test_features = extract_features_for_dataset(session, input_name, test_imgs, IMG_SIZE)

    # Train the classifier
    classifier = train_classifier(train_features, train_labels)

    # Save the model
    save_model(classifier)

    # Evaluate the model
    print("Evaluating model...")
    predictions = classifier.predict(test_features)
    print(classification_report(test_labels, predictions))
    print(f"Accuracy: {accuracy_score(test_labels, predictions)}")

    # Test single image prediction and move
    sample_img_path = "path/to/sample_image.jpg"
    predict_and_move_image(classifier, session, input_name, sample_img_path, RESULTS_DIR, IMG_SIZE)
