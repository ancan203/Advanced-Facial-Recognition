import argparse
import numpy as np
from deepface import DeepFace
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle


# Load embeddings and labels
embeddings = np.load("embeddings/embeddings.npy")
labels = np.load("embeddings/labels.npy")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.2, random_state=42)

# Train a classifier
classifier = LogisticRegression(max_iter=1000)
classifier.fit(X_train, y_train)

# Evaluate the model
y_pred = classifier.predict(X_test)
# print(classification_report(y_test, y_pred))

# Save the classifier model
with open("emotion_classifier.pkl", "wb") as f:
    pickle.dump(classifier, f)

# Function to predict emotion from an image
def predict_emotion(image_path, class_mapping):
    try:
        # Extract embedding using DeepFace
        embedding = DeepFace.represent(img_path=image_path, model_name="VGG-Face", enforce_detection=False)[0][
            "embedding"]

        # Predict emotion using the classifier
        probabilities = classifier.predict_proba([embedding])[0]  # Get probabilities for each class
        class_idx = classifier.predict([embedding])[0]
        class_name = class_mapping[class_idx]

        confidence_scores = {
            class_mapping[idx]: prob * 100 for idx, prob in enumerate(probabilities)
        }

        return class_name, confidence_scores
    except Exception as e:
        print(f"Error: {e}")
        return None

# Main function to parse arguments and run the prediction
def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Emotion Detection using DeepFace")
    parser.add_argument("image_path", type=str, help="Path to the image file")

    # Parse the command-line arguments
    args = parser.parse_args()

    # Test on a new image
    image_path = args.image_path
    class_mapping = {0: "Angry", 1: "Background", 2: "Happy", 3: "Sad"}
    predicted_emotion, confidence_scores = predict_emotion(image_path, class_mapping)

    if predicted_emotion:
        print(f"Predicted Emotion: {predicted_emotion}")
        for emotion, confidence in confidence_scores.items():
            print(f"{emotion}: {confidence:.2f}%")

    else:
        print("Could not predict emotion.")


if __name__ == "__main__":
    main()
