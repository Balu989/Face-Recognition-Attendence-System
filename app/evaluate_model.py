import cv2
import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import random

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()


def train_model_and_split():
    train_faces, train_labels = [], []
    test_faces, y_true = [], []
    label_map = {}
    label_id = 0

    for branch_folder in os.listdir("register_candidates"):
        branch_path = os.path.join("register_candidates", branch_folder)

        if not os.path.isdir(branch_path):
            continue

        print(f"📁 Found branch folder: {branch_path}")

        for student_folder in os.listdir(branch_path):
            student_path = os.path.join(branch_path, student_folder)
            if not os.path.isdir(student_path):
                continue

            print(f"📁 Loading from: {student_path}")
            label_map[label_id] = student_folder

            images = []
            for img_file in os.listdir(student_path):
                img_path = os.path.join(student_path, img_file)
                if img_path.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        images.append(img)
                    else:
                        print(f"⚠️ Could not read: {img_path}")
                else:
                    print(f"🚫 Skipped non-image file: {img_path}")

            if not images:
                print(f"❌ No valid images in: {student_path}")
                continue

            random.shuffle(images)
            split_index = int(len(images) * 0.8)
            train_imgs = images[:split_index]
            test_imgs = images[split_index:]

            for img in train_imgs:
                train_faces.append(img)
                train_labels.append(label_id)

            for img in test_imgs:
                test_faces.append((img, student_folder))
                y_true.append(student_folder)

            label_id += 1

    if not train_faces:
        print("❌ No valid training data.")
        exit(1)

    print(f"\n✅ Training with {len(train_faces)} images across {label_id} students.")
    recognizer.train(train_faces, np.array(train_labels))

    return label_map, test_faces, y_true


def evaluate_model(label_map, test_faces, y_true):
    y_pred = []

    for img, true_label in test_faces:
        img = cv2.resize(img, (200, 200))
        label, confidence = recognizer.predict(img)
        predicted_name = label_map[label] if confidence < 60 else "Incorrect"
        print(f"[✔] Actual: {true_label}, Predicted: {predicted_name}, Confidence: {confidence:.2f}")
        y_pred.append(predicted_name)

    return y_true, y_pred


def plot_results(y_true, y_pred):
    if not y_true or not y_pred:
        print("❌ No predictions to evaluate.")
        return

    print("\n📋 Classification Report:\n")
    report = classification_report(y_true, y_pred, zero_division=0)
    print(report)

    with open("model_evaluation.txt", "w") as file:
        file.write("Classification Report:\n")
        file.write(report + "\n")
        acc = accuracy_score(y_true, y_pred)
        file.write(f"Accuracy Score: {acc:.4f}\n")

    labels = sorted(list(set(y_true + y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.show()

    print("📁 Saved:")
    print(" → model_evaluation.txt")
    print(" → confusion_matrix.png")


if __name__ == "__main__":
    label_map, test_faces, y_true = train_model_and_split()
    y_true, y_pred = evaluate_model(label_map, test_faces, y_true)
    plot_results(y_true, y_pred)
