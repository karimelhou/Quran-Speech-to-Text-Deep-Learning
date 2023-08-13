import os
import json
import pandas as pd
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Replace with the path to your audio_list.txt and all_ayat.json files
audio_list_path = "/Users/karim/Documents/Machine Learning Models/Deep Speech Model 1 /Quran_Speech_Dataset/audio_list.txt"
transcriptions_path = "/Users/karim/Documents/Machine Learning Models/Deep Speech Model 1 /Quran_Speech_Dataset/all_ayat.json"

# Load text transcriptions from JSON
def load_transcriptions(transcriptions_path):
    with open(transcriptions_path, 'r', encoding='utf-8') as f:
        transcriptions = json.load(f)
    return transcriptions

# Load list of audio files and parse their paths, durations, and text transcriptions
def load_audio_list(audio_list_path):
    with open(audio_list_path, 'r') as file:
        audio_list = [line.strip() for line in file if line.strip()]
    return audio_list


# Prepare the dataset
def prepare_dataset(audio_list_path, transcriptions_path):
    audio_list = load_audio_list(audio_list_path)
    transcriptions = load_transcriptions(transcriptions_path)

    data = []
    for audio_path in audio_list:
        duration = transcriptions.get(audio_path, 0)
        text = ""  # Initialize an empty text if it's not available
        data.append((audio_path, duration, text))
    
    df = pd.DataFrame(data, columns=['path', 'duration', 'text'])
    return df


# Feature extraction from audio using Mel-frequency cepstral coefficients (MFCCs)
def extract_features(audio_path, sr=16000, n_mfcc=13):
    audio, _ = librosa.load(audio_path, sr=sr)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfccs.T, axis=0)

# Main Machine Learning Program
def main():
    # Prepare the dataset
    df = prepare_dataset(audio_list_path, transcriptions_path)

    # Split the dataset into training and testing sets
    X = df['path']
    y = df['text']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Extract MFCC features from audio files
    X_train_features = np.array([extract_features(audio_path) for audio_path in X_train])
    X_test_features = np.array([extract_features(audio_path) for audio_path in X_test])

    # Vectorize text labels
    vectorizer = CountVectorizer()
    y_train_vec = vectorizer.fit_transform(y_train)
    y_test_vec = vectorizer.transform(y_test)

    # Train an SVM classifier
    svm_classifier = SVC(kernel='linear')
    svm_classifier.fit(X_train_features, y_train_vec)

    # Make predictions on the test set
    y_pred = svm_classifier.predict(X_test_features)

    # Convert the vectorized predictions back to text labels
    y_pred_labels = vectorizer.inverse_transform(y_pred)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, [' '.join(labels) for labels in y_pred_labels])
    print(f"Accuracy: {accuracy:.2f}")

if __name__ == "__main__":
    main()
