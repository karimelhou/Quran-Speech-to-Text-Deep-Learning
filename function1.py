import os
import json
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load the dataset
dataset_path = "/Users/karim/Documents/Machine Learning Models/Deep Speech Model 1 /Quran_Speech_Dataset/audio_data"
all_ayat_path = "/Users/karim/Documents/Machine Learning Models/Deep Speech Model 1 /Quran_Speech_Dataset/all_ayat.json"
transcripts_path = "/Users/karim/Documents/Machine Learning Models/Deep Speech Model 1 /Quran_Speech_Dataset/transcripts.tsv"

with open(all_ayat_path, "r") as ayat_file:
    all_ayat_data = json.load(ayat_file)

# Load transcripts data
transcripts = {}
with open(transcripts_path, "r") as transcripts_file:
    for line in transcripts_file:
        parts = line.strip().split("\t")
        if len(parts) == 3:  # Ensure all parts are available
            audio_path = parts[0]
            transcript = parts[2]
            transcripts[audio_path] = transcript


# Prepare data
audio_list_path = "/Users/karim/Documents/Machine Learning Models/Deep Speech Model 1 /Quran_Speech_Dataset/audio_list.txt"
with open(audio_list_path, "r") as audio_list_file:
    audio_paths = audio_list_file.read().splitlines()

X = []
y = []

for audio_path in audio_paths:
    audio_name = os.path.basename(audio_path)
    surah_ayah = audio_name.split(".")[0]  # Extract surah/aya without extension
    
    if surah_ayah in all_ayat_data["tafsir"]:
        verse_text = all_ayat_data["tafsir"][surah_ayah]["text"]
        if audio_path in transcripts:  # Check if transcript exists for this audio
            transcript = transcripts[audio_path]
            X.append(verse_text)
            y.append(transcript)



# Feature extraction using CountVectorizer
vectorizer = CountVectorizer()
X_vec = vectorizer.fit_transform(X)

# Build a basic model (Naive Bayes)
model = MultinomialNB()
model.fit(X_vec, y)

# Example prediction
example_audio_path = "audio_data/AbdulSamad/001001.mp3"
example_surah_ayah = os.path.basename(example_audio_path).split(".")[0]
example_verse_text = all_ayat_data["tafsir"][example_surah_ayah]["text"]
example_X = vectorizer.transform([example_verse_text])
example_prediction = model.predict(example_X)

print(f"Predicted transcript: {example_prediction[0]}")
