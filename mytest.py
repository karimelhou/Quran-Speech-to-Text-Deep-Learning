import os
import json
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from pydub import AudioSegment
from pydub.playback import play


# loading data
audio_path_list = "/Users/karim/Documents/Machine Learning Models/Deep Speech Model 1 /Quran_Speech_Dataset/audio_list.txt"
transcript_file = "/Users/karim/Documents/Machine Learning Models/Deep Speech Model 1 /Quran_Speech_Dataset/transcripts.tsv"
all_ayat_file   = "/Users/karim/Documents/Machine Learning Models/Deep Speech Model 1 /Quran_Speech_Dataset/all_ayat.json"
dataset_path    = "/Users/karim/Documents/Machine Learning Models/Deep Speech Model 1 /Quran_Speech_Dataset/audio_data"


with open(all_ayat_file, "r") as file:
    for line_number, line in enumerate(file, start=1):
        print(line.strip())
        if line_number == 10:
            break