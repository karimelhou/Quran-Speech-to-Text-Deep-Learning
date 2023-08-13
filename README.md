# Quran Speech-to-Text Deep Learning Project

This project aims to develop an end-to-end deep learning model that converts Quranic speech recordings into text, enabling automatic transcription of Quranic recitations. The project involves several steps including data preprocessing, feature extraction, model building, and transcription. The dataset used in this project consists of Quranic audio recordings and corresponding text transcriptions.

## Project Structure

The project is organized into the following main components:

1- Data Collection: The dataset is compiled from open and public data sources freely available online. To obtain the audio dataset, please visit [insert link to data source] and follow the instructions to download the audio files.
2- Data Preprocessing: Once you have obtained the audio dataset, create a folder named audio_data in the root directory of the project. Organize the audio files according to the provided structure. Create a list of audio paths in a text file.
3- Transcriptions: Prepare the transcriptions with the corresponding audio paths and durations. Save this information in a tab-separated .tsv file.
Feature Extraction: The audio features are extracted using the Mel-frequency cepstral coefficients (MFCCs) technique. MFCCs are a common method to represent the spectral characteristics of audio signals in machine learning tasks.
4- Model Building: A deep neural network (RNN-based) is built to perform speech-to-text transcription. The model takes MFCC features as input and learns to predict the corresponding text transcription.
5- Transcription: The trained model is used to transcribe Quranic speech recordings into text. This allows for automatic conversion of audio recitations into readable text.
Getting Started

## Requirements: Make sure you have Python (>=3.6) installed along with the required libraries listed in the requirements.txt file.
Data Preprocessing: Use the provided scripts for data preprocessing. Run prepare_dataset.py to create the training and testing datasets.
Feature Extraction: The extract_features function in deepspecch.py extracts MFCC features from audio files.
Model Training: The RNN-based model is trained using the train_model function in deepspecch.py. Adjust hyperparameters as needed.
Transcription: After training, you can use the trained model to transcribe audio files using the transcribe_audio function.

## Contributing

Contributions and improvements to the project are welcome. If you encounter issues or have ideas for enhancements, feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License.

By following these instructions, you can set up and run the project with your own dataset. Make sure to provide clear instructions for users to obtain the audio data separately.
