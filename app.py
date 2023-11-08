import base64
import json
import random
from PIL import Image
from flask_cors import CORS
from flask import Flask, request, jsonify, render_template
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import cv2
import numpy as np
import os
import wave
import threading

import Rec as rec

app = Flask(__name__)
CORS(app)

from transformers import pipeline

corrector = pipeline(
    "text2text-generation",
    "pszemraj/grammar-synthesis-small",
)


processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")


model.to("cpu")

print("Model and Processor Ready !!!")

import pyaudio

# Global variables to manage recording
audio = pyaudio.PyAudio()
frames = []
recording = False


def record_audio():
    global frames, recording, recording_thread

    def _cleanup():
        # Function to stop and cleanup the audio recording

        audio.terminate()

        # Reset recording-related variables
        frames.clear()
        recording = False

    stream = audio.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=44100,
        input=True,
        frames_per_buffer=1024,
    )

    recording_thread = threading.current_thread()

    try:
        frames.clear()
        recording = True
        while recording:
            data = stream.read(1024)
            frames.append(data)
    except Exception as e:
        print(f"Error in recording audio: {e}")
    finally:
        stream.stop_stream()
        stream.close()
        _cleanup()



import librosa
import joblib
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import numpy as np


# Define a function to extract audio features (similar to the preprocessing code)
def extract_features(audio_file):
    audio, sr = librosa.load(audio_file)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1)
    mfccs_std = np.std(mfccs, axis=1)
    return np.hstack((mfccs_mean, mfccs_std))


# Define a function to classify audio using a pre-trained model
def classify_audio(audio_file, model_path):
    # Load the pre-trained classifier model
    loaded_model = joblib.load(model_path)

    # Extract features from the input audio
    input_features = extract_features(audio_file)

    # Make predictions using the loaded model
    prediction = loaded_model.predict([input_features])

    return prediction[0]


import io
from google.cloud import speech_v1p1beta1 as speech
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

global_text = ""

# Set the GOOGLE_APPLICATION_CREDENTIALS environment variable
os.environ[
    "GOOGLE_APPLICATION_CREDENTIALS"
] = "speaking/turing-thought-403619-e014fcc0ac89.json"


# Step 1: Convert audio to text using Google Cloud Speech-to-Text API
def transcribe_audio(audio_file_content):
    client = speech.SpeechClient()
    audio = speech.RecognitionAudio(content=audio_file_content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        enable_automatic_punctuation=True,
        audio_channel_count=1,
        language_code="en-US",
    )

    response = client.recognize(config=config, audio=audio)
    if len(response.results) > 0:
        transcribed_text = response.results[0].alternatives[0].transcript
        return transcribed_text
    else:
        return None


def audio_to_text_and_similarity(audio_file_path, provided_text):
    # Read the audio file using io and convert it to bytes
    with io.open(audio_file_path, "rb") as audio_file:
        audio_file_content = audio_file.read()

    transcribed_text = transcribe_audio(audio_file_content)

    print(transcribed_text)

    if transcribed_text is None:
        return 0.0  # Return a similarity score of 0 if audio transcription fails

    # Step 2: Calculate the similarity score between the transcribed text and provided text
    text_list = [provided_text, transcribed_text]

    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(text_list)

    similarity_score = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]

    return similarity_score


def generate_short_qa(text):
    # Load the question-answering pipeline
    qa_pipeline = pipeline(
        "question-answering",
        model="bert-large-uncased-whole-word-masking-finetuned-squad",
        tokenizer="bert-large-uncased-whole-word-masking-finetuned-squad",
    )

    # Split the text into sentences
    sentences = text.split(". ")

    # Generate questions and short answers
    qa_pairs = []

    for sentence in sentences:
        question = "What is " + sentence + "?"
        answer = qa_pipeline(question=question, context=text)["answer"]
        qa_pairs.append((question, answer))

    return qa_pairs


def inference_ocr(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(image, return_tensors="pt", padding=True)
    inputs = inputs.to("cpu")
    outputs = model.generate(**inputs, max_length=128)
    return processor.tokenizer.decode(outputs[0], skip_special_tokens=True)


def preprocess():
    # Load the image
    image = cv2.imread("uploads/image.jpg")
    # image = cv2.imread('data/test2.jpg')

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to binarize the image
    _, thresholded = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)
    cv2.imwrite("thresholded.png", thresholded)

    # Set the threshold value
    threshold = 200

    # Get image dimensions
    height, width = thresholded.shape

    # Initialize variables to keep track of segment start and end
    segment_start = None
    segment_end = None

    # Create a list to store the segments
    segments = []

    for row in range(height):
        row_mean = np.mean(thresholded[row, :])

        # Check if the row mean is greater than the threshold
        if row_mean < threshold and segment_start is None:
            segment_start = row
        elif row_mean >= threshold and segment_start is not None:
            segment_end = row
            if segment_end - segment_start > 0:
                segment = thresholded[segment_start - 20 : segment_end + 20, :]
                segments.append(segment)
            segment_start = None

    # Save each segment to a separate file
    for i, segment in enumerate(segments):
        segment_filename = f"segments/segment_{i}.png"
        cv2.imwrite(segment_filename, segment)

    print(f"Found {len(segments)} segments.")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/listening", methods=["POST", "GET"])
def listening():
    questions_and_answers = []
    if request.method == "POST":
        audio_file = request.files["audio"]
        audio_file.save("listening/data/user_audio.wav")

        # Read the audio file using io and convert it to bytes
        with io.open("listening/data/user_audio.wav", "rb") as audio_file:
            audio_file_content = audio_file.read()

        transcribed_text = transcribe_audio(audio_file_content)
        # Simulate a list of questions and answers (replace with your data)

        print(transcribed_text)

        qa_pairs = generate_short_qa(transcribed_text)

        for question, answer in qa_pairs:
            questions_and_answers.append({"question": question, "answer": answer})

        json_data = json.dumps(questions_and_answers)
        return render_template("listening.html", data=json_data)
    json_data = json.dumps(questions_and_answers)
    return render_template("listening.html", data=json_data)


@app.route("/start_recording", methods=["POST"])
def start_recording():
    global recording
    if not recording:
        recording_thread = threading.Thread(target=record_audio)
        recording_thread.start()
    return jsonify({"status": "Recording started"})


@app.route("/stop_recording", methods=["POST"])
def stop_recording():
    global recording
    recording = False  # Signal the recording thread to stop

    # Wait for the recording thread to complete
    if recording_thread is not None:
        recording_thread.join()

    with wave.open("recorded_audio.wav", "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(44100)
        wf.writeframes(b"".join(frames))

    # Example usage of the function
    audio_file_path = "recorded_audio.wav"
    model_path = "speaking/audio_classifier_model.pkl"

    result = classify_audio(audio_file_path, model_path)

    global global_text

    similarity_score = audio_to_text_and_similarity(audio_file_path, global_text)

    # Convert the scores to integers
    if int(result) == 0:
        fluency_score = "Weak"
    elif int(result) == 1:
        fluency_score = "Good"
    else:
        fluency_score = "Better"
    accuracy_score = int(similarity_score * 100)

    print(fluency_score, accuracy_score)

    return jsonify(
        {
            "status": "Recording stopped and saved to recorded_audio.wav",
            "fluency_score": fluency_score,
            "accuracy_score": accuracy_score,
        }
    )


# @app.route("/speaking")
# def speaking():
#     text_files = os.listdir("speaking/text_files")
#     random_file = random.choice(text_files)
#
#     global global_text
#
#     with open(
#         os.path.join("speaking/text_files", random_file), "r", encoding="utf-8"
#     ) as file:
#         global_text = file.read()
#     return render_template("speaking.html", text=global_text)

@app.route("/speaking", methods=["GET", "POST"])
def speaking():
    global global_text
    if request.method == "POST":
        print('post')
        rec.record_and_save_audio()

        # Example usage of the function
        audio_file_path = "recorded_audio.wav"
        model_path = "speaking/audio_classifier_model.pkl"

        result = classify_audio(audio_file_path, model_path)



        similarity_score = audio_to_text_and_similarity(audio_file_path, global_text)

        # Convert the scores to integers
        if int(result) == 0:
            fluency_score = "Weak"
        elif int(result) == 1:
            fluency_score = "Good"
        else:
            fluency_score = "Better"
        accuracy_score = int(similarity_score * 100)

        print(fluency_score, accuracy_score)

        return jsonify(
            {
                "status": "Recording stopped and saved to recorded_audio.wav",
                "fluency_score": fluency_score,
                "accuracy_score": accuracy_score,
            }
        )

    else:
        text_files = os.listdir("speaking/text_files")
        random_file = random.choice(text_files)


        with open(
            os.path.join("speaking/text_files", random_file), "r", encoding="utf-8"
        ) as file:
            global_text = file.read()
        return render_template("speaking.html", text=global_text)


@app.route("/writing", methods=["GET", "POST"])
def writing():
    if request.method == "POST":
        # image = request.files["image"]
        # image.save("uploads/image.jpg")
        data = request.get_json()
        image_data = data['image']
        with open('uploads/image.jpg', 'wb') as f:
            f.write(base64.b64decode(image_data.split(',')[1]))

        print('post called, image step1')
        preprocess()

        files = os.listdir("segments")

        output_text = ""
        # Loop through the files and delete each one
        for file in files:
            file_path = os.path.join("segments", file)
            output_text = output_text + " " + inference_ocr(file_path)

        files = os.listdir("segments")

        # Loop through the files and delete each one
        for file in files:
            file_path = os.path.join("segments", file)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
            except Exception as e:
                print(f"Failed to delete: {file_path} - {e}")

        corrected_output_text = corrector(output_text)
        return jsonify({'text': output_text, 'grammarly_corrected_text': corrected_output_text[0]["generated_text"]})

        # return render_template(
        #     "writing_output.html",
        #     text=output_text,
        #     grammarly_corrected_text=corrected_output_text[0]["generated_text"],
        # )

    return render_template("writing.html")


@app.route("/ocr", methods=["POST"])
def ocr():
    if request.method == "POST":
        image = request.files["image"]
        image.save("uploads/image.jpg")

        preprocess()

        files = os.listdir("segments")

        output_text = ""
        # Loop through the files and delete each one
        for file in files:
            file_path = os.path.join("segments", file)
            output_text = output_text + " " + inference_ocr(file_path)

        files = os.listdir("segments")

        # Loop through the files and delete each one
        for file in files:
            file_path = os.path.join("segments", file)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
            except Exception as e:
                print(f"Failed to delete: {file_path} - {e}")

        corrected_output_text = corrector(output_text)

        return jsonify(
            {
                "text": output_text,
                "grammarly_corrected_text": corrected_output_text[0]["generated_text"],
            }
        )


if __name__ == "__main__":
    app.run(debug=True)
