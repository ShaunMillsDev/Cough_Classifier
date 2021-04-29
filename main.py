import os
import csv
import librosa
import librosa.display
import numpy as np
import pyaudio
import wave
import time

import tkinter as tk
from tkinter.messagebox import showinfo
from tkinter import *

# import matplotlib.pyplot as plt
# import IPython.display as ipd
# import soundfile as sf

import random
import pandas as pd

from sklearn import metrics
from sklearn import preprocessing
from sklearn.svm import SVC

# from sklearn import neighbors, datasets
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import accuracy_score
# from sklearn.metrics import mean_squared_error, r2_score

train_list = []
test_list = []

# create initial files
def createCSV(header, mfcc_count, label, file_location):
    header = header
    for i in range(1, mfcc_count + 1):
        header += f' mfcc{i}'
    if label:
        header += ' label'
    header = header.split()

    file = open(file_location, 'w', newline='')
    with file:
        writer = csv.writer(file)
        writer.writerow(header)

# iterate through directory, extract mfccs and labels, save to dataset_location
def createInitialSet(directory, trim, threshold, dataset_location, is_cough, include_filename):
    i = 0
    if len(os.listdir(directory)) == 0:
        print(directory, ' is empty')
    else:
        for file in os.listdir(directory):
            audio_path = directory + '/' + file

            if include_filename:
                to_append = file
            else:
                to_append = ''

            # load file
            x, sr = librosa.load(audio_path)

            if trim:
                x_trim, index = librosa.effects.trim(x, threshold)

            # Generate MFCCs
            if trim:
                m = librosa.feature.mfcc(x_trim, n_mfcc=13, sr=sr)
                dm = librosa.feature.delta(m)
                mfccs = librosa.feature.delta(m, order=2)
            else:
                m = librosa.feature.mfcc(x, n_mfcc=13, sr=sr)
                dm = librosa.feature.delta(m)
                mfccs = librosa.feature.delta(m, order=2)

            # create row with label
            for e in m:
                to_append += f' {np.mean(e)}'
            for e in dm:
                to_append += f' {np.mean(e)}'
            for e in mfccs:
                to_append += f' {np.mean(e)}'

            if is_cough == 1:
                to_append += ' 1'
            elif is_cough == 2:
                to_append += ' 0'

            # write to csv dataset file
            dataset = open(dataset_location, 'a', newline='')
            with dataset:
                writer = csv.writer(dataset)
                writer.writerow(to_append.split())
            i += 1
            print('Features successfully extracted from: ', audio_path)

# store a set (the_list) into a csv file (filename)
def storeSet(filename, the_list, num_samples, mfcc_count):
    print('\n\n', filename, ': ', len(the_list), ' recordings -->\n')

    to_append = ''

    for x in range(num_samples):
        for y in range(1, mfcc_count+2):
            to_append += f' {the_list[x].iloc[y]}'

        print(the_list[x].iloc[0])

        set = open(filename, 'a', newline='')
        with set:
            writer = csv.writer(set)
            writer.writerow(to_append.split())

        to_append = ''

# train on the training and test set, output accuracy
def train(mfcc_count, training_name, testing_name, label_known):
    results = []
    svc = SVC(kernel='linear')

    # load the training and test sets
    datasets_train = pd.read_csv(training_name)
    datasets_test = pd.read_csv(testing_name)

    train_mfccs = datasets_train.iloc[:, 0:mfcc_count]
    train_labels = datasets_train.iloc[:, -1]

    test_mfccs = datasets_test.iloc[:, 0:mfcc_count]
    if label_known:
        test_labels = datasets_test.iloc[:, -1]

    # preprocessing by scaling
    scaler = preprocessing.StandardScaler().fit(train_mfccs)
    train_mfccs_scaled = scaler.transform(train_mfccs)
    test_mfccs_scaled = scaler.transform(test_mfccs)

    # from sklearn.feature_selection import RFE
    # rfe = RFE(estimator=svc, n_features_to_select=1, step=1)

    # Fit the model
    svc.fit(train_mfccs_scaled, train_labels)

    # Predict
    prediction = svc.predict(test_mfccs_scaled)

    print('\n', svc.n_support_)

    if label_known:
        print('\nPrediction vs Actual Label')
        print('--------------------------')

        for i in range(len(test_labels)):
            result = 'Prediction: ' + str(prediction[i]) + ' : Actual label: ' + str(test_labels.iloc[i])
            results.append(result)

        # Model Accuracy: how often is the classifier correct?
        result = "Accuracy: " + str(metrics.accuracy_score(test_labels, prediction) * 100) + '%'
        results.append(result)
    else:
        if int(prediction[0]) == 0:
            results.append('Prediction: NOT Cough')
        elif int(prediction[0]) == 1:
            results.append('Prediction: Cough')

    return results

# combine cough_test_sounds and non_cough_test_sounds into one testing_set.csv file
def combine_test_data(label_known):
    if not label_known:
        createInitialSet('Sound_Folders/Test_Sounds/Test_Sound',
                         True, 30, 'Datasets/testing_set.csv', 3, False)
    else:
        createInitialSet('Sound_Folders/Test_Sounds/Cough_Test_Sounds',
                         True, 30, 'Datasets/testing_set.csv', 1, False)
        createInitialSet('Sound_Folders/Test_Sounds/Non_Cough_Test_Sounds',
                         False, 0, 'Datasets/testing_set.csv', 2, False)

# test model on new test data from Test_Sounds folders
def test_new_data(file_path, label_known, mfcc_count):
    # test new data when model is already trained
    if label_known:
        createCSV('', mfcc_count, True, 'Datasets/testing_set.csv')
    else:
        createCSV('', mfcc_count, False, 'Datasets/testing_set.csv')

    combine_test_data(label_known)
    results = train(mfcc_count, 'Datasets/training_set.csv', 'Datasets/testing_set.csv', label_known)

    if not label_known:
        if os.path.exists(file_path):
            os.remove(file_path)

    return results

# record audio
def record_audio():
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 22050
    RECORD_SECONDS = 3

    WAVE_OUTPUT_FILENAME = r"Sound_Folders/Test_Sounds/Test_Sound/output" + str(time.time()) + ".wav"

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("\n-- recording for 3 seconds --")

    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("-- done recording --\n")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    return WAVE_OUTPUT_FILENAME

# randomise the data, create train_list & test_list
def randomiseOrder(dataset_location, num_train_samples):
    dataset = pd.read_csv(dataset_location)

    # randomise the data and create training set
    while len(train_list) < num_train_samples:
        random_index = random.randint(0, len(dataset) - 1)
        random_sample = dataset.iloc[random_index]
        train_list.append(random_sample)

        i = dataset[(dataset.filename == random_sample.iloc[0])].index
        dataset.drop(i, inplace=True)
        dataset.reset_index(drop=True, inplace=True)

    # set remaining data to test set
    current_index = 0
    while current_index < len(dataset):
        sample = dataset.iloc[current_index]
        test_list.append(sample)
        current_index += 1

# train model and test
def train_and_test(mfcc_count):
    # create CSV files for dataset, training set, testing set (boolean label)
    createCSV('filename', mfcc_count, True, 'Datasets/dataset.csv')
    createCSV('', mfcc_count, True, 'Datasets/training_set.csv')
    createCSV('', mfcc_count, True, 'Datasets/testing_set.csv')

    # create mfcc sets (boolean trimming)
    createInitialSet('Sound_Folders/Cough_Recordings', True, 30, 'Datasets/dataset.csv', 1, True)
    createInitialSet('Sound_Folders/Training_Sounds', False, 0, 'Datasets/dataset.csv', 2, True)

    # randomise the data order
    randomiseOrder('Datasets/dataset.csv', 112)

    # store the randomised and categorised sets back as separate csv files
    storeSet('Datasets/training_set.csv', train_list, 112, mfcc_count)
    storeSet('Datasets/testing_set.csv', test_list, 28, mfcc_count)

    # train
    train(mfcc_count, 'Datasets/training_set.csv', 'Datasets/testing_set.csv', True)

# GUI
root = Tk()

def recordAudio():
    path = record_audio()
    results = test_new_data(path, False, 39)
    showinfo("Window", results[0])

def runTest():
    results = test_new_data('', True, 39)
    results_string = ''

    for e in results:
        results_string += str(e) + '\n'
    showinfo("Results", results_string)

def trainData():
    train_and_test(39)

trainData = Button(root, text="Train the Classifier", padx=94, pady=20, command=trainData)
trainData.grid(row=0, column=0)
padLabel = Label(text=" ")
padLabel.grid(row=1, column=0)

runTest = Button(root, text="Run Test", padx=120, pady=20, command=runTest)
runTest.grid(row=2, column=0)
padLabel = Label(text=" ")
padLabel.grid(row=3, column=0)

recordAudio = Button(root, text="Record 3 Seconds of Audio", padx=71, pady=20, command=recordAudio)
recordAudio.grid(row=4, column=0)
# padLabel = Label(text=" ")
# padLabel.grid(row=5, column=0)

root.mainloop()
