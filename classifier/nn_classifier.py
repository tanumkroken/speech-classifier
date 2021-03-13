'''The Neural Network classifier'''
#  Copyright (c) 2021 by Ole Christian Astrup. All rights reserved.  Licensed under MIT
#   license.  See LICENSE in the project root for license information.
#

import os
import pandas as pd
import librosa
import glob
import librosa.display
import random

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from keras.utils.np_utils import to_categorical

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import Adam
from keras.utils import np_utils
from sklearn import metrics

from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.model_selection import train_test_split, GridSearchCV

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

from keras.callbacks import EarlyStopping

from keras import regularizers

from sklearn.preprocessing import LabelEncoder

from datetime import datetime
from pathlib import Path
import sys
from pandarallel import pandarallel
from collections import defaultdict
from classifier.utils import init_logger

AUDIO = '.flac'
TRAINING_DATA = '../training_data/'
MALE = TRAINING_DATA + 'male'
FEMALE = TRAINING_DATA + 'female'

''' A Neural Network gender classifier'''


class GenderClassifier:
    ''' The gender classifier.

    Attributes
    ----------
    dataset: `list`
        A list of tuples representing the dataset for the classifier:
        tuples:
        ("label" , "path_to_labeled_dataset_folder","file_extension")

    dimension: `int`
        Number of labels in the dataset. This will be the output form the classifier.

    label: `dict`
        A dictionary of the dataset labels.

    Methods
    -------
    prepare_dataset(load: `bool` = False)
        Prepare the dataset and randomize it. If load = True, load a previously randomized set from file.
    train(training: `int`, validation: `int`, test: `int`)
        train the dataset.
    delete(key: `Any`)
        Delete a node based on the given key from the binary tree.
    get_leftmost(node: `AVLNode`)
        Return the node whose key is the smallest from the given subtree.
    get_rightmost(node: `AVLNode`)
        Return the node whose key is the biggest from the given subtree.
    get_successor(node: `AVLNode`)
        Return the successor node in the in-order order.
    get_predecessor(node: `AVLNode`)
        Return the predecessor node in the in-order order.
    get_height(node: `Optional[AVLNode]`)
        Return the height of the given node.
    '''
    def __init__(self, source_folder: str, number_of_classes: int, labels: list, data_folders: list , audio_formats: list, logger):
        ''' Initialize the class'''
        self.source = source_folder
        self.classes = number_of_classes
        self.labels = {}
        self.logger = logger
        self.formats = audio_formats
        self.count = 0
        index = 0
        for label in labels:
            self.labels[label] = data_folders[index]
            index += 1

    def randomize_daset(self, csv_file)-> pd.DataFrame:
        ''' Organize and randomize the dataset'''
         # Organize the dataset and randomize it
        self.logger.info('Organizing and randomizing the datasets')
        index = 0
        dataframes = []
        for label in self.labels:
            files = []
            folder = self.labels[label]
            file_ext = self.formats[0]
            p = Path(folder)
            file_list = p.glob('*' + file_ext)
            for file in file_list:
                files.append(file.absolute())
            # Create the dataframe for the labeled set
            df = pd.DataFrame(files)
            df = df.rename(columns = {0:'file'})
            # Add the label index
            df['label'] = index
            self.logger.info('Dataset labeled {}: {}, {} files'.format(label,folder,len(files)))
            # Store the df instance
            dataframes.append(df)
            index += 1
        # Join the dataframes
        dataset = pd.concat(dataframes, ignore_index=True)
        # Randomizing our files to be able to split into train, validation and test
        dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)
        # Store the randomized set if we want to reproduce the results later
        dataset.to_csv(csv_file)
        self.logger.info('Randomized dataset written to "{}"'.format(csv_file))
        return dataset

    def load_randomized_dataset(self, csv_file: str) -> pd.DataFrame:
        # Load the randomized dataset from file
        p = Path(csv_file)
        if p.exists() is True:
            dataset = pd.read_csv(csv_file)
            self.logger.info('Randomized dataset loaded from "{}"'.format(csv_file))
        else:
            self.logger.error('No randomized dataset {} to load'.format(csv_file))
            sys.exit()
        return dataset

    def check_label_balance(self,df : pd.DataFrame, bins: list):
        ''' Check the balance of the training, validation and test datasets'''
        # Split the dataset int training, validation & test
        train, validate, test = self.split_data_set(df, bins)
        # Training set
        result = train['label'].value_counts(normalize=True).tolist()
        n = len(train.index)
        self.logger.info('Training set length: {}, Normalized split {}'.format(n,result))
        # Validation set
        n = len(validate.index)
        result = validate['label'].value_counts(normalize=True).tolist()
        self.logger.info('Validation set length: {}, Normalized split {}'.format(n,result))
        # Test set
        n = len(test.index)
        result = test['label'].value_counts(normalize=True).tolist()
        self.logger.info('Test set length: {}, Normalized split {}'.format(n,result))
        # Check
        # print (len(df_train.index)+len(df_validate.index)+len(df_test.index),' == ', size)
        return

    def split_data_set(self, df: pd.DataFrame,bins: list) -> tuple:
        ''' Split the dataset into train, validate and test bins '''
        size = len(df.index)
        n_train = int(size*bins[0]/100)
        n_validate = int(size*bins[1]/100)
        n_test = size - n_train - n_validate
        # Data bins
        train = df[:n_train]
        validate = df[n_train:n_train + n_validate]
        test = df[n_train+n_validate:]
        return train, validate, test

    def split_array(self, df: np.ndarray,bins: list) -> tuple:
        ''' Split the dataset into train, validate and test bins '''
        size = len(df)
        n_train = int(size*bins[0]/100)
        n_validate = int(size*bins[1]/100)
        n_test = size - n_train - n_validate
        # Data bins
        train = df[:n_train]
        validate = df[n_train:n_train + n_validate]
        test = df[n_train+n_validate:]
        return train, validate, test

    def compute_dataset_features(self,  dataset: pd.DataFrame)->np.asarray:
        # Test on a small dataset
        # Extract the features from the dataset
        # pandarallel will execute the feature extraction on multiple processors
        pandarallel.initialize()
        start_time = datetime.now()
        self.logger.info('Extracting dataset features .....')
        features_label = dataset.parallel_apply(self.extract_features, axis=1)
        self.logger.info('Elapsed time: {}'.format(datetime.now()-start_time))
        # Save the feature extraction for later retrieval
        np.save('features_label', features_label)
        return features_label


    def extract_features(self, files):
        ''' Function to extract features from an audio file '''

        # The path to the audio file
        file_name = str(files.file)

        # Loads the audio file as a floating point time series and assigns the default sample rate
        # Sample rate is set to 22050 by default
        X, sample_rate = librosa.load(file_name, res_type='kaiser_fast')

        # Generate Mel-frequency cepstral coefficients (MFCCs) from a time series
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)

        # Generates a Short-time Fourier transform (STFT) to use in the chroma_stft
        stft = np.abs(librosa.stft(X))

        # Computes a chromagram from a waveform or power spectrogram.
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)

        # Computes a mel-scaled spectrogram.
        mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)

        # Computes spectral contrast
        contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)

        # Computes the tonal centroid features (tonnetz)
        tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X),
                                                  sr=sample_rate).T,axis=0)

        # We add also the classes of each file as a label at the end
        label = files.label

        return mfccs, chroma, mel, contrast, tonnetz, label

    def load_features(self, dataset: str)-> np.asarray:
        # loading the features
        features_label = np.load(dataset, allow_pickle=True)
        return features_label

    def train(self, trained_model:str, features_label: pd.DataFrame, bins: list) -> tuple:
        ''' Train the Neural Network'''
        # We create an empty list where we will concatenate all the features into one long feature
        # for each file to feed into our neural network

        features = []
        for i in range(0, len(features_label)):
            features.append(np.concatenate((features_label[i][0], features_label[i][1],
                                            features_label[i][2], features_label[i][3],
                                            features_label[i][4]), axis=0))
        # Similarly, we create a list where we will store all the labels
        labels = []
        for i in range(0, len(features_label)):
            labels.append(features_label[i][5])
        # Hot encoding y and pre processing X and y
        # Setting our X as a numpy array to feed into the neural network
        X = np.array(features)
        # Setting our y
        y = np.array(labels)
        # Hot encoding y
        lb = LabelEncoder()
        y = to_categorical(lb.fit_transform(y))
        # Print the shape
        self.logger.info('X shape: {}'.format(X.shape))
        self.logger.info('y shape: {}'.format(y.shape))
        # Split into train, validate, test
        X_train, X_val, X_test = self.split_array(X, bins)
        y_train, y_val, y_test = self.split_array(y,bins)
        ss = StandardScaler()
        X_train = ss.fit_transform(X_train)
        X_val = ss.transform(X_val)
        X_test = ss.transform(X_test)

        # Build the NN
        # Build a simple dense model with early stopping with softmax for categorical classification
        # Note that we use softmax for binary classification because it gives us a better result
        # than sigmoid for our probabilities in case we decide to use a voting classifier

        self.logger.info('Starting the model training')
        model = Sequential()

        model.add(Dense(193, input_shape=(193,), activation = 'relu'))
        model.add(Dropout(0.1))

        model.add(Dense(128, activation = 'relu'))
        model.add(Dropout(0.25))

        model.add(Dense(128, activation = 'relu'))
        model.add(Dropout(0.5))

        model.add(Dense(2, activation = 'softmax'))

        model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

        early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=100, verbose=1, mode='auto')

        # fitting the model with the train data and validation with the validation data
        # we used early stop with patience 15
        history = model.fit(X_train, y_train, batch_size=256, epochs=100,
                            validation_data=(X_val, y_val),
                            callbacks=[early_stop])
        # Save the trained model
        model.save(trained_model)

        return model, history