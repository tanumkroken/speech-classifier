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
    def __init__(self, dataset: list, logger):
        ''' Initialize the class'''
        self.dataset = dataset
        self.label = defaultdict(list)
        self.logger = logger
        self.csv_file = "randomized_dataset.csv"
        for tup in dataset:
            self.label[tup[0]].append(tup[1])
            self.label[tup[0]].append(tup[2])

    def prepare_dataset(self, load: bool = False):
        ''' Organize and randomize the dataset'''
        if load is False:
            # Organize the dataset and randomize it
            logger.info('Organizing and randomizing the datasets')
            index = 0
            dataframes = []
            for label in self.label:
                files = []
                folder = self.label[label][0]
                file_ext = self.label[label][1]
                p = Path(folder)
                file_list = p.glob('*' + file_ext)
                for file in file_list:
                    files.append(file.name)
                # Create the dataframe for the labeled set
                df = pd.DataFrame(files)
                df = df.rename(columns = {0:'file'})
                # Add the label index
                df['label'] = index
                logger.info('Dataset labeled {}: {}, {} files'.format(label,folder,len(files)))
                # Store the df instance
                dataframes.append(df)
                index += 1
            # Join the dataframes
            self.dataset = pd.concat(dataframes, ignore_index=True)
            # Randomizing our files to be able to split into train, validation and test
            self.dataset = self.dataset.sample(frac=1, random_state=42).reset_index(drop=True)
            # Store the randomized set if we want to reproduce the results later
            self.dataset.to_csv(self.csv_file)
            logger.info('Randomized dataset written to "{}"'.format(self.csv_file))
        # Load the randomized datset from file
        else:
            self.dataset = pd.read_csv(self.csv_file)
            logger.info('Randomized dataset loaded from "{}"'.format(self.csv_file))
        return

    def train(self,train:int, validate:int, test:int):
        ''' The the neural network'''
        # Training set
        size = len(self.dataset.index)
        training = int(size*train/100)
        validating = int(size*validate/100)
        testing = size - training - validating
        # Training set
        df_train = self.dataset[:training]
        result = df_train['label'].value_counts(normalize=True)
        self.logger.info('Training set Length: {}, Normalized split '.format(result))
        # Validation set
        df_validate = self.dataset[training:validating]
        result = df_validate['label'].value_counts(normalize=True)
        self.logger.info('Validation set: {}'.format(result))
        # Test set
        df_test = self.dataset[validating:]
        result = df_test['label'].value_counts(normalize=True)
        self.logger.info('Test set: {}'.format(result))
        return

    def extract_features(files):
        ''' Function to extract features from a audio file '''

        # Sets the name to be the path to where the file is in my computer
        file_name = os.path.join(os.path.abspath('voice')+'/'+str(files.file))

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