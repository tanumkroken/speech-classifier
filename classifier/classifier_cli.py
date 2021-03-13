#!/usr/bin/env python
# -*- coding: utf-8 -*-

#  Copyright (c) 2021 by Ole Christian Astrup. All rights reserved.  Licensed under MIT
#   license.  See LICENSE in the project root for license information.
#

import click
import json
from classifier.nn_classifier import GenderClassifier
from classifier.utils import init_logger
import os, sys, time, datetime
from pathlib import Path
import matplotlib.pyplot as plt


@click.command()
@click.argument('dataset', type=click.Path(exists=True), default='dataset.json')  # The input dataset described in json

@click.option('-r','--randomize', is_flag=True,
              help='Use this flag to create a randomized dataset. If not used, a previously randomized set will be loaded from file.')

@click.option('-rd', '--randomized_dataset', default='randomized_dataset.csv',
              help='The file name of the randomized dataset.')

@click.option('-ff', '--features_file', default='features_label.npy',
              help='The file name to store the computed dataset features.')

@click.option('-f', '--features', is_flag=True,
              help='Use this flag to compute the dataset features and store this on file')

@click.option('-t', '--train', is_flag=True,
              help='Use this flag to train the NN from tha computed features')

@click.option('-to', '--test_only', is_flag=True,
              help='Use this flag to test the workflow on a reduced dataset')

@click.option('-ts', '--test_size', type=int, default = 100,
              help='The size of the test set')

@click.option('-m', '--model', type=click.Path(exists=False),
              default='./trained_model',
              help='The folder where a trained keras model can be stored')




def classify(dataset, randomize, randomized_dataset, features_file, features, train,
             test_only, test_size, model):
    #    click.echo('classifier')
    """ Build a Neural Network from a given datset.
        The dataset is divided into subset representing the classes
    """
    # Set up the logger
    logger = init_logger('classifier',False, 'INFO')
    logger.info('Validating input dataset {}'.format(dataset))
    with open(dataset) as json_file:
        data = json.load(json_file)
        source_folder = data['source_folder']
        number_of_classes = int(data['number_of_classes'])
        p = Path(source_folder)
        if p.exists() is True:
            logger.info('Source folder : {} OK'.format(source_folder))
        else:
            logger.error('The folder {} does not exist'.format(source_folder))
            sys.exit()
        labels = data['labels']
        folders = []
        for folder in labels:
            data_folder = source_folder + folder
            p = Path(data_folder)
            if p.exists() is True:
                logger.info('Data folder {} exists'.format(data_folder))
                folders.append(data_folder)
            else:
                logger.error('The data folder {} does not exist'.format(data_folder))
                sys.exit()
        audio_formats = data['audio_formats']
        logger.info('Audio formats: {}'.format(data['audio_formats']))
        bins = []
        size = 0
        for i in range(number_of_classes):
            bin = int(data['bin_size'][i])
            size += bin
            logger.info('Bin {} size: {}%'.format(i+1,bin))
            bins.append(bin)
        # Last bin
        logger.info('Bin {} size: {}%'.format(i+2,100 - size))
    # Create the classifier
        classifier = GenderClassifier(source_folder,number_of_classes, labels, folders,  audio_formats, logger)
        if randomize is True:
            dataset = classifier.randomize_daset(randomized_dataset)
        else:
            dataset = classifier.load_randomized_dataset(randomized_dataset)
        # Check the dataset balance
        classifier.check_label_balance(dataset, bins)
        if features is True:
            if test_only is True:
                fatures_label = classifier.compute_dataset_features(dataset[:test_size])
            else:
                 features_label = classifier.compute_dataset_features(dataset)
        # Load previously computed features
        else:
            p = Path(features_file)
            if p.exists() is True:
                feature_label = classifier.load_features(features_file)
                logger.info('Loaded features from file: {}'.format(features_file))
            else:
                logger.error('The features file {} does not exist. Not possible to load feature set'.format(features_file))
                sys.exit()
        #Train the data
        model, history = classifier.train(model, feature_label, bins)

        # Print model summary
        model.summary()

        # Check out our train accuracy and validation accuracy over epochs.
        train_accuracy = history.history['accuracy']
        val_accuracy = history.history['val_accuracy']

        # Set figure size.
        plt.figure(figsize=(12, 8))

        # Generate line plot of training, testing loss over epochs.
        plt.plot(train_accuracy, label='Training Accuracy', color='#185fad')
        plt.plot(val_accuracy, label='Validation Accuracy', color='orange')

        # Set title
        plt.title('Training and Validation Accuracy by Epoch', fontsize = 25)
        plt.xlabel('Epoch', fontsize = 18)
        plt.ylabel('Categorical Crossentropy', fontsize = 18)
        plt.xticks(range(0,100,5), range(0,100,5))

        plt.legend(fontsize = 18)
        plt.show()

if __name__ == "__main__":
    classify()