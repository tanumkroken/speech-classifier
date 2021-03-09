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
from pathlib import PurePath



@click.command()
@click.argument('dataset', type=click.Path(exists=True), default='dataset.json')  # The input dataset described in json

@click.option('-l','--load', is_flag=False,
              help='Use this flag to load a randomized dataset from file')

@click.option('-r', '--randomized_dataset', default='randomized_dataset.csv',
              help='A previously randomized dataset stored in the randomized_dataset.csv file')


def classify(dataset, load, randomized_dataset):
    #    click.echo('classifier')
    """ Build a Neural Network from a given datset.
        The dataset is divided into subset representing the classes
    """
    # Set up the logger
    logger = init_logger('classifier',False, 'INFO')
    with open(dataset) as json_file:
        data = json.load(json_file)
        logger.info('Source folder : {}'.format(data['source_folder']))
        logger.info('Labels: {}'.format(data['labels']))
        logger.info('Audio formats: {}'.format(data['audio_formats']))

    dataset = [('male',MALE, AUDIO),('female',FEMALE,AUDIO)]
    classifier = GenderClassifier(dataset,logger)
    classifier.prepare_dataset(True)
    classifier.train(70,20,10)

    #gender = GenderClassifier(dataset)



if __name__ == "__main__":
    classify()