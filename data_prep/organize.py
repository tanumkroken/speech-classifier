'''Organize the LibriSpeech training data
   We are using the train-clean-100 data set'''

import os
import sys
import re
from collections import defaultdict
import shutil
from pathlib import Path
import glob

AUDIO = '.flac'
TARGET_DIR = '../training_data/'
SOURCE_DIR = '../LibriSpeech/'
sys.path.insert(0, os.path.abspath("../../"))

filelist = os.listdir('../LibriSpeech/')

# Read the speaker meta data
speakers = open(SOURCE_DIR+'SPEAKERS.TXT', 'r')

training = {}
for line in speakers:
    # skip comment lines
    if ';' not in line:
#        ;ID  |SEX| SUBSET           |MINUTES| NAME
        # Filter the correct data sub-set
        if 'train-clean-100' in line:
            # Parse the metadata
            # The id is an integer at the sart of each line
            id = re.match('\d+',line)[0]
            # The gender is coded as capital letter M or F after the first pipe | delimiter
            sex = re.search('[MF]',line)[0]
            # The speakers name is between the | and the end of the line
            name = re.search(r"[\w\s]*$",line)[0].rstrip()
            #Remove end of line
            training[id]=(id,sex,name)
#for id in training:
#    print('Id: {} Sex: {} Name{}'.format(id,training[id][0],training[id][1]))
# Organize the training set in gender folders
source = SOURCE_DIR + 'train-clean-100'
# Get all audio files in the source and subfolders
p = Path(source)
audio_files = p.glob('**/*'+AUDIO)
# Organise the audio according to gender
for file in audio_files:
    pf = Path(file)
    # The file name only
    name = pf.name
    # The speaker id is coded in the file name
    id = re.match(r'^\d+',name)[0]
    # Only speakers in the training set
    if id in training:
        # Copy file to gender folders
        if training[id][1] == 'M':
            target = TARGET_DIR +'male/'+ name
#            shutil.copy(file,target)
            print('Copying source {} target {}'.format(file,target))
        else:
            target = TARGET_DIR + 'female/'+name
#            shutil.copy(file,target)
            print('Copying source {} target {}'.format(file,target))
# Write the Speaker meta data files
fh_fm = open(TARGET_DIR + 'female/SPEAKERS.TXT','w')
fh_fm.write('# ID; SEX; NAME\n')
fh_m = open(TARGET_DIR + 'male/SPEAKERS.TXT','w')
fh_m.write('# ID; SEX; NAME\n')
for id in training:
    if training[id][1] == 'M':
        fh_m.write('{}; {}; {}\n'.format(id, training[id][1],training[id][2]))
    else:
        fh_fm.write('{}; {}; {}\n'.format(id, training[id][1],training[id][2]))
fh_m.close()
fh_fm.close()