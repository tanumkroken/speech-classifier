#  Copyright (c) 2021 by Ole Christian Astrup. All rights reserved.  Licensed under MIT
#   license.  See LICENSE in the project root for license information.
#

import pyttsx3
engine = pyttsx3.init(driverName='nsss')
#engine.say("I will speak this text")
#engine.runAndWait()

""" RATE"""
rate = engine.getProperty('rate')   # getting details of current speaking rate
print ('Rate: ', rate)                        #printing current voice rate


"""VOLUME"""
volume = engine.getProperty('volume')   #getting to know current volume level (min=0 and max=1)
print('Volume:',volume)                          #printing current volume level

"""VOICE"""

voices = engine.getProperty('voices')       #getting details of current voice
print(voices)
i = 0
for voice in voices:
    print(i, voice.id, voice.age, voice.gender, voice.languages)
    i +=1
engine.setProperty('voice', voices[30].id)  #changing index, changes voices. 0 for male
#engine.setProperty('voice', voices[1].id)   #changing index, changes voices. 1 for female
engine.setProperty('volume',1.0)    # setting up volume level  between 0 and 1
engine.setProperty('rate', rate + 50)     # setting up new voice rate

engine.say("Hello World!")
engine.say('My current speaking rate is ' + str(rate))
engine.runAndWait()
engine.stop()

"""Saving Voice to a file"""
# On linux make sure that 'espeak' and 'ffmpeg' are installed
engine.save_to_file('Hello World', 'test.mp3')
engine.runAndWait()