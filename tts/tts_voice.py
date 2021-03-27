#  Copyright (c) 2021 by Ole Christian Astrup. All rights reserved.  Licensed under MIT
#   license.  See LICENSE in the project root for license information.
#

import speech_recognition as sr
from gtts import gTTS
import os
import time
import playsound

def speak(text):
    tts = gTTS(text=text, lang='en-uk')
    filename = 'voice.mp3'
    tts.save(filename)
    playsound.playsound(filename)


speak("hello tim, can you hear me.")