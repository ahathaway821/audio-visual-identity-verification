from keras.engine import Model
from keras import models
from keras import layers
from keras.layers import Input
from keras.preprocessing import image
from keras_vggface.vggface import VGGFace
import numpy as np
from keras_vggface import utils
import scipy as sp
from scipy import spatial
import cv2
import os
import glob
import pickle
import pyaudio
import wave
import threading
import time
import subprocess
from pathlib import Path
from face_authentication_inference import FaceIdentify
from audio_authentication_inference_en import AudioIdentify

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 32000

# SECONDS OF AUDIO TO CAPTURE SNIPPET
ROLLING_RECORD_SECONDS = 4
WAVE_OUTPUT_DIR="temp_files/"
WAVE_OUTPUT_FILENAME = WAVE_OUTPUT_DIR + "voice.wav"
CASE_PATH = "./pretrained_models/haarcascade_frontalface_alt.xml"

p = pyaudio.PyAudio()
frames = []

def write_audio():
    Path(WAVE_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    # if there are more than 4 seconds, trim to last 4
    if RATE / CHUNK > 4:
        wf.writeframes(b''.join(frames[-int(RATE / CHUNK * 4):]))
    else:
        wf.writeframes(b''.join(frames))
    wf.close()

def start_user_authentication_loop():
    face_model = FaceIdentify(precompute_features_file="./data/precompute_features.pickle")
    audio_model = AudioIdentify("./data/audio/x_english_ref.wav")

    face_cascade = cv2.CascadeClassifier(CASE_PATH)

    # 0 means the default video capture device in OS
    video_capture = cv2.VideoCapture(1)

    i = 0
    AUTHORIZED_AUDIO_TIME = 0
    AUTHORIZED_FACE_TIME = 0
    AUTHORIZED_AUDIO = False
    AUTHORIZED_FACE = False

    try:
        while True:
            if not video_capture.isOpened():
                sleep(5)
            # Capture frame-by-frame
            ret, frame = video_capture.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=10,
                minSize=(64, 64)
            )

            if i % 25 == 0:
                print(i)

            # Audio Inference
            if i % 100 == 0:
                #TODO - hold onto audiofile and maybe instead don't write the file?
                write_audio()

                #if audio was good within last TBD seconds (200 loops) -- call audio good
                if AUTHORIZED_AUDIO and AUTHORIZED_AUDIO_TIME > i - 200:
                    # audio is still good -- no need to run it again
                    pass
                else:
                    is_audio_authed, conf = audio_model.authenticate_voice(WAVE_OUTPUT_FILENAME)
                    AUTHORIZED_AUDIO = AUTHORIZED_AUDIO or is_audio_authed
                    if AUTHORIZED_AUDIO:
                        print("AUTHORIZED_AUDIO")
                        AUTHORIZED_AUDIO_TIME = i
                    else:
                        print(f"Audio auth: {AUTHORIZED_AUDIO} conf: {conf}")
            
            # Face Inference
            if i % 100 == 0:
                print("inside face")
                if AUTHORIZED_FACE and AUTHORIZED_FACE_TIME > i - 200:
                    # video is still good -- no need to run it again
                    pass
                else: 
                    is_face_authed = face_model.authenticate_face(faces, frame)
                    AUTHORIZED_FACE = AUTHORIZED_FACE or is_face_authed
                    if AUTHORIZED_FACE:
                        print("AUTHORIZED FACE")
                        AUTHORIZED_FACE_TIME = i

            if AUTHORIZED_FACE & AUTHORIZED_AUDIO:
                AUTHORIZED_BOTH = True
                print("AUTHORIZATION SUCCESSFUL")
                break

            i+=1

        # When everything is done, release the capture
        video_capture.release()
        cv2.destroyAllWindows()

    except Exception as exception:
        print("error")
        print(exception)

def main():
    start_user_authentication_loop()

if __name__ == "__main__":
    main()