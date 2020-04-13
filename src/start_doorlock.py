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
import sys
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
import traceback

from face_authentication_inference import FaceIdentify
from audio_authentication_inference_en import AudioIdentify

FACE_SIZE = 224

CHUNK = 512
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 48000
RECORD_SECONDS = 6

# SECONDS OF AUDIO TO CAPTURE SNIPPET
ROLLING_RECORD_SECONDS = 6
WAVE_OUTPUT_DIR="temp_files/"
WAVE_OUTPUT_FILENAME = WAVE_OUTPUT_DIR + "voice.wav"
CASE_PATH = "./pretrained_models/haarcascade_frontalface_alt.xml"

p = pyaudio.PyAudio()
audio_frames = []

# to enable non-blocking audio recording -- capturing audio in 'frames' global
def callback(in_data, frame_count, time_info, status):
    global audio_frames
    print(f'frame count: {frame_count}', flush=True)
    audio_frames.append(in_data)
    return in_data, pyaudio.paContinue

audio_stream = p.open(format=FORMAT,
                channels=CHANNELS,
                input_device_index=22,
                rate=RATE,
                input=True,
                output=False,
                frames_per_buffer=CHUNK)

print("**** Starting Audio Recording ****")
#audio_stream.start_stream()
#while(audio_stream.isActive()):
#    time.sleep(1)

def write_audio():
    Path(WAVE_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    wf = wave.open(WAVE_OUTPUT_FILENAME, 'w')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(audio_frames))
    wf.close()

def draw_label(image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,
                font_scale=1, thickness=2):
    size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    x, y = point
    cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
    cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness)

def crop_face(imgarray, section, margin=20, size=224):
    """
    :param imgarray: full image
    :param section: face detected area (x, y, w, h)
    :param margin: add some margin to the face detected area to include a full head
    :param size: the result image resolution with be (size x size)
    :return: resized image in numpy array with shape (size x size x 3)
    """
    img_h, img_w, _ = imgarray.shape
    if section is None:
        section = [0, 0, img_w, img_h]
    (x, y, w, h) = section
    margin = int(min(w, h) * margin / 100)
    x_a = x - margin
    y_a = y - margin
    x_b = x + w + margin
    y_b = y + h + margin
    if x_a < 0:
        x_b = min(x_b - x_a, img_w - 1)
        x_a = 0
    if y_a < 0:
        y_b = min(y_b - y_a, img_h - 1)
        y_a = 0
    if x_b > img_w:
        x_a = max(x_a - (x_b - img_w), 0)
        x_b = img_w
    if y_b > img_h:
        y_a = max(y_a - (y_b - img_h), 0)
        y_b = img_h
    cropped = imgarray[y_a: y_b, x_a: x_b]
    resized_img = cv2.resize(cropped, (size, size), interpolation=cv2.INTER_AREA)
    resized_img = np.array(resized_img)
    return resized_img, (x_a, y_a, x_b - x_a, y_b - y_a)

def start_user_authentication_loop():
    face_model = FaceIdentify(precompute_features_file="./data/precompute_features.pickle")
    audio_model = AudioIdentify("./data/audio/x_english_ref_7.wav")

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

            #if i % 25 == 0:
            #    print(i)

            # Audio Inference
            if i % 100 == 0 and AUTHORIZED_FACE:
                print("STARTING AUDIO RECORDING")                        
                audio_frames.clear()
                for j in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
                    data = audio_stream.read(CHUNK, exception_on_overflow=False)
                    audio_frames.append(data)
                write_audio()

                #if audio was good within last TBD seconds (200 loops) -- call audio good
                if AUTHORIZED_AUDIO and AUTHORIZED_AUDIO_TIME > i - 200:
                    # audio is still good -- no need to run it again
                    pass
                else:
                    is_audio_authed, conf = audio_model.authenticate_voice(WAVE_OUTPUT_FILENAME)
                    AUTHORIZED_AUDIO = AUTHORIZED_AUDIO or is_audio_authed
                    #if AUTHORIZED_AUDIO:
                    #    print("AUTHORIZED_AUDIO")
                    AUTHORIZED_AUDIO_TIME = i
                    #else:
                    print(f"Audio authorized?: {AUTHORIZED_AUDIO} | Confidence: {conf}")
            
            # Face Inference
            #if i % 2 == 0:
                #if AUTHORIZED_FACE and AUTHORIZED_FACE_TIME > i - 200:
                    # video is still good -- no need to run it again
                #    pass
                #else: 
            is_face_authed, predicted_names = face_model.authenticate_face(faces, frame)
            if is_face_authed and not AUTHORIZED_FACE:
                print("AUTHORIZED FACE")
                AUTHORIZED_FACE_TIME = i
                AUTHORIZED_FACE = AUTHORIZED_FACE or is_face_authed
            for face_index, face in enumerate(faces):
                label = "{}".format(predicted_names[face_index])
                draw_label(frame, (face[0], face[1]), label)
                    

            if AUTHORIZED_FACE & AUTHORIZED_AUDIO:
                AUTHORIZED_BOTH = True
                print("---FULL AUTHORIZATION SUCCESSFUL---")
                break

            face = None
            color = None
            if len(faces) > 1:  # Get the largest face as main face
                face = max(faces, key=lambda rectangle: (rectangle[2] * rectangle[3]))  # area = w * h
            elif len(faces) == 1:
                face = faces[0]
            if face is not None:
                face_img, cropped = crop_face(frame, face, margin=40, size=FACE_SIZE)
                (x, y, w, h) = cropped
                if AUTHORIZED_FACE:
                    color = (255, 200, 0)
                else:
                    color = (255, 0, 255)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.imshow('Faces', frame)
            cv2.imshow('Faces', frame)
            i+=1
            if cv2.waitKey(5) == 27:  # ESC key press
                break          

        # When everything is done, release the capture
        audio_stream.stop_stream()
        audio_stream.close()
        p.terminate()
        video_capture.release()
        cv2.destroyAllWindows()

    except Exception as exception:
        print("Error: unhandled exception")
        print(exception)
        print(traceback.format_exc())

def main():
    start_user_authentication_loop()

if __name__ == "__main__":
    main()
