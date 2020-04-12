import os
import numpy as np
import tarfile
import ibm_boto3
from ibm_botocore.client import Config, ClientError

import tensorflow as tf
import keras
from keras import layers, models, Input, optimizers, metrics, regularizers
from keras import backend as K

import librosa

# Constants for IBM S3 values
COS_ENDPOINT = 'https://s3.us-south.cloud-object-storage.appdomain.cloud'
COS_API_KEY_ID = 'LKRr_5OhOyBgvHG6WH2wm9F_2bHC2sn1vV4eaCYdgpsm'
COS_AUTH_ENDPOINT = 'https://iam.cloud.ibm.com/identity/token'
COS_RESOURCE_CRN = 'crn:v1:bluemix:public:cloud-object-storage:global:a/ea337a3eba2f43c6b813f319db505255:0f9730a8-f2b8-42ce-b276-f9e13877a5f0::'

DATA_PATH = 'data/'
BUCKET_NAME = 'cv-audio'
DEV_FILE = 'dev.tsv'
AUDIO_FILE = 'dev-clips.tgz'
MFCC_PICKLE = 'mfcc.bin'
CLIPS_PATH = DATA_PATH + 'dev-clips/'
TEST_CLIPS = DATA_PATH + 'test-clips/'
MODEL_NAME = "EN-TF1-20200405-024254"
BEST_MODEL_FILE = f'{MODEL_NAME}.tgz'
TEST_RECORDINGS = 'test-clips.tgz'

PROD_PATH = DATA_PATH + 'production/'
REF_CLIP = PROD_PATH + 'ref_clip.wav'
REF_CLIPS_PICKLE = PROD_PATH + 'ref_clips.pickle'
BEST_MODEL_PATH = DATA_PATH + 'production/model/'
SCORE_THRESHOLD = 0.5
REF_CLIPS = None
MODEL = None

bucket_name = 'cv-audio'

class AudioIdentify(object):
    def __init__(self, ref_clip_path):
        self.load_model_from_bucket()
        self.ref_clips, self.model = self.start_voice_system(ref_clip_path)

    def load_model_from_bucket(self):
        """### Pull Down Necessary Files from Bucket for Inference"""
        cos = ibm_boto3.resource("s3",
        ibm_api_key_id=COS_API_KEY_ID,
        ibm_service_instance_id=COS_RESOURCE_CRN,
        ibm_auth_endpoint=COS_AUTH_ENDPOINT,
        config=Config(signature_version="oauth"),
        endpoint_url=COS_ENDPOINT
        )
        
        # Load model
        modelResponse = cos.Object(bucket_name, BEST_MODEL_FILE).get()
        model_tar = modelResponse['Body'].read()
        if not os.path.exists(BEST_MODEL_PATH):
            os.makedirs(BEST_MODEL_PATH)
        with open(BEST_MODEL_PATH + BEST_MODEL_FILE , 'wb') as file:
            file.write(model_tar)

        tar = tarfile.open(f"data/production/model/{MODEL_NAME}.tgz")
        tar.extractall("data/production/model")
        tar.close()

    def get_siamese_model(self):
    
        # Define the tensors for the two input images
        left_input = Input((20, 400, 1))
        right_input = Input((20, 400, 1))
        
        # Convolutional Neural Network
        model = models.Sequential()    
        model.add(layers.Conv2D(
            32, 
            (10,10), 
            padding = 'same',
            activation='relu', 
            input_shape=(20, 400, 1), 
            kernel_regularizer=regularizers.l2(2e-4)))
        model.add(layers.MaxPooling2D())
        
        model.add(layers.Conv2D(
            64, 
            (7,7),  
            padding = 'same',
            activation='relu',
            kernel_regularizer=regularizers.l2(2e-4)))    
        model.add(layers.MaxPooling2D())
        
        model.add(layers.Conv2D(
            64, 
            (4,4), 
            padding = 'same', 
            activation='relu', 
            kernel_regularizer=regularizers.l2(2e-4)))
        model.add(layers.MaxPooling2D())
        
        model.add(layers.Conv2D(
            128, 
            (4,4),  
            padding = 'same',
            activation='relu', 
            kernel_regularizer=regularizers.l2(2e-4)))
        
        model.add(layers.Flatten())
        
        model.add(layers.Dense(
            1024, 
            activation='sigmoid',
            kernel_regularizer=regularizers.l2(1e-3)))
        
        # Generate the encodings (feature vectors) for the two images
        encoded_l = model(left_input)
        encoded_r = model(right_input)
        
        # Add a customized layer to compute the absolute difference between the encodings
        L1_layer = layers.Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
        L1_distance = L1_layer([encoded_l, encoded_r])
        
        # Add a dense layer with a sigmoid unit to generate the similarity score
        prediction = layers.Dense(1, activation='sigmoid')(L1_distance)
        
        # Connect the inputs with the outputs
        siamese_net = models.Model(inputs=[left_input, right_input],outputs=prediction)
        
        # return the model
        return siamese_net



    """### Helper Functions"""

    def get_clips2(self, clip_path, recorded_byte_array, min_size=1):    
        """
        Splits a long clip in multiple smaller clips with MFCC length of 400. 
        Also discards the final part of the audio file the is not a multiple of 400.
        """
        
        #TODO: this function should also work with a recorded byte array
        # since we don't need to save the user's voice everytime they try to open the door
        
        pad_length = 400
        
        wave, sr = librosa.load(clip_path, mono=True)
        wave = wave[::3]
        mfcc = librosa.feature.mfcc(wave, sr=sr)
        print(f'mfcc: {mfcc.shape[1]}')
        n_clips = int(mfcc.shape[1] / pad_length)
        
        if (n_clips < min_size):
            raise Exception('Audio file too short. Expected MFCC equal or greater than {}, but was: {}'.format(
                min_size*400, mfcc.shape[1]))
        
        # cut the recording in smaller clips
        ref_clips = []
        for i in range(1, n_clips+1):
            ref_clips.append(mfcc[:, (i-1)*pad_length:i*pad_length])
        print(f'Ref clips len: {len(ref_clips)}')
        return ref_clips

    def get_clips(self, file_name, recorded_byte_array, min_size=1):
        max_pad_len=400
        wave, sr = librosa.load(file_name, mono=True)
        wave = wave[::3]
        mfcc = librosa.feature.mfcc(wave, sr=sr)
        pad_width = max_pad_len - mfcc.shape[1]
        if (pad_width < 0):
            pad_width = 0
            mfcc = mfcc[:, 0:max_pad_len]
        mfcc = np.pad(mfcc, pad_width=((0,0), (0, pad_width)), mode='constant')
        return [mfcc.astype(np.float32)]

    def setup_voice_system(self):   
        """
        Records the voice of the user, splits the voice in small clippings and saves
        the clippings for later inference.
        """
        
        #TODO
        #record the voice and save the file in path REF_CLIP
        #if time allows, multiple users to open the door
        
        # expect at least 3 clips from reference recording 
        ref_clip_path = "./data/audio/x_english_ref_short.mp3"
        ref_clips = self.get_clips(os.fspath(ref_clip_path), 3)

        return ref_clips
        
        with open(REF_CLIPS_PICKLE, 'wb') as handle: 
            pickle.dump(ref_clips, handle)

    def start_voice_system(self, ref_clip_path):
        """
        Loads the reference clippings and the weights of the model.
        """

        #ref_clip_path = "./data/audio/x_english_ref.wav"
        ref_clips = self.get_clips(os.fspath(ref_clip_path), 3)

        model = self.get_siamese_model()
        model.load_weights(f'./data/production/model/{MODEL_NAME}')
        
        return ref_clips, model


    def get_user_voice(self, path):
        """
        Record the user voice for an attempted unlock and return smaller processed MFCC clippings
        """
        # TODO
        #recorded_byte_array = np.array()
        
        return self.get_clips(path, None, min_size=1)

    def get_inference_dataset(self, ref_clips, actual_clips):
        left = []
        right = []
        
        for ref in ref_clips:
            for actual in actual_clips:
                left.append([ref]) 
                right.append([actual]) 
        
        left = np.array(left).astype(np.float32)
        right = np.array(right).astype(np.float32)

        left = np.rollaxis(np.rollaxis(left, 3, 1), 3, 1)
        right = np.rollaxis(np.rollaxis(right, 3, 1), 3, 1)

        return left, right

    def speech_unlock(self, model, left, right, SCORE_THRESHOLD):
        scores = model.predict([left, right])
        print('model scores')
        print(scores)
        score = np.mean(scores)
        unlock = score > SCORE_THRESHOLD
        return unlock, score

    def authenticate_voice(self, unlock_attempt_clip_path):
        #get new trial voice for attempted unlock
        unlock_attempt_clip_path = "./data/audio/x_english_ref_short_2.mp3"
        print(f'unlock attempt clip path: {unlock_attempt_clip_path}')
        actual_clips = self.get_user_voice(os.fspath(unlock_attempt_clip_path))
        print(f'actual clips len: {len(actual_clips)}')
        left, right = self.get_inference_dataset(self.ref_clips, actual_clips)

        unlock, score = self.speech_unlock(self.model, left, right, SCORE_THRESHOLD)

        return unlock, score