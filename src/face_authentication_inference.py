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
from pathlib import Path

def load_stuff(filename):
    saved_stuff = open(filename, "rb")
    stuff = pickle.load(saved_stuff)
    saved_stuff.close()
    return stuff


class FaceIdentify(object):
    """
    Singleton class for real time face identification
    """
    CASE_PATH = "./pretrained_models/haarcascade_frontalface_alt.xml"

    def __new__(cls, precompute_features_file=None):
        if not hasattr(cls, 'instance'):
            cls.instance = super(FaceIdentify, cls).__new__(cls)
        return cls.instance

    def __init__(self, precompute_features_file=None):
        self.face_size = 224
        self.precompute_features_map = load_stuff(precompute_features_file)
        print("Loading VGG Face model...")
        self.model = VGGFace(model='resnet50',
                             include_top=False,
                             input_shape=(224, 224, 3),
                             pooling='avg')  # pooling: None, avg or max

        VIDEOS_FOLDER = "./data/videos"
        folders = list(glob.iglob(os.path.join(VIDEOS_FOLDER, '*')))
        self.authenticated_names = [os.path.basename(folder) for folder in folders]

        print("Loading VGG Face model done")

    @classmethod
    def draw_label(cls, image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,
                   font_scale=1, thickness=2):
        size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        x, y = point
        cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
        cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness)

    def crop_face(self, imgarray, section, margin=20, size=224):
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

    def identify_face(self, features, threshold=100):
        distances = []
        for person in self.precompute_features_map:
            person_features = person.get("features")
            # distance = sp.spatial.distance.euclidean(person_features, features)
            distance = spatial.distance.euclidean(person_features, features) #madia

            distances.append(distance)
        min_distance_value = min(distances)
        min_distance_index = distances.index(min_distance_value)
        if min_distance_value < threshold:
            return self.precompute_features_map[min_distance_index].get("name")
        else:
            return "?"

    def authenticate_face(self, faces, frame):
        if len(faces) == 0:
            return False, []

        # placeholder for cropped faces
        face_imgs = np.empty((len(faces), self.face_size, self.face_size, 3))
        for i, face in enumerate(faces):
            face_img, cropped = self.crop_face(frame, face, margin=10, size=self.face_size)
            face_imgs[i, :, :, :] = face_img
        if len(face_imgs) > 0:
            features_faces = self.model.predict(face_imgs)
            predicted_names = [self.identify_face(features_face) for features_face in features_faces]
            valid_names = [pred_name for pred_name in predicted_names if pred_name in self.authenticated_names]
            if len(valid_names) > 0:
                return True, predicted_names
            else:
                return False, predicted_names
        return False, []
