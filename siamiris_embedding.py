"""SiamIris Object"""

import configparser
import logging

import os

import cv2 as cv
import numpy as np

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


from tensorflow.keras.applications import mobilenet_v2, resnet50
from scipy.spatial import distance

logger = logging.getLogger("fakeid_classification")

class siamiris_embedding(object):
    def __init__(self, backbone='resnet50', input_size=(224,224), distance='euclidean'):
        """Class instantiation.

        Parameters
        ----------
        backbone : str
            Model backbone < 'resnet50' | 'mobilenetv2' >
        input_size : tuple, optional
            Image target size, by default (224, 224)
        distance: str, optional
            Distance function for comparing two embeddings < 'L1' | 'euclidean' | 'cosine' >
        """

        self.input_size = input_size
        self.backbone = backbone
        self.distance = distance

        if self.backbone == 'mobilenetv2':
            self._preprocessor = mobilenet_v2.preprocess_input
            modelpath = '/home/daniel/GitHub/SiamIris-v1/weights/SiamIris-MN2-iris/'
        elif self.backbone == 'resnet50':
            self._preprocessor = resnet50.preprocess_input
            modelpath = '/home/daniel/GitHub/SiamIris-v1/weights/SiamIris-R50-iris/'
        else:
            raise ValueError('Backbone %s is currently not supported' % self.backbone)

        self.model = tf.keras.models.load_model(modelpath, compile=False, custom_objects={'tf': tf})
        self.warm_up()

        logger.debug(f"Instanciating {self}\nModel path: '{modelpath}'")

    def warm_up(self):
        dummy_image = np.zeros((1,224,224,3))
        embedding = self.model.predict(dummy_image, verbose=0)
        self.embedding_size = len(embedding[0])
        return

    def process_image(self, image):
        """Resample and process the input image.

        Parameters
        ----------
        image : numpy.ndarray
            Input image.

        Returns
        -------
        numpy.ndarray
            Resampled and processed image.
        """
        if image.shape[:2] != self.input_size[::-1]:
            image = cv.resize(image, self.input_size)

        return image

    def get_embedding(self, image):
        """Get embedding of an imput image.

        Parameters
        ----------
        image : numpy.ndarray
            Image to classify.

        Returns
        -------
        numpy.ndarray
            Embedding or feature vector.
        """
        if len(image.shape) < 3:
            image = np.expand_dims(image, axis=-1)
        image = np.expand_dims(image, axis=0)
        image = self._preprocessor(image)

        embedding = self.model.predict(image, verbose=0)

        return embedding[0]

    def compare(self, emb1, emb2):
        """Compute distance between two embeddings.

        Parameters
        ----------
        emb1 : numpy.ndarray
            Embedding 1.
        emb2 : numpy.ndarray
            Embedding 2.

        Returns
        -------
        dist: float
            distance.
        """
        if self.distance in ['euclidean', 'L2']:
            dist = np.linalg.norm(emb1 - emb2, axis=0)
        elif self.distance == 'cosine':
            if np.linalg.norm(emb1) < 1e-10 or np.linalg.norm(emb2) < 1e-10:  # Case when the CNN generates a vector that is all 0's
                dist = np.ones(emb1.shape)
            dist = distance.cdist(emb1, emb2, metric='cosine')
        else:  # L1 distance by default
            dist = np.sum(np.abs(emb1 - emb2))

        return dist
