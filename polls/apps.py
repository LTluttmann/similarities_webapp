from django.apps import AppConfig
from django.conf import settings
import os
import pickle
import tensorflow as tf
from keras import Model
import numpy as np

class PollsConfig(AppConfig):
    name = 'polls'


class LSHConfig(AppConfig):
    """
    This class loads the persisted LSH class. LSH has to be persisted as we need all the information
    about how hash tables have been generated (which random hyperplanes have been used). All this information
    is used to hash new items. Therefore it is pickled and stored in myapp/persisted_models. In the settings.py of
    the similarities_webapp directory we specify the path to all persisted models (so this one, NNs etc) in a variable
    called MODELS. This is used here to find the pickled LSH class and to load it. It can then be accessed in a view
    by calling: from .apps import LSHConfig and on LSHConfig calling .data
    """
    # create path to lsh
    path = os.path.join(settings.MODELS, 'lsh.p')
    # load models into separate variables
    # these will be accessible via this class
    with open(path, 'rb') as pickled:
        data = pickle.load(pickled)


class AutoencoderConfig(AppConfig):
    # create path to lsh
    path = os.path.join(settings.MODELS, 'autoencoder')
    autoencoder = tf.keras.models.load_model(path)
