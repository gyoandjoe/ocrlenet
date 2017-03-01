__author__ = 'win-g'
import cPickle

import theano
import numpy as np
import pickle
from Domain import LenetCNNDeploy
from Domain import LayerEnum

"""
Esta clase implementa el servicio de consumo final para el usuario
"""


class LenetService(object):
    def __init__(self, batch_size, dataset_size, weigthts_service, initial_weights, dataset_file=None):
        self.initial_weights = initial_weights
        self.dataset_size = int(dataset_size)
        self.batch_size = int(batch_size)
        self.weigthts_service = weigthts_service

        self.rawInput = None
        if dataset_file is not None:
            fLoaded = file(dataset_file, 'rb')
            self.rawInput = cPickle.load(fLoaded)

        self.no_batchs = self.dataset_size / self.batch_size

    def Predict(self, image):
        XimgLetras = np.asarray(image, dtype=theano.config.floatX).reshape((self.dataset_size, 1, 28, 28))

        lenetcnn = LenetCNNDeploy.LenetCNNDeploy(
            batch_size=self.batch_size,
            data_set =XimgLetras,
            initial_weights=self.initial_weights,
            weigths_service=self.weigthts_service
        )

        return lenetcnn.predictWithModel(0)
