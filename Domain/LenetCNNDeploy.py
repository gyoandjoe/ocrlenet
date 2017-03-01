from Domain import LayerEnum
import Utilities

__author__ = 'win-g'

import theano
import theano.tensor as T
import numpy as np

import Domain.OCRLenetArquitecture as arqui

""" instanciacion de la arquitectura y creacion de funciones y variables que necesita para ser entrenada """


class LenetCNNDeploy(object):
    def __init__(self, batch_size,data_set, initial_weights=None,
                 weigths_service=None):  # lowAndHigh_c1Values= [-0.3,0.3],lowAndHigh_c3Values = [-0.1,0.1],lowAndHigh_fc5Values = [-0.01,0.01],lowAndHigh_fc6Values = [-0.001,0.001]):
        x = T.tensor4('x')  # the data is presented as rasterized images
        y = T.lvector('y')

        # batch_size = 50000
        # img_input =  x #T.reshape(x,(batch_size, 1, 28, 28))
        self.cnn = arqui.OCRLenetArquitecture(
            img_input=x,
            batch_size=batch_size,
            initWeights=initial_weights,
            weigths_service=weigths_service
        )


        # the cost we minimize during training is the NLL of the model

        pred = self.cnn.LR6.y_pred

        self.Weights = [self.cnn.LR6.Filter, self.cnn.LR6.Bias, self.cnn.FC5.Filter, self.cnn.FC5.Bias,
                        self.cnn.C3.Filter, self.cnn.C1.Filter]

        data_x = theano.shared(data_set)

        index = T.lscalar()

        self.predictWithModel = theano.function(
            [index],
            pred,
            givens={
                x: data_x[index * batch_size: (index + 1) * batch_size]
                # y: trainset_y[index * batch_size: (index + 1) * batch_size]
            }
        )
