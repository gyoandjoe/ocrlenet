from Domain import LayerEnum
import Utilities

__author__ = 'win-g'

import theano
import theano.tensor as T
import numpy as np

import Domain.OCRLenetArquitecture as arqui

""" instanciacion de la arquitectura y creacion de funciones y variables que necesita para ser entrenada """
class LenetCNN(object):
    def __init__(self, batch_size, train_set,initial_weights=None,weigths_service = None): # lowAndHigh_c1Values= [-0.3,0.3],lowAndHigh_c3Values = [-0.1,0.1],lowAndHigh_fc5Values = [-0.01,0.01],lowAndHigh_fc6Values = [-0.001,0.001]):
        x = T.tensor4('x')  # the data is presented as rasterized images
        y = T.lvector('y')

        # batch_size = 50000
        # img_input =  x #T.reshape(x,(batch_size, 1, 28, 28))
        self.cnn = arqui.OCRLenetArquitecture(
            img_input=x,
            batch_size=batch_size,
            initWeights=initial_weights,
            weigths_service = weigths_service
            )


        # the cost we minimize during training is the NLL of the model
        cost = self.cnn.LR6.negative_log_likelihood(y)
        errors = self.cnn.LR6.errors(y)

        self.Weights = [self.cnn.LR6.Filter, self.cnn.LR6.Bias, self.cnn.FC5.Filter, self.cnn.FC5.Bias, self.cnn.C3.Filter, self.cnn.C1.Filter]

        grads = T.grad(cost, self.Weights, disconnected_inputs="raise")


        # train_model is a function that updates the model parameters by
        # SGD Since this model has many parameters, it would be tedious to
        # manually create an update rule for each model parameter. We thus
        # create the updates list by automatically looping over all
        # (params[i], grads[i]) pairs.

        learningRate = T.dscalar()

        updates = [
            (param_i, param_i + (learningRate * grad_i))
            for param_i, grad_i in zip(self.Weights, grads)
            ]

        trainset_x = theano.shared(train_set[0])
        trainset_y = theano.shared(train_set[1])

        index = T.lscalar()

        #bs = T.lscalar()

        self.train_model = theano.function(
            [index, learningRate],
            cost,  # self.classifier.FC.p_y_given_x,#dropout.output
            updates=updates,
            givens={
                x: trainset_x[index * batch_size: (index + 1) * batch_size],
                y: trainset_y[index * batch_size: (index + 1) * batch_size]
            }
        )

        self.evaluation_model_with_cost = theano.function(
            [index],
            cost,  # self.classifier.FC.p_y_given_x,#dropout.output
            givens={
                x: trainset_x[index * batch_size: (index + 1) * batch_size],
                y: trainset_y[index * batch_size: (index + 1) * batch_size]
            }
        )

        self.evaluation_model_with_errors = theano.function(
            [index],
            errors,
            givens={
                x: trainset_x[index * batch_size: (index + 1) * batch_size],
                y: trainset_y[index * batch_size: (index + 1) * batch_size]
            }
        )

    def GetWeigthsValuesByLayer(self,layer):
        if layer is LayerEnum.LayerEnum.c1Values:
            return np.asarray(self.cnn.C1.Filter.get_value())
        elif layer is LayerEnum.LayerEnum.c3Values:
            return np.asarray(self.cnn.C3.Filter.get_value())
        elif layer is LayerEnum.LayerEnum.fc5Values:
            return np.asarray(self.cnn.FC5.Filter.get_value())
        elif layer is LayerEnum.LayerEnum.fc5BiasValues:
            return np.asarray(self.cnn.FC5.Bias.get_value())
        elif layer is LayerEnum.LayerEnum.fc6Values:
            return np.asarray(self.cnn.LR6.Filter.get_value())
        elif layer is LayerEnum.LayerEnum.fc6BiasValues:
            return np.asarray(self.cnn.LR6.Bias.get_value())
        return

