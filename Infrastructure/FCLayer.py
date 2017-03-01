# coding=utf-8
__author__ = 'win-g'

import numpy as np
import theano
import theano.tensor as T


class FCLayer(object):
    def __init__(self, name, input, initial_filter_values, initialbiasvalues, activationFunction):
        """
        Constructor
        Inicializa los pesos, asigna los filtros y define las funciones para predecir.

        :param name:
        :param input:
        :param initial_filter_values:
        :param initialbiasvalues:
        :param activationFunction:
        :return:
        """

        self.Filter = theano.shared(
            value=initial_filter_values,
            name='FilterFCLayer'+name,
        )

        self.Bias = theano.shared(
            value=initialbiasvalues,
            name='BiasFCLayer'
        )

        #self.Weights = (self.Filter, self.Bias)


        ## T.dot(input, self.Filter) + self.Bias ###  produce una matriz de tamaño NoEjemplosXNoClasesDeseadasClasificar
        self.ProductoCruz = T.dot(input, self.Filter) + self.Bias

        self.output = activationFunction(self.ProductoCruz)
        #self.output = (
        #    self.ProductoCruz if activationFunction is None
        #    else activationFunction(self.ProductoCruz)
        #)
