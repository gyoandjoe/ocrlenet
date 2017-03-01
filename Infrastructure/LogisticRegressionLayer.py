# coding=utf-8
__author__ = 'win-g'

import theano
import theano.tensor as T

class LogisticRegressionLayer(object):
    def __init__(self, name, input,initial_filter_values, initialbiasvalues):

        self.Filter = theano.shared(
            value= initial_filter_values,
            name='FilterLRLayer'+name,

        )

        self.Bias = theano.shared(
            value=initialbiasvalues,
            name='BiasLRLayer'
        )

        self.ProductoCruz = T.dot(input, self.Filter) + self.Bias

        """ ### T.nnet.softmax(self.ProductoCruz) ### para cada ejemplo(row) de self.ProductoCruz, por cada uno de sus clases(filas), produce una probabilidad con respecto de sus otras clases calculadas"""

        self.p_y_given_x = T.nnet.softmax(self.ProductoCruz)
        """ A partir de las probabilidades calculadas de cada imagen(self.p_y_given_x), con #T.argmax(self.p_y_given_x, axis=1)# obtenemos un arreglo de tamaño NoEjemplos, donde cada valor contiene el indice de la probabilidad mas alta, es decir la clase calculada o predecida"""
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

    def negative_log_likelihood(self, y):
        """
        Esta funcion es nuestro criterio de medida de que tan bien ha realizado el calculo la CNN,
        Checamos la probabilidad predecida para Y, despues sumamos todas las probabilidades y les sacamos el promedio de que tanto se equivoca
        Cost Function with mean and not sum
        :param y: es una coleccion de indices donde cada row representa un ejemplo y su valor el indice correcto
        :return:
        """
        listaindices = T.arange(y.shape[0]) #creamos una secuencia de 0 hasta el numero de de elementos en y

        resultLog = T.log(self.p_y_given_x)  #aplicamos la funcion log a las probabilidades predecidas por cada una de las posibles clases, Siempre resultara un numero negativo, si la probabilidad es muy baja dara un resultado mas negativo
        result =resultLog[listaindices, y]  #por cada row(ejemplo predecido) obtenemos la probabilidad de la clase correcta(y), si es correcta debe ser muy alta y si es incorrecta debe ser muy baja

        return T.mean(result) #Regresamos el promedio de las respuestas calculadas, si es muy alto significa que va bien por lo que queremos Maximizar el resultado

    def errors(self, y):
        """
        Regresa el promedio de errores, el resultado esta en el rango DE 0 a 1 donde 0 significa que no hubo error y 1 significa que en todos hubo error
        :param y:
        :return:
        """
        result = T.neq(self.y_pred, y)  # the T.neq operator returns a vector of 0s and 1s, where 1 represents a mistake in prediction

        return T.mean(result)
