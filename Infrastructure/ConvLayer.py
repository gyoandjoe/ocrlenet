# coding=utf-8
__author__ = 'win-g'

import theano
from theano.tensor.nnet import conv



class ConvLayer(object):
    def __init__(self, name, input, initial_filter_values, filter_shape, image_shape, stride=(1, 1)):
        self.Name = name
        self.Filter = theano.shared(
            value=initial_filter_values,
            name='FilterFCLayer'+self.Name,
        )

        self.Out = conv.conv2d(
            input=input,
            filters=self.Filter,
            subsample=stride,  # Stride
            filter_shape=filter_shape,
            image_shape=image_shape
        )
