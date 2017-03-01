# coding=utf-8
__author__ = 'win-g'



import theano
import theano.tensor as T
import numpy as np
from theano.tensor.signal import pool
import Infrastructure.FCLayer as fcl
import Infrastructure.ConvLayer as convl
import Infrastructure.LogisticRegressionLayer as logl
import Utilities

class OCRLenetArquitecture(object):
    def __init__(self, img_input, batch_size, initWeights=None,c1NoFilters = 20, c3NoFilters = 50, weigths_service =None):
        """
        Layer No 1
        ConvolutionalLayer No 1
        OutputSize=((28-5+2*0)/1)+1 = 24
        OutPut Shape 10000x20x24x24
        """

        c1FilterShape = (c1NoFilters, 1, 5, 5)

        if initWeights is None or initWeights['c1Values'] is None:
            c1Values = weigths_service.GetWeigths('c1Values', c1FilterShape)
        else:
            c1Values = initWeights['c1Values']

        c1ImageShape=(batch_size, 1, 28, 28)
        self.C1 = convl.ConvLayer('c1ConvLayer',img_input,c1Values,c1FilterShape,c1ImageShape)

        """
        Layer No 2
        Subampling Layer 1
        Output shape 10000x20x12x12
        """
        self.S2 = pool.pool_2d(
            input=self.C1.Out,
            st=(2, 2), #stride
            ds=(2, 2),
            mode='max',
            ignore_border=True
        )


        """
        Layer No 3
        ConvolutionalLayer No 2
        w=((12-5+2*0)/1)+1 = 8
        OutPut Shape 10000x50x8x8
        """

        c3FilterShape = (c3NoFilters, c1NoFilters, 5, 5)

        if initWeights is None or initWeights['c3Values'] is None:
            c3Values = weigths_service.GetWeigths('c3Values',c3FilterShape)
        else:
            c3Values = initWeights['c3Values']

        c3ImageShape = (batch_size, c1NoFilters, 12, 12)
        self.C3 = convl.ConvLayer('c3ConvLayer', self.S2,c3Values,c3FilterShape,c3ImageShape)

        """
        Layer No 4
        Subampling Layer 2
        Output shape 10000x50x4x4
        """
        self.S4 = pool.pool_2d(
            input=self.C3.Out,
            st=(2, 2),  # stride, number of pixels to take the next area
            ds=(2, 2),
            mode='max',
            ignore_border=True
        )


        """
        Layer No 5
        Full connected Layer 1
        Output shape 10000x500
        """
        fc5FilterShape = (c3NoFilters*4*4, 500) #800 x 500

        if initWeights is None or initWeights['fc5Values'] is None:
            fc5InitialValues = weigths_service.GetWeigths('fc5Values',fc5FilterShape)

        else:
            fc5InitialValues = initWeights['fc5Values']

        if initWeights is None or initWeights['fc5BiasValues'] is None:
            fc5InitialBiasValues = weigths_service.GetWeigths('fc5BiasValues',(500))
        else:
            fc5InitialBiasValues = initWeights['fc5BiasValues']

        fc5input = self.S4.flatten(2) #10000 X 800
        self.FC5 = fcl.FCLayer(
            name= "FCLayer",
            input=fc5input,
            initial_filter_values = fc5InitialValues,
            initialbiasvalues =fc5InitialBiasValues,
            activationFunction=T.nnet.sigmoid
        )


        """
        Layer No 6 classify the values of the fully-connected sigmoidal layer
        Logistic regression
        Output shape 1000 X 10
        """

        fc6FilterShape = (500, 10) #800 x 500

        if initWeights is None or initWeights['fc6Values'] is None:
            fc6InitialValues = weigths_service.GetWeigths('fc6Values',fc6FilterShape)
        else:
            fc6InitialValues = initWeights['fc6Values']

        if initWeights is None or initWeights['fc6BiasValues'] is None:
            fc6InitialBiasValues = weigths_service.GetWeigths('fc6BiasValues',(10))
        else:
            fc6InitialBiasValues = initWeights['fc6BiasValues']

        self.LR6 = logl.LogisticRegressionLayer(
            name="LogisticRegLayer",
            input=self.FC5.output,
            initial_filter_values= fc6InitialValues,
            initialbiasvalues=fc6InitialBiasValues
        )