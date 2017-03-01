__author__ = 'win-g'
import numpy as np
def Generate_normal_distributionValues(shape, initMean = 0, initSD = 0.001):
    numberWeights = np.prod(shape)
    normalDistributionValues = np.random.normal(initMean, initSD, numberWeights)
    return normalDistributionValues.reshape(shape)


def Generate_uniform_distributionValues(low, high, shape):
    numberWeights = np.prod(shape)
    uniformDistributionValues = np.random.uniform(low, high, numberWeights)
    return uniformDistributionValues.reshape(shape)