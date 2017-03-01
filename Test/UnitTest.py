__author__ = 'win-g'
import cPickle
import os
import theano
import numpy as np

fan_in = 1*3*3
fan_out = 6*3*3

result1 = np.sqrt(6. / (fan_in + fan_out))

fan_in = 6*3*3
fan_out = 16*3*3

resul2 = np.sqrt(6. / (fan_in + fan_out))

fname = "Joe"
lname = "Who"
age = "24"
nuevo = 'w\'ow \'%s\'' % (age)
otro = repr(nuevo)
print otro
print nuevo
batch_size = 10000
fLoaded = file(os.path.join("D:\\Gyo\\Dev\\PycharmProjects\\OCR\\DataSet\\", "mnist.pkl"), 'rb')
rawData = cPickle.load(fLoaded)
rawTrainingDataSet = rawData[0]
rawTestDataSet = rawData[1]
rawValidationDataSet = rawData[2]
XimgLetras = np.asarray(rawTrainingDataSet[0][0:batch_size], dtype=theano.config.floatX)
YimgLetras = np.asarray(rawTrainingDataSet[1][0:batch_size])
dataset = XimgLetras.reshape((batch_size, 1, 28, 28))
print "finish"