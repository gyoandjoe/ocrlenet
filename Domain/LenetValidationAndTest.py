from Domain.LenetCNN import LenetCNN
import cPickle
import theano
import numpy as np
import Analisys.TiposDataSetEnum as TiposDataSetEnum
__author__ = 'win-g'
class LenetValidationAndTest(object):
    def __init__(self,dataset_file,batch_size, dataset_size,weigthts_service, tipoDataSet):
        self.weigthts_service = weigthts_service
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.no_batchs = self.dataset_size // self.batch_size

        fLoaded = file(dataset_file, 'rb')
        rawData = cPickle.load(fLoaded)
        self.rawTrainingDataSet = rawData[0]
        self.rawTestDataSet = rawData[1]
        self.rawValidationDataSet = rawData[2]
        self.no_batchs = self.dataset_size / self.batch_size

        if (tipoDataSet is TiposDataSetEnum.TiposDataSetEnum.trainSet):
            self.XimgLetras = np.asarray(self.rawTrainingDataSet[0], dtype=theano.config.floatX).reshape((self.dataset_size, 1, 28, 28))
            self.YimgLetras = np.asarray(self.rawTrainingDataSet[1])
        elif (tipoDataSet is TiposDataSetEnum.TiposDataSetEnum.testSet):
            self.XimgLetras = np.asarray(self.rawTestDataSet[0], dtype=theano.config.floatX).reshape((self.dataset_size, 1, 28, 28))
            self.YimgLetras = np.asarray(self.rawTestDataSet[1])
        elif (tipoDataSet is TiposDataSetEnum.TiposDataSetEnum.validationSet):
            self.XimgLetras = np.asarray(self.rawValidationDataSet[0], dtype=theano.config.floatX).reshape((self.dataset_size, 1, 28, 28))
            self.YimgLetras = np.asarray(self.rawValidationDataSet[1])

    def CalculateCost(self,id_weigths,noBatchsToEvaluate = -1):
        weigths = self.weigthts_service.LoadWeigths(idWeigths=id_weigths)
        lenetcnn = LenetCNN(batch_size= self.batch_size,train_set=  (self.XimgLetras, self.YimgLetras),initial_weights=weigths)

        if noBatchsToEvaluate == -1:
           noBatchsToEvaluate = self.no_batchs
        sumaCost = 0.0
        for batch_index in xrange(noBatchsToEvaluate):
            cost = lenetcnn.evaluation_model_with_cost(batch_index)
            print "calculando costos: costo: "+str(cost)+" en batch: " + str(batch_index)
            sumaCost=sumaCost + cost
        promedio = sumaCost / noBatchsToEvaluate
        return promedio

    def CalculateError(self,id_weigths,noBatchsToEvaluate = -1):
        weigths = self.weigthts_service.LoadWeigths(idWeigths=id_weigths)
        lenetcnn = LenetCNN(batch_size= self.batch_size,train_set=  (self.XimgLetras, self.YimgLetras),initial_weights=weigths)

        if noBatchsToEvaluate == -1:
           noBatchsToEvaluate = self.no_batchs

        sumaCost = 0.0
        for batch_index in xrange(noBatchsToEvaluate):
            cost = lenetcnn.evaluation_model_with_errors(batch_index)
            print "calculando costos: errores: "+str(cost)+" en batch: " + str(batch_index)
            sumaCost=sumaCost + cost
        promedio = sumaCost / noBatchsToEvaluate
        return promedio

