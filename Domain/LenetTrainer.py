__author__ = 'win-g'
import cPickle

import theano
import numpy as np
import pickle
from Domain import LenetCNN
from Domain import LayerEnum

class LenetTrainer(object):
    def __init__(self,learning_rate, with_lr_decay,dataset_file, batch_size, dataset_size, logger, max_epochs,weigthts_service, experimentsRepo, saveWeigthsFrecuency=10,initial_weights=None,frecuency_lr_decay =2):
        self.initial_weights=initial_weights
        self.dataset_size = int(dataset_size)
        self.batch_size = int(batch_size)
        self.logger = logger
        self.max_epochs = max_epochs
        self.learning_rate = float( learning_rate)
        self.weigthts_service = weigthts_service
        self.saveWeigthsFrecuency = saveWeigthsFrecuency
        self.with_lr_decay = with_lr_decay
        self.frecuency_lr_decay = frecuency_lr_decay
        fLoaded = file(dataset_file, 'rb')
        rawData = cPickle.load(fLoaded)
        rawTrainingDataSet = rawData[0]
        rawTestDataSet = rawData[1]
        rawValidationDataSet = rawData[2]
        self.experimentsRepo = experimentsRepo

        self.no_batchs =  self.dataset_size / self.batch_size

        XimgLetras = np.asarray(rawTrainingDataSet[0], dtype=theano.config.floatX).reshape((self.dataset_size, 1, 28, 28))
        YimgLetras = np.asarray(rawTrainingDataSet[1])
        self.lenetcnn = LenetCNN.LenetCNN(
            batch_size= self.batch_size,
            train_set=  (XimgLetras, YimgLetras),
            initial_weights=initial_weights,
            weigths_service=self.weigthts_service
          )


    def Train(self,current_epoch=0,id_train='',extra_info='',):

        if self.initial_weights is None: #Significa que no ecisten pesos predeterminados por lo tanto deben inicializarse por la funcion de weigthsService
            self.SaveWeights(-1,-1,-1,-1)
            initParamsWeigthsInStringFormat = self.weigthts_service.GetInitParamsWeigthsInStringFormat()
            print("Parametros de inicio:" + initParamsWeigthsInStringFormat)
            self.logger.Log("initParamsForInitWeigths" + initParamsWeigthsInStringFormat + " learning rate: " +str(self.learning_rate) + ","+extra_info, "InicioEntrenamiento", current_epoch, -1)


        for epoch_index in xrange(self.max_epochs):
            if self.initial_weights is not None and epoch_index < current_epoch: #hacemos esta verificacion pues solo tiene sentido iniciar en una epoca diferente cuando existen pesos iniciales (para reanudar)
                continue

            if epoch_index != 0 and self.with_lr_decay == True and epoch_index % self.frecuency_lr_decay == 0:
               self.learning_rate *= 0.1
            elif self.with_lr_decay == False:
                decreaseNow = self.experimentsRepo.ObtenerDecreaseNow()
                if decreaseNow == True:
                    self.experimentsRepo.SetFalseDecreaseNow()
                    self.learning_rate *= 0.1


            for batch_index in xrange(self.no_batchs):
                cost = self.lenetcnn.train_model(batch_index,self.learning_rate)
                print "costo: "+str(cost)+" epoca: " + str(epoch_index)
                self.logger.Log(str(cost), "costo",str(epoch_index), str(batch_index),id_train,"learning rate: " +str(self.learning_rate) + ","+extra_info)
            if (epoch_index + 1) % self.saveWeigthsFrecuency == 0:
                self.SaveWeights(epoch_index,batch_index,-1)


    def SaveWeights(self, epoch, batch, iteration, cost=0,error=0,costVal=0,errorVal=0,costTest=0,errorTest=0):
        allWeiths = {
            "c1Values": self.lenetcnn.GetWeigthsValuesByLayer(LayerEnum.LayerEnum.c1Values),
            "c3Values":self.lenetcnn.GetWeigthsValuesByLayer(LayerEnum.LayerEnum.c3Values),
            "fc5Values": self.lenetcnn.GetWeigthsValuesByLayer(LayerEnum.LayerEnum.fc5Values),
            "fc5BiasValues": self.lenetcnn.GetWeigthsValuesByLayer(LayerEnum.LayerEnum.fc5BiasValues),
            "fc6Values":self.lenetcnn.GetWeigthsValuesByLayer(LayerEnum.LayerEnum.fc6Values),
            "fc6BiasValues": self.lenetcnn.GetWeigthsValuesByLayer(LayerEnum.LayerEnum.fc6BiasValues)
        }

        hyper_params = "learning rate: " + str(self.learning_rate)
        # = [c1_v,c3_v,fc5v_v,fc5b_v,fc6v,fc6b_v]
        self.weigthts_service.SaveWeights(allWeiths,epoch, batch, iteration,hyper_params,cost,error,costVal,errorVal,costTest,errorTest)
        return

