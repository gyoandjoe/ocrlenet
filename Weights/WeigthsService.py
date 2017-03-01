import Utilities

__author__ = 'win-g'
import cPickle
from Domain.TypeInitWFunctionEnum import TypeInitWFunctionEnum
import WeigthsRepo
import numpy as np
import theano

class WigthsService(object):
    def __init__(self,database_name,folder_path, id_experiment,type_init_weigths_function):
        """
        :rtype : object
        """
        self.database_name = database_name
        self.id_experiment = id_experiment
        self.type_init_weigths_function = type_init_weigths_function
        self.repo = WeigthsRepo.WeigthsRepo(self.database_name,folder_path, id_experiment)
        self.paramsForInitWeigths = None
        return

    def SaveWeights(self, weights, epoch, batch, iteracion,hyper_params,cost=0,error=0,costVal=0,errorVal=0,costTest=0,errorTest=0):
        self.repo.SaveWeights(weights, epoch, batch, iteracion,hyper_params,cost,error,costVal,errorVal,costTest,errorTest)
        return

    def LoadWeigths(self, idWeigths):
        wf = self.repo.GetWeithsInfoById(idWeigths)
        if wf is not None:
            fileName =  wf[7] #FileName
            fLoaded = file(fileName, 'rb')
            data = cPickle.load(fLoaded)
            fLoaded.close()
            return data

        return

    def GetListOfWeightsByIdExperiment(self, id_experiment):
        return self.repo.GetGeigthsByExperimentId(id_experiment)


    def UpdateValCostWeigth(self, id_weigth, cost):
        self.repo.UpdateValCostForWeigth(id_weigth,cost)
        return

    def UpdateValErrorWeigth(self, id_weigth, error):
        self.repo.UpdateValErrorForWeigth(id_weigth,error)
        return


    def UpdateTestCostWeigth(self, id_weigth, cost):
        self.repo.UpdateTestCostForWeigth(id_weigth,cost)
        return

    def UpdateTestErrorWeigth(self, id_weigth, error):
        self.repo.UpdateTestErrorForWeigth(id_weigth,error)
        return


    def UpdateTrainCostWeigth(self, id_weigth, cost):
        self.repo.UpdateCostForWeigth(id_weigth,cost)
        return

    def UpdateTrainErrorWeigth(self, id_weigth, error):
        self.repo.UpdateErrorForWeigth(id_weigth,error)
        return

    def SetInitParamsForTypeInit(self, params):
        """
        :type lowAndHigh_c1Values: object
        """
        self.paramsForInitWeigths = params




    def GetWeigths(self, layer, shape):
        if self.type_init_weigths_function is TypeInitWFunctionEnum.UniformDistribution:
            if layer == 'c1Values':
                lowAndHigh_c1Values = self.paramsForInitWeigths['lowAndHigh_c1Values']
                return Utilities.Generate_uniform_distributionValues(lowAndHigh_c1Values[0],lowAndHigh_c1Values[1],shape)
            elif layer == 'c3Values':
                lowAndHigh_c3Values = self.paramsForInitWeigths['lowAndHigh_c3Values']
                return Utilities.Generate_uniform_distributionValues(lowAndHigh_c3Values[0],lowAndHigh_c3Values[1],shape)
            elif layer == 'fc5Values':
                lowAndHigh_fc5Values = self.paramsForInitWeigths['lowAndHigh_fc5Values']
                return Utilities.Generate_uniform_distributionValues(lowAndHigh_fc5Values[0],lowAndHigh_fc5Values[1],shape)
            elif layer == 'fc6Values' or layer == 'fc6BiasValues' or layer == 'fc5BiasValues':
                #lowAndHigh_fc6Values = self.paramsForInitWeigths['lowAndHigh_fc6Values']
                #return Utilities.Generate_uniform_distributionValues(lowAndHigh_fc6Values[0],lowAndHigh_fc6Values[1],shape)
                return np.zeros(shape, dtype=theano.config.floatX)
        return None

    def GetInitParamsWeigthsInStringFormat(self):
        niceFormat = ""
        if self.type_init_weigths_function is TypeInitWFunctionEnum.UniformDistribution:
            lowAndHigh_c1Values = self.paramsForInitWeigths['lowAndHigh_c1Values']
            niceFormat  = "{UniformDistribution: {lowAndHigh_c1Values : { lowValue: " + str(lowAndHigh_c1Values[0]) + " highValue: " + str(lowAndHigh_c1Values[1]) + "},"
            lowAndHigh_c3Values = self.paramsForInitWeigths['lowAndHigh_c3Values']
            niceFormat += "lowAndHigh_c3Values : { lowValue: " +  str(lowAndHigh_c3Values[0]) + " highValue: " + str(lowAndHigh_c3Values[1]) + "},"
            lowAndHigh_fc5Values = self.paramsForInitWeigths['lowAndHigh_fc5Values']
            niceFormat += "lowAndHigh_fc5Values : { lowValue: " +  str(lowAndHigh_fc5Values[0]) + " highValue: " + str(lowAndHigh_fc5Values[1]) + "},"
            niceFormat += "fc6Values : All Zeros,"
            niceFormat += "fc6BiasValues : All Zeros,"
            niceFormat += "fc5BiasValues  : All Zeros}}"

        return niceFormat