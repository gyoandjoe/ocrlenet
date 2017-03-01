__author__ = 'win-g'
from Domain.LenetValidationAndTest import LenetValidationAndTest
from Weights import WeigthsService
import Experiments.ExperimentsRepo
import Domain.TypeInitWFunctionEnum as TypeInitWFunctionEnum
import Analisys.TiposDataSetEnum as TiposDataSetEnum

id_experiment=18
#id_weigths=630#663#672
bd = '..\OCR.db'
experiment_repo = Experiments.ExperimentsRepo.ExperimentsRepo(bd,id_experiment)
weigths_service = WeigthsService.WigthsService(bd,'D:\\Gyo\\Dev\\OCR\\Weights',id_experiment,TypeInitWFunctionEnum.TypeInitWFunctionEnum.UniformDistribution)
weigthsOfExperiment = weigths_service.GetListOfWeightsByIdExperiment(id_experiment)

print '--------------------------- VALIDATION SET -------------------------------------------------'

lvt = LenetValidationAndTest(
    dataset_file=experiment_repo.ObtenerArchivoDataSet(),
    batch_size=5000,
    dataset_size=10000,#50000,
    weigthts_service=weigths_service,
    tipoDataSet=TiposDataSetEnum.TiposDataSetEnum.validationSet
)

print "Calculando Errores en validationSet y costos en validation set"
for w in weigthsOfExperiment:
    averageError = lvt.CalculateError(id_weigths=w[0])
    weigths_service.UpdateValErrorWeigth(w[0], averageError)
    print "--------[Validation Set] El error promedio es: " + str(averageError)

    averageCost = lvt.CalculateCost(id_weigths=w[0])
    weigths_service.UpdateValCostWeigth(w[0], averageCost)
    print "--------[Validation Set] El costo promedio es: " + str(averageCost)

print '--------------------------- TEST SET -------------------------------------------------'

lvt = LenetValidationAndTest(
    dataset_file=experiment_repo.ObtenerArchivoDataSet(),
    batch_size=5000,
    dataset_size=10000,#50000,
    weigthts_service=weigths_service,
    tipoDataSet=TiposDataSetEnum.TiposDataSetEnum.testSet
)

print "Calculando Errores en test Set y costos en test set"
for w in weigthsOfExperiment:
    averageError = lvt.CalculateError(id_weigths=w[0])
    weigths_service.UpdateTestErrorWeigth(w[0], averageError)
    print "--------[Test Set] El error promedio es: " + str(averageError)

    averageCost = lvt.CalculateCost(id_weigths=w[0])
    weigths_service.UpdateTestCostWeigth(w[0], averageCost)
    print "--------[Test Set] El costo promedio es: " + str(averageCost)

print '--------------------------- TRAIN SET -------------------------------------------------'

lvt = LenetValidationAndTest(
    dataset_file=experiment_repo.ObtenerArchivoDataSet(),
    batch_size=5000,
    dataset_size=50000,
    weigthts_service=weigths_service,
    tipoDataSet=TiposDataSetEnum.TiposDataSetEnum.trainSet
)

#Actualizamos los costos en el dataset entero con cada uno de los pesos de que se han generado durante un determinado experimento
#Por cada conjunto de pesos obtenidos en el experimento, hacemos el calculo del costo y del error en el train set
print "Calculando errores en trainSet y costos en trainSet"
for w in weigthsOfExperiment:
    averageError = lvt.CalculateError(id_weigths=w[0])
    weigths_service.UpdateTrainErrorWeigth(w[0], averageError)
    print "--------[Train Set] El error promedio es: " + str(averageError)

    averageCost = lvt.CalculateCost(id_weigths=w[0])
    weigths_service.UpdateTrainCostWeigth(w[0], averageCost)
    print "--------[Train Set] El costo promedio es: " + str(averageCost)


print "End Validation :)"