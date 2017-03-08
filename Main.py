from Domain.TypeInitWFunctionEnum import TypeInitWFunctionEnum
from Domain.LenetTrainer import LenetTrainer
import Logger
import Experiments.ExperimentsRepo
from Weights import WeigthsService

__author__ = 'win-g'

id_experiment = 5
bd = 'OCR.db'
logger = Logger.Logger(id_experiment,bd)
experiment_repo= Experiments.ExperimentsRepo.ExperimentsRepo(bd,id_experiment)
weigths_service = WeigthsService.WigthsService(bd,'D:\\Gyo\\Dev\\OCR\\Weights',id_experiment, type_init_weigths_function=TypeInitWFunctionEnum.UniformDistribution)
weigths_service.SetInitParamsForTypeInit(params={
            "lowAndHigh_c1Values": [-0.2,0.2],
            "lowAndHigh_c3Values": [-0.08,0.08],
            "lowAndHigh_fc5Values": [-0.06,0.06],
            "lowAndHigh_fc6Values": [-0.001,0.001]
        })

#generamos los pesos iniciales



lt = LenetTrainer(
    dataset_file=experiment_repo.ObtenerArchivoDataSet(),
    batch_size= experiment_repo.ObtenerBatchSize(),
    dataset_size= experiment_repo.ObtenerSizeDataSet(),
    logger=logger,
    max_epochs= experiment_repo.ObtenerMaxEpoch(),
    weigthts_service = weigths_service,
    experimentsRepo = experiment_repo,
    initial_weights = None, #weigths_service.LoadWeigths(25),  # weigths_service.LoadWeigths(9), #None,
    saveWeigthsFrecuency=experiment_repo.ObtenerFrecuencySaveWeigths(),
    learning_rate= experiment_repo.ObtenerLearningRate(),
    with_lr_decay = experiment_repo.ObtenerWithLRDecay(),
    frecuency_lr_decay=experiment_repo.ObtenerFrecuencyLRDecay()
    )


lt.Train(
    current_epoch= 0,#0, #Este parametro solo tiene sentido cuando existen initial_weights
    id_train='learning rate automatico :)',
    extra_info='(+)'
)

print "OK"