__author__ = 'Gyo'
import Domain.LenetService as LenetService
import png
from PIL import Image
import numpy as np
import theano
import Experiments.ExperimentsRepo as ExperimentsRepo
import Weights.WeigthsService  as WeigthsService
import Domain.TypeInitWFunctionEnum as TypeInitWFunctionEnum
import os

id_experiment = 21
bd = '..\OCR.db'

experiment_repo = ExperimentsRepo.ExperimentsRepo(bd, id_experiment)
weigths_service = WeigthsService.WigthsService(bd, 'D:\\Gyo\\Dev\\OCR\\Weights', id_experiment,
                                               type_init_weigths_function=TypeInitWFunctionEnum.TypeInitWFunctionEnum.UniformDistribution)

w = weigths_service.LoadWeigths(663)
ls = LenetService.LenetService(
    batch_size=1,
    dataset_size=1,
    weigthts_service=weigths_service,
    initial_weights=w
)

im = Image.open(
    os.path.join('D:\\Gyo\\Dev\\GIT\\mnist_png', 'result', 'testing', '6', '54.png'))  # Can be many different formats.
# pix = im.load()
ari = np.asarray(list(im.getdata()), dtype=theano.config.floatX)

result = ls.Predict(ari)  # 784 Array

print 'OK'