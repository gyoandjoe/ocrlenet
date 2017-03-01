__author__ = 'Gyo'
import Analisys.Analizador as Analizador


analyzer = Analizador.Analizador('..\OCR.db')

analyzer.BuildLearningCurveAnalysisByExamples(
    id_experiment=21,
    id_Analisys=1,
    bd='..\OCR.db',
    id_weigths=672,
    folderWeigths='D:\\Gyo\\Dev\\OCR\\Weights'
)
#id_experiment, id_Analisys,bd,id_weigths, folderWeigths):

print "OK"