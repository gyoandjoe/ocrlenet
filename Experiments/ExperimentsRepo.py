__author__ = 'win-g'
import sqlite3

class ExperimentsRepo(object):
    def __init__(self,database_name, id_experiment):
        self.database_name = database_name
        self.experiment = self.BuscarExperimento(id_experiment)
        self.id_experiment =id_experiment

    def BuscarExperimento(self, id_experimento):
        conn = sqlite3.connect(self.database_name)
        c = conn.cursor()
        id_ex = str(id_experimento)
        c.execute("select * from Experiments where Id = ? ",[id_ex])
        registro= c.fetchone()
        return registro

    def ObtenerArchivoDataSet(self):
        if self.experiment is None:
            return None
        return self.experiment[1]

    def ObtenerSizeDataSet(self):
        if self.experiment is None:
            return None
        return self.experiment[2]

    def ObtenerBatchSize(self):
        if self.experiment is None:
            return None
        return self.experiment[3]

    def ObtenerLearningRate(self):
        if self.experiment is None:
            return None
        return self.experiment[4]

    def ObtenerStatus(self):
        if self.experiment is None:
            return None
        return self.experiment[5]

    def ObtenerBatchActual(self):
        if self.experiment is None:
            return None
        return self.experiment[6]

    def ObtenerMaxEpoch(self):
        if self.experiment is None:
            return None
        return self.experiment[7]

    def ObtenerFrecuencySaveWeigths(self):
        if self.experiment is None:
            return None
        return self.experiment[8]

    def ObtenerWithLRDecay(self):
        if self.experiment is None:
            return None
        return self.experiment[9]

    def ObtenerFrecuencyLRDecay(self):
         if self.experiment is None:
            return None
         return self.experiment[10]

    def SetFalseDecreaseNow(self):
        conn = sqlite3.connect(self.database_name)
        c = conn.cursor()
        query = "UPDATE Experiments SET DeseaseNow = {0} WHERE Id = {1}".format('False', self.id_experiment)

        c.execute(query)
        conn.commit()
        conn.close()

    def ObtenerDecreaseNow(self):
        if self.experiment is None:
            return None
        if self.experiment[11] == 'True':
            return True
        return False



