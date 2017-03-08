import datetime

__author__ = 'win-g'
import sqlite3


class Logger(object):
    def __init__(self, id_experiment, database_name):
        self.Id_Experiment = id_experiment
        self.Database_Name = database_name

    def Log(self, contenido, tipo_log, epoch_index,batch_index,referencia='',extra_info=''):
        conn = sqlite3.connect(self.Database_Name)
        c = conn.cursor()
        query = "INSERT INTO LogExperiment VALUES (NULL,{0},\'{1}\',\'{2}\',\'{3}\',{4},{5},\'{6}\',\'{7}\')".format(str(self.Id_Experiment),datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),contenido,tipo_log,epoch_index,batch_index,extra_info,referencia)
        c.execute(query)

        # Save (commit) the changes
        conn.commit()

        # We can also close the connection if we are done with it.
        # Just be sure any changes have been committed or they will be lost.
        conn.close()
