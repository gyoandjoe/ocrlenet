__author__ = 'win-g'
import sqlite3



conn = sqlite3.connect('OCR.db')

c = conn.cursor()




#c.execute("INSERT INTO Experiments VALUES (NULL ,'C:\\Users\\win-g\\PycharmProjects\\OCR\\DataSet\\mnist.pkl',50000,5000,0.0001, 'Pendiente',0)")
c.execute('''CREATE TABLE Experiments
             (Id INTEGER PRIMARY KEY, DataSetFile text, SizeDataSet text, BatchSize real, LearningRate real, Status text, BatchActual integer,MaxEpoch INTEGER,FrecuencySaveWeigths INTEGER,WithLRDecay INTEGER, FrecuencyLRDecay INTEGER  )''')

c.execute('''CREATE TABLE Weights
             (Id INTEGER PRIMARY KEY, IdExperiment integer, Fecha text, Epoch integer, Batch integer, Iteracion integer, Cost real, FileName text,HyperParams text, Error real,CostVal real,ErrorVal real,CostTest real,ErrorTest real)''')
#['Id','IdExperiment','Fecha','Epoch','Batch','Iteration','Cost', 'FileName' ,'HyperParams','Error','CostVal','ErrorVal','CostTest','ErrorTest']


c.execute('''CREATE TABLE LogExperiment
             (Id INTEGER PRIMARY KEY, IdExperient integer, Fecha text, Contenido text,  TipoLog text, EpochIndex integer, BatchIndex integer,ExtraInfo text,Referencia text)''')
#LogExperiment ['Id','IdExperient','Fecha','Contenido','TipoLog','EpochIndex','BatchIndex','ExtraInfo','Referencia']

c.execute('''CREATE TABLE LearningCurveAnalysisXNoExp (Id INTEGER PRIMARY KEY, IdExperiment integer, IdWeigths INTEGER)''')

c.execute('''CREATE TABLE LearningCurveXNoExamp (Id INTEGER PRIMARY KEY, NoExperiments INTEGER, Cost REAL , Error REAL, TipoDataSet TEXT, DataSetSize INTEGER,IdLearningCurveAnalysis integer)''')
#LearningCurveXNoExamp ['Id','NoExperiments','Cost','Error','TipoDataSet','DataSetSize','IdLearningCurveAnalysis']

#Agregar intentoId, batch, iteracion

# Save (commit) the changes
conn.commit()

# We can also close the connection if we are done with it.
# Just be sure any changes have been committed or they will be lost.
conn.close()


