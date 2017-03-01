__author__ = 'win-g'

import sqlite3



conn = sqlite3.connect('..\OCR.db')

c = conn.cursor()

maxEpoch = 500
frecuencySaveWeigths = 1
withLRDecay = 1
frecuencyLRDecay = 5

c.execute("INSERT INTO Experiments VALUES (NULL ,'C:\\Users\\win-g\\PycharmProjects\\OCR\\DataSet\\mnist.pkl',50000,5000,0.001, 'Pendiente',0,maxEpoch,frecuencySaveWeigths,withLRDecay,frecuencyLRDecay)")

conn.commit()

# We can also close the connection if we are done with it.
# Just be sure any changes have been committed or they will be lost.
conn.close()
