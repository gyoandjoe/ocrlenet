__author__ = 'Gyo'

import Analisys.Analizador as Analizador

analyzer = Analizador.Analizador('..\OCR.db')
analyzer.GraficarLCXErrors(1)
#analyzer.GraficarLCXCosts(1)
print "OK"
