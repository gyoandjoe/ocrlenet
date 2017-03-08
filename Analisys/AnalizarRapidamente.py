import Analizador

__author__ = 'win-g'
"""
AnalizarRapidamente esta pensado para graficar de manera rapida los costos promedio de cada epoca con sus pesos en ese momento,
es decir, los pesos en cada iteracion dentro de la epoca son diferentes, No se grafica el costo obtenido con unos pesos en particular sino por cada
epoca el el costo promedio de sus iteraciones con sus respectivos pesos diferentes en cada iteracion
"""
analizador = Analizador.Analizador('../OCR.db')
#analizador.Iniciar(20)
analizador.Iniciar(4)
print "Fin analisis :)"