import fuzzylab as fz
import matplotlib.pyplot as plt
import numpy as np

mountain = fz.mamfis("Mountain")

mountain.AggregationMethod = "max"          # "sum" o union difusa (s-norma)
mountain.AndMethod = "min"
mountain.OrMethod = "max"
mountain.ImplicationMethod = "min"          #  "prod" o interseccion difusa (t-norma)
mountain.DefuzzificationMethod = "lom" #   "bisector"  "mom"  "som"  "lom" "wtaver"

mountain.addInput([-1.2,0.6],Name="Position")
mountain.addMF("Position","gaussmf",[.24, -1.2],Name="Inclinacion_derecha")
mountain.addMF("Position","gaussmf",[.34, 0.6],Name="Inclinacion_izquierda")
mountain.addMF("Position","gaussmf",[.06, -0.5],Name="Poca_inclinacion")
mountain.addMF("Position","gaussmf",[.04, -0.54],Name="Poca_der")
mountain.addMF("Position","gaussmf",[.04, -0.46],Name="Poca_iz")

fz.plotmf(mountain,"input",1)

mountain.addInput([-0.07,0.07],Name="Velocity")
mountain.addMF("Velocity","gaussmf",[.025, -0.07],Name="Izquierda")
mountain.addMF("Velocity","gaussmf",[.025, 0.07],Name="Derecha")
mountain.addMF("Velocity","gaussmf",[.002, 0],Name="Min_velocity")
mountain.addMF("Velocity","gaussmf",[.003, -0.007],Name="Poca_vel_izquierda")
mountain.addMF("Velocity","gaussmf",[.003, 0.007],Name="Poca_vel_derecha")

#fz.plotmf(mountain,"input",2)

mountain.addOutput([0,2],Name="Action")
mountain.addMF("Action","trimf",[0,0,.5],Name="Izquierda")
mountain.addMF("Action","trimf",[.5,1,1.5],Name="Nothing")
mountain.addMF("Action","trimf",[1.5,2,2],Name="Derecha")

#fz.plotmf(mountain, "output",1)

mountain_rules = [
    #pos  vel  Act  peso operador
    [1,   1,   2,   1,   1], #si Position = inclin_der AND Velocity = izquierda Then Action = nothing
    [1,   2,   3,   1,   1], #si Position = inclin_der AND Velocity = derecha Then Action = derecha
    
    [2,   1,   2,   1,   1], #si Position = inclin_izq AND Velocity = izquierda Then Action = nothing
    [2,   2,   3,   1,   1], #si Position = inclin_izq AND Velocity = derecha Then Action = derecha
    
    [3,   4,   1,   1,   1], #si Position = inclin_poca AND Velocity = poca_izq Then Action = izquierda
    [3,   5,   3,   1,   1], #si Position = inclin_poca AND Velocity = poca_der Then Action = derecha

    [3,   3,   3,   1,   1], #si Position = inclin_poca AND Velocity = min Then Action = derecha

    [4,   3,   2,   1,   1], #si Position = poca_der AND Velocity = min Then Action = nothing
    [5,   3,   2,   1,   1], #si Position = poca_izq AND Velocity = min Then Action = nothing
]

mountain.addRule(mountain_rules)

fz.writeFIS(mountain,"mountainFIS.fis")

from FuzzyMethods import *

plot_rules_activation(mountain, [-0.55,-0.0013])