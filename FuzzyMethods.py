
import numpy as np
import fuzzylab as fz
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def FuzzyUnion(mua, mub, tipo='max') -> np.array:
    union = np.zeros(len(mua))
    if tipo == 'max':
        for i in range(len(mua)):
            union[i] = max(mua[i], mub[i])
    elif tipo == 'sum':
        for i in range(len(mua)):
            union[i] = mua[i] + mub[i] - mua[i]*mub[i]
    elif tipo == 'bounded':
        for i in range(len(mua)):
            union[i] = min(1, mua[i]+mub[i])
    elif tipo == "drastic":
        for i in range(len(mua)):
            if mub[i] == 0:
                union[i] = mua[i]
            elif mua[i] == 0:
                union[i] = mub[i]
            else:
                union[i] = 1

    return union


def FuzzyIntersect(mua, mub, tipo='min') -> np.array:
    inter = np.zeros(len(mua))
    if tipo == 'min':
        for i in range(len(mua)):
            inter[i] = min(mua[i], mub[i])
    elif tipo == 'prod':
        for i in range(len(mua)):
            inter[i] = mua[i]*mub[i]
    elif tipo == 'bounded':
        for i in range(len(mua)):
            inter[i] = max(0, mua[i]+mub[i]-1)
    elif tipo == "drastic":
        for i in range(len(mua)):
            if mub[i] == 1:
                inter[i] = mua[i]
            elif mua[i] == 1:
                inter[i] = mub[i]
            else:
                inter[i] = 0
    return inter


def corteAlpha(mua, alpha):

    corte = np.zeros_like(mua)
    for i in range(len(mua)):
        # print(mua[i],alpha)
        corte[i] = max(0, min(mua[i], alpha))
    return corte


def implicacionMamdani(mua, alpha, tipo='min'):
    corte = np.zeros(len(mua))
    if tipo == 'min':
        corte = corteAlpha(mua, alpha)
    else:
        corte = mua.copy()
        for i in range(len(mua)):
            corte[i] = mua[i]*alpha
    return corte


def agregacionDifusa(arraymuB, tipo='max'):
    y = arraymuB[0].copy()
    for mub in arraymuB:
        y = FuzzyUnion(y, mub, tipo)
    return y


def plot_rules_activation(FIS, input):
    num_rules = len(FIS.Rules)
    num_activemf = len(FIS.Inputs)+len(FIS.Outputs)
    num_inputs = len(FIS.Inputs)
    fig, ax = plt.subplots(num_rules+1, num_activemf,sharey=True, figsize=(15, 5*num_rules))  # f,

    implicacion = FIS.ImplicationMethod  # min/prod = estrella #interseccion difusa
    agregacion = FIS.AggregationMethod  # max/sum = s-norma    #union difusa

    MUY = [[] for r in range(len(FIS.Outputs))]
    ODOMINIOS = []

    for i in range(num_rules):
        antecedentes = FIS.Rules[i].Antecedent
        consecuente = FIS.Rules[i].Consequent
        coneccion = FIS.Rules[i].Connection
        fuerza_disparo = -np.inf if coneccion == 2 else np.inf

        # antecedentes
        for j in range(len(antecedentes)):
            rango = FIS.Inputs[j].Range
            dominio = np.linspace(rango[0], rango[1], 100)

            mf = fz.evalmf(FIS.Inputs[j].MembershipFunctions[antecedentes[j]-1], dominio)
            crisp = input[j]
            id = antecedentes[j]-1
            
            # no hay entrada en la regla
            if id != -1:
                muX = fz.evalmf(FIS.Inputs[j].MembershipFunctions[id], crisp)
                ax[i, j].plot(dominio, mf, label="$\mu_A(x_"+str(j+1)+")$")
            else:
                muX = 0
                ax[i, j].plot(dominio, np.zeros_like(dominio),label="$\mu_A(x_"+str(j+1)+")$")
            ax[i, j].plot(crisp, muX, color='red', marker='o')
            ax[i, j].hlines(muX, rango[0], rango[1],color='red', linestyle='--')

            ax[i, j].set_title(FIS.Inputs[j].Name+"= "+FIS.Inputs[j].MembershipFunctions[id].Name)
            #ax[i, j].legend()
            if id == -1:
                continue
            # coneccion 1 = and(min) 2 = or
            if coneccion == 2:  # union (OR)
                fuerza_disparo = max(fuerza_disparo, muX)
            elif coneccion == 1:  # interseccion (AND)
                fuerza_disparo = min(fuerza_disparo, muX)
                
            # texto entre las graficas
            if j <len(antecedentes)-1: # el ultimo es para la implicación
                txt_coneccion = "AND\n(MIN)" if coneccion == 1 else "OR\n(MAX)" 
                offset = max(dominio)*0.2
                #              xy                     height            
                ax[i, j].text(max(dominio)+offset,0.5*(1),txt_coneccion,horizontalalignment='center',verticalalignment='center')

        # consecuentes
        for o in range(len(antecedentes), len(consecuente)+len(antecedentes)):
            # print(f"rule {o}")
            mfid = o-len(antecedentes)
            o_rango = FIS.Outputs[mfid].Range
            o_dominio = np.linspace(o_rango[0], o_rango[1], 100)
            ODOMINIOS.append(o_dominio)

            o_mf = fz.evalmf(FIS.Outputs[mfid].MembershipFunctions[consecuente[mfid]-1], o_dominio)

            # implicación -> intersección difusa
            muBy = implicacionMamdani(o_mf, np.squeeze(fuerza_disparo), tipo=implicacion)

            ax[i, o].plot(o_dominio, o_mf, color='green',linestyle='--', label="$\mu_B(y_"+str(mfid+1)+")$")
            ax[i, o].hlines(fuerza_disparo, o_rango[0],o_rango[1], color='red', linestyle='--')
            ax[i, o].fill_between(o_dominio, muBy, color='green',alpha=0.5, label="$\mu_{A\\rightarrow B}(x,y)$")
            #ax[i, o].title.set_text(f"Regla {i+1} - operador {'OR(Max)' if coneccion == 2 else 'AND(Min)'}")
            ax[i, o].set_title(FIS.Outputs[mfid].Name+"= "+FIS.Outputs[mfid].MembershipFunctions[consecuente[mfid]-1].Name)
            # ax[i, o].legend()
            
            # texto entre las graficas
            if o == len(antecedentes): 
                txt_coneccion = "$\mu_{A\\rightarrow B}(x,y)$\n("+FIS.ImplicationMethod+")" 
                offset = max(o_dominio)*0.25
                #              xy                     height            
                ax[i, o].text(min(o_dominio)-offset,0.5*(1),txt_coneccion,horizontalalignment='center',verticalalignment='center')

            MUY[mfid].append(muBy)

    # agregación ->  unión difusa
    for agg in range(len(MUY)):
        outputmf = agregacionDifusa(MUY[agg], tipo=agregacion)
        ax[num_rules, num_inputs +agg].plot(ODOMINIOS[agg], outputmf, color='orange')
        ax[num_rules, num_inputs+agg].fill_between(ODOMINIOS[agg], outputmf, color='orange', alpha=0.5, label="$\mu_B(y)$")

        # defusificar
        
        salida = fz.defuzz(ODOMINIOS[agg], outputmf, FIS.DefuzzificationMethod) 
        ax[num_rules, num_inputs+agg].vlines(salida, 0, 1, color='red', linestyle='solid', label=f"output={salida:.2f}")
        
        # ax[num_rules, num_inputs+agg].legend()
        ax[num_rules, num_inputs+agg].title.set_text(f"Agregación, Desfusificar\n({agregacion}),({FIS.DefuzzificationMethod})")
        ax[num_rules, num_inputs+agg].set_xlabel(FIS.Outputs[agg].Name+"= "+f"{salida:.3f}")

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":

    prueba = fz.mamfis("prueba")

    prueba.addInput([0, 1], Name="entrada1")
    prueba.addMF("entrada1", "gaussmf", [0.1, 0.5], Name="termino1")
    prueba.addMF("entrada1", "gaussmf", [0.1, 1], Name="termino2")
    prueba.addMF("entrada1", "gaussmf", [0.1, 0], Name="termino3")

    # fz.plotmf(prueba, "input", 1)

    prueba.addInput([0, 1], Name="entrada2")
    prueba.addMF("entrada2", "gaussmf", [0.1, 0.5], Name="termino2.1")
    prueba.addMF("entrada2", "gaussmf", [0.1, 1], Name="termino2.2")
    prueba.addMF("entrada2", "gaussmf", [0.1, 0], Name="termino2.3")

    # fz.plotmf(prueba, "input", 2)

    prueba.addOutput([0, 10], Name="salida1")
    prueba.addMF("salida1", "gaussmf", [1, 5], Name="termino_o1.1")
    prueba.addMF("salida1", "gaussmf", [1, 10], Name="termino_o1.2")
    prueba.addMF("salida1", "gaussmf", [1, 0], Name="termino_o1.3")

    # fz.plotmf(prueba, "output", 1)

    prueba.addOutput([0, 1], Name="salida2")
    prueba.addMF("salida2", "gaussmf", [0.1, 0.5], Name="termino_o2.1")
    prueba.addMF("salida2", "gaussmf", [0.1, 1], Name="termino_o2.2")
    prueba.addMF("salida2", "gaussmf", [0.1, 0], Name="termino_o2.3")

    # fz.plotmf(prueba, "output", 2)

    reglas = [
        # si entrada1 = 1 y entrada2 = 1 entonces salida1 = 1 y salida2 = 1
        [1, 1, 1, 1, 1, 1],
        # si entrada1 = 1 y entrada2 = 2 entonces salida1 = 1 y salida2 = 2
        [1, 2, 1, 2, 1, 1],
        # si entrada1 = 2 y entrada2 = 1 entonces salida1 = 2 y salida2 = 1
        [2, 1, 2, 1, 1, 1],
        # si entrada1 = 2 y entrada2 = 2 entonces salida1 = 2 y salida2 = 2
        [2, 2, 2, 2, 1, 1],
        # si entrada1 = 3 o entrada2 = 3 entonces salida1 = 1 y salida2 = 3
        [3,1,1,3,1,2]
    ]
    prueba.addRule(reglas)

    fz.writeFIS(prueba, "prueba.fis")

    plot_rules_activation(prueba, [0.3, 0.5])
    plot_rules_activation(prueba, [1, 0.6])
