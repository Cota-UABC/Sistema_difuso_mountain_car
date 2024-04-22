import gymnasium as gym
import fuzzylab as fz
import matplotlib.pyplot as plt


"""
Entradas del entorno (observaciones)
0: position of the car along the x-axi [-1.2, 0.6]
1: velocity of the car [-0.07, 0.07]

Salidas del entorno
0: Accelerate to the left
1: Dont accelerate
2: Accelerate to the right

End
fin: The position of the car is greater than or equal to 0.5 (the goal position on top of the right hill)
truncado: The length of the episode is 200.
"""

if __name__ == "__main__":
    entorno = gym.make('MountainCar-v0', render_mode='human')

    mountainfis = fz.readfis("mountainFIS.fis")

    rewards = []

    initial_pos_array = []

    for i in range(5):
        observaciones, info = entorno.reset()

        max_reward = 0
        stop = False

        steps = 0

        stale = 0

        pos, vel = observaciones

        initial_pos_array.append(pos)

        while (not stop):
            pos, vel = observaciones

            print(f" Posicion: {pos:.4f} Velocidad: {vel:.4f} ", end=" ")

            #evaluar fis
            muB = fz.evalfis(mountainfis, [pos, vel])

            accion = round(muB)
            #accion = 2
            print(f" Accion: {accion}")

            #evitar que no haga nada el idiota
            if(accion == 1):
                stale += 1
            else:
                stale = 0

            observaciones, recompensa, fin, truncado, info = entorno.step(accion)

            max_reward += recompensa

            steps += 1

            if fin or steps==350 or stale == 60:
                stop = True
                if stale == 60:
                    max_reward = -500

        print(f"Episidio: {i+1} Recompensa {max_reward}")
        rewards.append(max_reward)

    entorno.close()
    print(f"Reward maximo: {max(rewards)}")
    #plt.plot(rewards, color="green", linestyle="solid", marker="o")
    #plt.show()

    fig, axs = plt.subplots(2, 1, figsize=(8, 6))
    axs[0].plot(rewards, color="green", linestyle="solid", marker="o")
    axs[0].set_title('Rewards') 
    axs[0].grid(True)  
    
    axs[1].plot(initial_pos_array, color="red", linestyle="solid", marker="o")
    axs[1].set_title('Initial Position')
    axs[1].grid(True) 

    plt.tight_layout()
    plt.show()



