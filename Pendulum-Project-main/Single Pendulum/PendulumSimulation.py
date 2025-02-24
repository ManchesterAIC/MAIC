from pendulumNet import PendulumNet
from SinglePendulum import Pendulum
import math
import tkinter as tk
import pickle
import time

def simulate(net):
    pendulum = Pendulum()
    score = 0

    for _ in range(10000):
        pendulum.bx__ = net.forward([pendulum.theta_, pendulum.theta])
        pendulum.update()
        
        if pendulum.y < pendulum.by - 98:
            score += 1
    return score

def load_networks():
    with open(f"networks.pkl", "rb") as file:
        return pickle.load(file)

def simulateGeneration():
    initial = load_networks()

    offspring = []
    for brain in initial:
        for i in range(5):
            offspring.append(brain.reproduce())
    
    avg_score = 0
    for i in range(len(offspring)):
        total = 0
        Iter = 1
        for _ in range(Iter):
            total += simulate(offspring[i])
        total /= Iter

        offspring[i] = [total, offspring[i]]
        avg_score += total

    avg_score /= len(offspring)
    print(avg_score)

    offspring.sort(reverse= True, key = lambda x: x[0])
    offspring = offspring[:5]
    offspring = [child[1] for child in offspring]

    with open("networks.pkl", "wb") as file:
        pickle.dump(offspring, file)


def displayGeneration():
    net = load_networks()[0]

    if True:
        window = tk.Tk()
        canvas = tk.Canvas(window, bg = 'black', highlightthickness= 0)
        screen_width = 1000
        screen_height = 500
        canvas.config(width = screen_width, height = screen_height)
        canvas.pack()

        p = Pendulum()

        line = canvas.create_line(p.bx, p.by, p.x, p.y, fill = 'white', width = 2)
        point = canvas.create_oval(p.x - 10, p.y - 10, p.x + 10, p.y + 10, fill='white')
        base = canvas.create_line(p.bx-20, p.by, p.bx+20, p.by, fill = 'gray', width = 5)

    def game_loop():
        p.bx__ = net.forward([p.theta_, p.theta])
        p.update()

        canvas.coords(line, [p.bx, p.by, p.x, p.y])
        canvas.coords(point, [p.x - 10, p.y - 10, p.x + 10, p.y + 10])
        canvas.coords(base, [p.bx-20, p.by, p.bx+20, p.by])
        canvas.after(2, game_loop)
    
    game_loop()
    window.mainloop()

displayGeneration()