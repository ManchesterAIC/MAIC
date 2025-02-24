import tkinter as tk
import math
import matplotlib.pyplot as plt
import random

#constants
g = 9.81
m1 = 1
m2 = 1
L1 = 100
L2 = 100

#angles defining the system

#0.000000000000000309 - min deviation 
th1 = 1.5*math.pi/2
th2 = math.pi

#w1 = th1'. w2 = th2'
w1 = 0
w2 = 0

dt = 0.025

#define 2 differential equations to describe the system. 
def fun1(th1, th2, w1, w2):
    #o1 = w1'
    r = -g*(2*m1 + m2)*math.sin(th1)
    r -= m2*g*math.sin(th1 - 2*th2)
    r -= 2*m2*math.sin(th1 - th2)*((w2**2) * L2 + (w1**2)*L1*math.cos(th1 - th2))
    r /= L1*(2*m1 + m2 - m2*math.cos(2*th1 - 2*th2))
    return r

def fun2(th1, th2, w1, w2):
    #o2 = w2'
    r = (w1**2)*L1*(m1 + m2)
    r += g*(m1 + m2)*math.cos(th1)
    r += (w2**2)*L2*m2*math.cos(th1 - th2)
    r = r* 2*math.sin(th1 - th2)
    r = r/ (L2*(2*m1 + m2 - m2*math.cos(2*th1 - 2*th2)))
    return r

#o1 = w1'. o2 = w2'
o1 = fun1(th1, th2, w1, w2)
o2 = fun2(th1, th2, w1, w2)

#used for verficiation
def energy(th1, th2, w1, w2):
    KE = 0.5 * m1 * (L1 * w1)**2 + 0.5 * m2 * ((L1 * w1)**2 + (L2 * w2)**2 + 2 * L1 * L2 * w1 * w2 * math.cos(th1 - th2))
    PE = -m1 * g * L1 * math.cos(th1) - m2 * g * (L1 * math.cos(th1) + L2 * math.cos(th2))
    return KE + PE

energies = []

window = tk.Tk()
canvas = tk.Canvas(window, bg = 'black', highlightthickness= 0)

screen_width = 700
screen_height = 700

#window.attributes("-fullscreen", True)
canvas.config(width = screen_width, height = screen_height)

canvas.pack()

x1 = screen_width/2 + L1*math.sin(th1)
y1 = screen_height/2 + L1*math.cos(th1)
p1 = canvas.create_oval(x1 - 10, y1 - 10, x1 + 10, y1 + 10, fill='white')

l1 = canvas.create_line(screen_width/2, screen_height/2, x1, y1, fill = 'white', width = 2)

x2 =  x1 + L2*math.sin(th2)
y2 = y1 + L2*math.cos(th2)
p2 = canvas.create_oval(x2 - 10, y2 - 10, x2 + 10, y2 + 10,fill = 'white')

l2 = canvas.create_line(x1, y1, x2, y2, fill = 'white', width = 2)

#tracing logic
trace = True
if trace:
    trace_points = [x2, y2, x2, y2]
    trace_line = canvas.create_line(trace_points, fill="white", width=1, smooth=True)

def update(th1, th2, w1, w2, o1, o2, trace_points, trace_line):
    #using simplectic method instead for updating variables
    o1 = fun1(th1, th2, w1, w2)
    o2 = fun2(th1, th2, w1, w2)

    if random.randint(1,10000) == 5:
        o1 += 10

    w1 = (w1 + dt*o1)
    w2 = (w2 + dt*o2)

    th1 += dt * w1
    th2 += dt * w2

    #energies.append(energy(th1, th2, w1, w2))

    x1 = screen_width/2 + L1*math.sin(th1)
    y1 = screen_height/2 + L1*math.cos(th1)

    x2 =  x1 + L2*math.sin(th2)
    y2 = y1 + L2*math.cos(th2)

    if trace:
        if len(trace_points) == 10000:
            canvas.delete(trace_line)
            trace_points = [x2, y2, x2, y2]
            trace_line = canvas.create_line(trace_points, fill="white", width=1, smooth=True)
        else:
            trace_points.extend([x2, y2])
            canvas.coords(trace_line, *trace_points)

    canvas.coords(p1, [x1 - 10, y1 - 10, x1 + 10, y1 + 10])
    canvas.coords(p2, [x2 - 10, y2 - 10, x2 + 10, y2 + 10])
    canvas.coords(l1, [screen_width/2, screen_height/2, x1, y1])
    canvas.coords(l2, [x1, y1, x2, y2])

    canvas.after(1, lambda: update(th1, th2, w1, w2, o1, o2, trace_points, trace_line))

update(th1, th2, w1, w2, o1, o2, trace_points, trace_line)
window.mainloop()
if len(energies) != 0:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(energies)
    plt.show()