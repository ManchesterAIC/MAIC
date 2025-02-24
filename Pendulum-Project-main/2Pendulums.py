import tkinter as tk
import math
import matplotlib.pyplot as plt

dt = 0.025
g = 9.81

#first pendulum
m1 = 1
m2 = 1
L1 = 100
L2 = 100

th1 = math.pi
th2 = math.pi/2

w1 = 0
w2 = 0


#second pendulum
m1_ = 1
m2_ = 1
L1_ = 100
L2_ = 100

th1_ = math.pi/2 + 0.000000001
th2_ = math.pi/2

w1_ = 0
w2_ = 0

def fun1(th1, th2, w2):
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

o1 = fun1(th1, th2, w2)
o2 = fun2(th1, th2, w1, w2)

o1_ = fun1(th1_, th2_, w2_)
o2_ = fun2(th1_, th2_, w1_, w2_)



window = tk.Tk()
canvas = tk.Canvas(window, bg = 'black', highlightthickness= 0)

screen_width = window.winfo_screenwidth()
screen_height = window.winfo_screenheight()

window.attributes("-fullscreen", True)
canvas.config(width = screen_width, height = screen_height)

canvas.pack()

#first pendulum
x1 = screen_width/2 + L1*math.sin(th1)
y1 = screen_height/2 + L1*math.cos(th1)
p1 = canvas.create_oval(x1 - 10, y1 - 10, x1 + 10, y1 + 10, fill='white')

l1 = canvas.create_line(screen_width/2, screen_height/2, x1, y1, fill = 'white', width = 2)

x2 =  x1 + L2*math.sin(th2)
y2 = y1 + L2*math.cos(th2)
p2 = canvas.create_oval(x2 - 10, y2 - 10, x2 + 10, y2 + 10,fill = 'white')

l2 = canvas.create_line(x1, y1, x2, y2, fill = 'white', width = 2)


#second pendulum
x1_ = screen_width/2 + L1_*math.sin(th1_)
y1_ = screen_height/2 + L1_*math.cos(th1_)
p1_ = canvas.create_oval(x1_ - 10, y1_ - 10, x1_ + 10, y1_ + 10, fill='white')

l1_ = canvas.create_line(screen_width/2, screen_height/2, x1_, y1_, fill = 'white', width = 2)

x2_ =  x1_ + L2_*math.sin(th2_)
y2_ = y1_ + L2_*math.cos(th2_)
p2_ = canvas.create_oval(x2_ - 10, y2_ - 10, x2_ + 10, y2_ + 10,fill = 'white')

l2_ = canvas.create_line(x1_, y1_, x2_, y2_, fill = 'white', width = 2)

#tracing logic
trace = True
if trace:
    trace_points = [x2, y2, x2, y2]
    trace_line = canvas.create_line(trace_points, fill="gray", width=1, smooth=True)

    trace_points_ = [x2_, y2_, x2_, y2_]
    trace_line_ = canvas.create_line(trace_points_, fill="white", width=1, smooth=True)

def update(th1, th2, w1, w2, o1, o2, th1_, th2_, w1_, w2_, o1_, o2_):

    #First Pendulum
    o1 = (fun1(th1, th2, w2))
    o2 = (fun2(th1, th2, w1, w2))

    w1 = (w1 + dt*o1)
    w2 = (w2 + dt*o2)

    th1 += dt * w1
    th2 += dt * w2

    x1 = screen_width/2 + L1*math.sin(th1)
    y1 = screen_height/2 + L1*math.cos(th1)

    x2 =  x1 + L2*math.sin(th2)
    y2 = y1 + L2*math.cos(th2)

    if trace:
        trace_points.extend([x2, y2])
        canvas.coords(trace_line, *trace_points)

    canvas.coords(p1, [x1 - 10, y1 - 10, x1 + 10, y1 + 10])
    canvas.coords(p2, [x2 - 10, y2 - 10, x2 + 10, y2 + 10])
    canvas.coords(l1, [screen_width/2, screen_height/2, x1, y1])
    canvas.coords(l2, [x1, y1, x2, y2])

    #Second Pendulum
    o1_ = (fun1(th1_, th2_, w2_))
    o2_ = (fun2(th1_, th2_, w1_, w2_))

    w1_ = (w1_ + dt*o1_)
    w2_ = (w2_ + dt*o2_)

    th1_ += dt * w1_
    th2_ += dt * w2_

    x1_ = screen_width/2 + L1_*math.sin(th1_)
    y1_ = screen_height/2 + L1_*math.cos(th1_)

    x2_ =  x1_ + L2_*math.sin(th2_)
    y2_ = y1_ + L2_*math.cos(th2_)

    if trace:
        trace_points_.extend([x2_, y2_])
        canvas.coords(trace_line_, *trace_points_)

    canvas.coords(p1_, [x1_ - 10, y1_ - 10, x1_ + 10, y1_ + 10])
    canvas.coords(p2_, [x2_ - 10, y2_ - 10, x2_ + 10, y2_ + 10])
    canvas.coords(l1_, [screen_width/2, screen_height/2, x1_, y1_])
    canvas.coords(l2_, [x1_, y1_, x2_, y2_])


    canvas.after(1, lambda: update(th1, th2, w1, w2, o1, o2, th1_, th2_, w1_, w2_, o1_, o2_))

update(th1, th2, w1, w2, o1, o2, th1_, th2_, w1_, w2_, o1_, o2_)
window.mainloop()