
def f(x):
    return x-15 

def gradient_descent(x0, lr,  epochs):
    for _ in range(epochs):
        x0 = x0- lr*f(x0)
    print(x0)
    return x0

print(gradient_descent(1, 0.01, 10000))
