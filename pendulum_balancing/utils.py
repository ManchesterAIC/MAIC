#Activation function that only returns 1 for positive inputs
def step(x):
    return 1 if x > 0 else 0