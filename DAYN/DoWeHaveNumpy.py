import numpy as np
import matplotlib.pyplot as plt

X = np.random.rand(100)
Y = 0.2*X+ 0.5

plt.figure(figsize=(8,6))
plt.scatter(X,Y)
plt.show()
plt.close()

def predictor(pred,y):
    plt.figure(figsize=(8,6))
    plt.scatter(X,Y)
    plt.scatter(X,pred)
    plt.show()

##GD
w= np.random.uniform(-1,1)
b= np.random.uniform(-1,1)

LR=0.5


for epoch in range(200):
    Y_pred = w*X+b
    error = np.abs(Y_pred-Y).mean()
    if error < 0.001:
        break
    w_grad= LR*((Y_pred-Y)*X).mean()
    b_grad= LR*((Y_pred-Y)).mean()

    w= w-w_grad
    b= b-b_grad

    if epoch % 20 == 0:
        Y_pred = w*X+b
        predictor(Y_pred,Y)





    #gd

