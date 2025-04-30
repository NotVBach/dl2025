import csv
import matplotlib.pyplot as plt

x = []
y = []
loss_arr = []
time = []
with open("lr.csv", 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        x.append(float(row[0]))
        y.append(float(row[1]))

def loss_func(x, y, w1, w0):
    n = len(x)
    err = 0
    for i in range(n):
        y_hat = w1 * x[i] + w0
        err = err + (1/2) * (y_hat - y[i]) ** 2
    return err/n

def GradientDescent(iter, lr, x, y):
    w0 = 0.0
    w1 = 1.0

    n = len(x)
    
    for i in range(100):
        dw0 = 0.0
        dw1 = 0.0
        for j in range(n):
            y_hat = w1 * x[j] + w0
            err = y_hat - y[j]
            dw0 += err/n
            dw1 += x[j] * err/n
        
        w0 = w0 - lr * dw0
        w1 = w1 - lr * dw1

        iter = iter + 1
        time.append(iter)

        loss = loss_func(x, y, w1, w0)
        loss_arr.append(loss)
        print(f"iter{iter}: \t w1 = {w1} \t w0 = {w0} \t loss = {loss}")

    return f"Final w1: {w1} \t w0: {w0}"

lr = 0.001
print(GradientDescent(0, lr, x, y))

plt.plot(time, loss_arr)
plt.xlabel('Time')
plt.ylabel('Loss')
plt.show()




