import math
import matplotlib.pyplot as plt

def sigmoid(z):
    try:
        return 1 / (1 + math.exp(-z))
    except OverflowError:
        return 0 if z < 0 else 1

def compute_y_hat(x1, x2, w1, w2, w0):
    z = w1 * x1 + w2 * x2 + w0
    y_hat = sigmoid(z)
    return y_hat

def log(x):
    x = max(1e-10, min(x, 1 - 1e-10))
    return math.log(x)

def loss_func(X1, X2, y, w1, w2, w0):
    m = len(y)
    loss = 0
    for i in range(m):
        y_hat = compute_y_hat(X1[i], X2[i], w1, w2, w0)
        loss = loss - (y[i] * log(sigmoid(y_hat)) + (1 - y[i]) * log(1 - sigmoid(y_hat)))
    return loss / m

def compute_gradients(X1, X2, y, w1, w2, w0):
    m = len(y)
    dw1 = 0.0
    dw2 = 0.0
    dw0 = 0.0
    
    for i in range(m):
        y_hat = compute_y_hat(x1, x2, w1, w2, w0)
        err = y_hat - y[i]

        dw1 = dw1 + err * X1[i]  # Gradient for w1 # df/dw1 = (y_hat - y) * x1
        dw2 = dw2 + err * X2[i]  # Gradient for w2
        dw0 = dw0 + err # df/dw0 = (y_hat - y)

    return dw1, dw2, dw0

def gradient_descent(X1, X2, y, lr, iter, loss, iteration):
    w1 = 0.0
    w2 = 0.0
    w0 = 0.0
    
    for i in range(iter):

        loss_val = loss_func(X1, X2, y, w1, w2, w0)
        loss.append(loss_val)
        iteration.append(i)
        # Compute gradients
        dw1, dw2, dw0 = compute_gradients(X1, X2, y, w1, w2, w0)
        # Update weights
        w1 = w1 - lr * dw1
        w2 = w2 - lr * dw2
        w0 = w0 - lr * dw0

        print(f"Iteration {i}: Loss={loss_val}, w0={w0}, w1={w1}, w2={w2}")
    return w1, w2, w0



X1 = [] #Experience
X2 = [] #Salary
y = [] #Loan / Not loan
loss = []
iterations = []

with open('loan2.csv', 'r') as file:
    next(file)
    for line in file:
        values = line.strip().split(',')
        x1 = float(values[0])  # Experience
        x2 = float(values[1])  # Salary
        y_val = float(values[2])  # Loan
        X1.append(x1)
        X2.append(x2)
        y.append(y_val)

lr = 0.1
iter = 1000
gradient_descent(X1, X2, y, lr, iter, loss, iterations)

plt.figure(figsize=(10, 8))
plt.plot(iterations, loss, label='Loss', color='blue')
plt.xlabel('Iteration')
plt.ylabel('Loss (Binary Cross-Entropy)')
# plt.title('Loss Function Over Iterations')
plt.show()  # Display the plot


