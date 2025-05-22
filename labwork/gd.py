import matplotlib.pyplot as plt

arr = []
time = []

x0 = 10
lr = 1.1
f_x =  x0 * x0


def GradientDescent(iter, lr, x):
    x = x - lr * 2 * x
    iter = iter + 1
    f_x = x * x
    print("iter: ", iter, "\tx: ", x, "\ty: ", f_x)
    arr.append(f_x)
    time.append(iter)

    if (iter < 10):
        return GradientDescent(iter, lr, x)
    else:
        return iter, lr, x
    

GradientDescent(0, lr, x0)

plt.plot(time, arr)
plt.xlabel('Time')
plt.ylabel('F(x) value')
# plt.title('Sample Plot')
plt.show()
