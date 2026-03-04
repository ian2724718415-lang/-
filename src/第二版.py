import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
x1=sp.Symbol('x')
y1=sp.Symbol('y')
f1 = sp.Pow(x1 - 3,2)+sp.Pow(y1 - 2,2)
f2 = [sp.diff(f1,z) for z in [x1,y1]]
def loss(x):
    return float(f1.subs({x1:x[0],y1:x[1]}))
def grad_func(x):
    return np.array([
        float(f2[0].subs({x1:x[0],y1:x[1]})),
        float(f2[1].subs({x1:x[0],y1:x[1]}))
    ])
def gradient_descent(x0,rate,number):
    x=np.array(x0,dtype=float)
    xn=[x.copy()]
    loss_history=[loss(x)]
    for k in range(number):
        grad=grad_func(x)
        x=x-rate*grad
        xn.append(x.copy())
        loss_history.append(loss(x))
        print(f"第{k+1}步迭代:x={x},偏离值={loss_history[-1]:.6f}")
    return np.array(xn),np.array(loss_history)
x0=[0,0]
rate=0.1
number=30
xn,loss_history=gradient_descent(x0,rate,number)
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(loss_history)
plt.xlabel("迭代次数")
plt.ylabel("损失值")
plt.title("损失函数下降曲线")
plt.subplot(1,2,2)
x1_grid,y1_grid = np.meshgrid(np.linspace(-1,5,100),np.linspace(-1,5,100))
f_grid = (x1_grid -3)**2 + (y1_grid - 2)**2
plt.contour(x1_grid,y1_grid,f_grid,levels=20)
plt.plot(xn[:,0],xn[:,1],'ro-',label="最小值点(3,2)")
plt.xlabel('x1')
plt.ylabel('y1')
plt.legend()
plt.title("迭代路径")
plt.tight_layout()
plt.show()