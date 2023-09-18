import matplotlib.pyplot as plt
import numpy as np

def f1(t):return -t * t * 0.5

def f2(x):return -t*t *0.5 + t**4 * 0.01


if __name__ == "__main__":
    plt.figure("results comparison")
    t = np.linspace(0,2.5,10)
    x = -t * t * 0.45 + 0.02
    plt.plot(t,f1(t))
    plt.plot(t,f2(t))
    plt.plot(t,x)
    plt.legend(["no-drag","drag","observation"])
    plt.show()