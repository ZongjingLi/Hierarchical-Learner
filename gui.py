import taichi as ti
import numpy as np
import taichi.math as tm

ti.init(arch=ti.gpu)

@ti.func
def complex_sqrt(z):
    return 0

def run_gui():
    X = np.random.random((2, 2))
    Y = np.random.random((2, 2))
    Z = np.random.random((2, 2))
    gui = ti.GUI("triangles", res=(400, 400))
    while gui.running:
        gui.triangles(a=X, b=Y, c=Z, color=0xED553B)
        gui.show()

def run_arrows():
    begins = np.random.random((100, 2))
    directions = np.random.uniform(low=-0.05, high=0.05, size=(100, 2))
    gui = ti.GUI('arrows', res=(400, 400))
    while gui.running:
        gui.arrows(orig=begins, direction=directions, radius=1)
        gui.show()

class SquareArea:
    def __init__(self, start, end):
        super().__init__()

def run_main():
    gui = ti.GUI("main",res = (1400,1400))
    while gui.running:
        for e in gui.get_events(gui.PRESS):
            if e.key == gui.ESCAPE:
                gui.running = False
        r = 5
        x,y = [0.5,0.5]
        gui.circle([x,y], radius = r)
        x,y = [.7,0.5]
        gui.circle([x,y], radius = r)
        gui.line([0.5,0.5],[x,y],radius = 1,color=0xB4B3E1)
        gui.text(content="Ice Crown",pos=[0.5,0.6])
        gui.show()

run_main()