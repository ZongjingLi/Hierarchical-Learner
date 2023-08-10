import taichi as ti

ti.init()

N = 8
dt = 1e-5

x = ti.Vector.field(2, dtype=ti.f32, shape=N, needs_grad=True)
v = ti.Vector.field(2, dtype=ti.f32, shape=N)
U = ti.field(dtype=ti.f32, shape=(), needs_grad=True)

@ti.kernel
def compute_U():
    for i,j in ti.ndrange(N,N):
        r = x[i] - x[j]
        U[None]+= -1/r.norm(1e-3)

@ti.kernel
def advance():
    for i in x:
        v[i] += dt * -x.grad[i]
    for i in x:
        x[i] += dt * v[i]

@ti.kernel
def init():
    for i in x: x[i] = [ti.random(), ti.random()]

def substep():
    with ti.ad.Tape(loss = U):
        compute_U()
    advance()

init()

gui = ti.GUI("Gravity")
while gui.running:
    for i in range(50):
        substep()
    gui.circles(x.to_numpy(), radius=3)
    gui.show()