import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
import IPython.display as display

class Plant:
    """Base class for a plant"""
    PLANT_DIM = 0
    CONTROL_DIM = 0
    OTYPE = 'Empty object'

    def __init__(self, seed_value=0):
        pass

    def __str__(self):
        return self.OTYPE

    def __call__(self, x, u=np.zeros(CONTROL_DIM), t=0.):
        return np.empty(0, dtype='double')

class Pendulum(Plant):
    """Plant representation of a pendulum on a carriage"""
    PLANT_DIM = 4
    CONTROL_DIM = 1
    OTYPE = 'Pendulum on a carriage'

    def __init__(self, seed_value=0):
        np.random.seed(seed_value)
        z = np.random.rand(5)
        self.omega = 0.3 + 1.5 * z[0]
        self.alpha = 0.3 + 0.5 * z[1]
        self.nu = 0.01 + 0.1 * z[2]
        self.lambd = self.nu + 0.01 + 0.1 * z[3]
        self.mu = self.alpha * self.nu + 0.01 + 0.1 * z[4]

    def __str__(self):
        return self.OTYPE + (": alpha = {0:.2g}, omega = {1:.2g}, " +
                             "lambda = {2:.2g}, mu = {3:.2g}, " +
                             "nu = {4:.2g}").format(self.alpha, self.omega,
                                                    self.lambd, self.mu,
                                                    self.nu)

    def __call__(self, x, u=np.zeros(CONTROL_DIM), t=0.):
        f = np.empty(4, dtype='double')
        cosx0 = np.cos(x[0])
        sinx0 = np.sin(x[0])
        cosx0sqr = cosx0 * cosx0
        q = 1. / (1. - self.alpha * cosx0sqr)
        omegasqr = self.omega * self.omega
        f[0] = x[1]
        f[1] = q * ((omegasqr - self.alpha * x[1] * x[1] * cosx0) * sinx0 -
                    (self.lambd - self.alpha * self.nu * cosx0sqr) * x[1] + 
                    (self.mu - self.nu) * cosx0 * x[3] - cosx0 * u[0])
        f[2] = x[3]
        f[3] = q * (-self.alpha * (omegasqr * cosx0 - x[1] * x[1]) * sinx0 +
                    self.alpha * (self.lambd -self.nu) * cosx0 * x[1] - 
                    (self.mu - self.alpha * self.nu * cosx0sqr) * x[3] +
                    u[0])
        return f

class Control:
    """Base class for a control strategy"""
    PLANT_DIM = 0
    CONTROL_DIM = 0
    HIDDEN_DIM = 0
    CTYPE = 'Empty control'

    def __init__(self):
        pass

    def __str__(self):
        return self.CTYPE

    def __call__(self, x, v, t=0.):
        return np.empty(0, dtype='double')

    def control(self, x, v, t=0.):
        return np.empty(0, dtype='double')

class ZeroControl(Control):
    """Zero control strategy"""
    CTYPE = 'Zero control'
    
    def __init__(self, plant_dim=1, control_dim=1):
        self.PLANT_DIM = plant_dim
        self.CONTROL_DIM = control_dim

    def control(self, x, v, t=0.):
        return np.zeros(self.CONTROL_DIM, dtype='double')
    
class StateFeedbackControl(ZeroControl):
    """State feedback control strategy"""
    CTYPE = 'State feedback control'

    def __init__(self, plant_dim=1, control_dim=1, fun=None):
        ZeroControl.__init__(self, plant_dim, control_dim)
        if fun is not None:
            self.fun = fun
        else:
            self.fun = lambda x: np.zeros(self.CONTROL_DIM, dtype='double')
    
    def __str__(self):
        return self.CTYPE + str(self.fun)
    
    def control(self, x, v, t=0.):
        return self.fun(x)
    
class StateTimeFeedbackControl(ZeroControl):
    """State-time feedback control strategy"""
    CTYPE = 'State-time feedback control'

    def __init__(self, plant_dim=1, control_dim=1, fun=None):
        ZeroControl.__init__(self, plant_dim, control_dim)
        if fun is not None:
            self.fun = fun
        else:
            self.fun = lambda x, t: np.zeros(self.CONTROL_DIM, dtype='double')
    
    def __str__(self):
        return self.CTYPE + str(self.fun)
    
    def control(self, x, v, t=0.):
        return np.array(self.fun(x, t))

class LinearStateControl(ZeroControl):
    """Linear state feedback control strategy"""
    CTYPE = 'Linear state feedback control'

    def __init__(self, plant_dim=1, control_dim=1, k=None):
        ZeroControl.__init__(self, plant_dim, control_dim)
        if k is not None:
            self.k = np.atleast_2d(k)
        else:
            self.k = np.zeros((self.CONTROL_DIM, self.PLANT_DIM), dtype='double')
    
    def __str__(self):
        return self.CTYPE + str(self.k)
    
    def control(self, x, v, t=0.):
        return self.k @ x
    
class LinearSinPendulumControl(Control):
    """Linear-like control strategy for pendulum on carriage"""
    PLANT_DIM = 4
    CONTROL_DIM = 1
    CTYPE = 'Linear (sin-type) pendulum control'

    def __init__(self, k=np.zeros(4, dtype='double')):
        self.k = np.copy(k)

    def __str__(self):
        return self.CTYPE + (": k0 = {0:.2g}, k1 = {1:.2g}, k2 = {2:.2g}, " +
                             "k3 = {3:.2g}").format(self.k[0], self.k[1],
                                                    self.k[2], self.k[3])

    def control(self, x, v, t=0.):
        return np.array([self.k[0] * np.sin(x[0]) +
                         np.dot(self.k[1:4], x[1:4])], dtype='double')

d16 = 1 / 6

def rk(rhs, y0, t, N=1, stop_condition=None):
    """Solve system of ODEs using the Runge–Kutta method"""
    t = np.array(t)
    y0 = np.array(y0)
    
    if stop_condition is None:
        stop_condition = lambda y1, t: False

    y = np.empty((t.size,) + y0.shape, dtype='double')
    y[0] = y0
    if N > 1:
        yt = y0
        h = 0.
        for i in range(t.size - 1):
            h = (t[i + 1] - t[i]) / N
            yt = np.copy(y[i])
            for n in range(N):
                k1 = h * rhs(yt, t[i] + n * h)
                k2 = h * rhs(yt + 0.5 * k1, t[i] + (n + 0.5) * h)
                k3 = h * rhs(yt + 0.5 * k2, t[i] + (n + 0.5) * h)
                k4 = h * rhs(yt + k3, t[i] + (n + 1) * h)
                yt += d16 * (k1 + 2 * (k2 + k3) + k4)
            y[i + 1] = yt
            if stop_condition(y[i + 1], t[i + 1]):
                return t[:i + 2], y[:i + 2]
    else:
        th = 0.
        h = 0.
        for i in range(t.size - 1):
            th = 0.5 * (t[i] + t[i + 1])
            h = t[i + 1] - t[i]
            k1 = h * rhs(y[i], t[i])
            k2 = h * rhs(y[i] + 0.5 * k1, th)
            k3 = h * rhs(y[i] + 0.5 * k2, th)
            k4 = h * rhs(y[i] + k3, t[i + 1])
            y[i + 1] = y[i] + d16 * (k1 + 2 * (k2 + k3) + k4)
            if stop_condition(y[i + 1], t[i + 1]):
                return t[:i + 2], y[:i + 2]
    return t, y

def pcrhs(p, c):
    """Create function determining plant (with regulator) in the state
    space
    """
    n = p.PLANT_DIM
    return lambda y, t: np.hstack((p(y[:n], c.control(y[:n], y[n:], t), t),
                                   c(y[:n], y[n:], t)))

def control_output(c):
    """Create function calculating control
    """
    n = c.PLANT_DIM
    return lambda y, t: c.control(y[:n], y[n:], t)

def integrate(p, c, x0, v0, dt, T, N=1, method=rk, return_control=False, stop_condition=None):
    """Calculate temporal evolution of a plant for some initial state"""
    t = np.arange(0., T, dt)
    y0 = np.hstack((x0, v0))
    t, y = method(pcrhs(p, c), y0, t, N, stop_condition)
    if return_control:
        u = np.empty((t.size, c.CONTROL_DIM), dtype='double')
        for i in range(t.size):
            u[i] = control_output(c)(y[i], t[i])
        return t, y, u
    else:
        return t, y

def animate_pendulum(t, y, xlim='auto', ylim=(-1.2, 1.2),
                     resolution=(960, 540), dpi=108, spacing=1, invsec=1.0,
                     filename=None, codec=None, progress=True):
    """Create Matplotlib animation of a pendulum on a carriage"""

    with plt.style.context('default'):
        fig = plt.figure(figsize=(resolution[0] / dpi, resolution[1] / dpi),
                         dpi=dpi)

        carriage_width = 0.2
        carriage_height = 0.1

        ymin = ylim[0]
        ymax = ylim[1]
        ratio = resolution[0] / resolution[1]

        if xlim == 'auto':
            x0 = 0.5 * (np.nanmin(y[:, 2]) + np.nanmax(y[:, 2]))    
            xmin = x0 - 0.5 * (ymax - ymin) * ratio
            xmax = x0 + 0.5 * (ymax - ymin) * ratio
        elif xlim == 'center':
            x0 = y[0, 2]
            xmin = x0 - 0.5 * (ymax - ymin) * ratio
            xmax = x0 + 0.5 * (ymax - ymin) * ratio
        else:
            xmin = xlim[0]
            xmax = xlim[1]

        ax = plt.axes(xlim=(xmin, xmax), ylim=(ymin, ymax), aspect='equal')

        time_text = ax.text(0.04, 0.9, '', transform=ax.transAxes)

        barline, = ax.plot([xmin, xmax], [0., 0.], lw=3, color='black')
        rect = patches.Rectangle([y[0, 2] - 0.5 * carriage_width,
                                  -0.5 * carriage_height], carriage_width,
                                 carriage_height, fill=True, color='red',
                                 ec='black')
        ax.add_patch(rect)
        barline.set_zorder(0)

        line, = ax.plot([], [], lw=2, marker='o', markersize=6)

        def init():
            time_text.set_text('')
            barline.set_data([xmin, xmax], [0., 0.])
            rect.set_xy((y[0, 2] - 0.5 * carriage_width, -0.5 * carriage_height))
            line.set_data([], [])
            return time_text, barline, rect, line,

        def animate(i):
            l = i * spacing
            time_text.set_text('time = {:2.2f}'.format(t[l]))
            rect.set_xy((y[l, 2] - 0.5 * carriage_width, -0.5 * carriage_height))
            line.set_data([y[l, 2], y[l, 2] + np.sin(y[l, 0])],
                          [0., np.cos(y[l, 0])])
            return time_text, rect, line,

        anim = animation.FuncAnimation(fig, animate, frames=t.size // spacing,
                                       init_func=init, interval=(t[1] - t[0]) *
                                       1000 * spacing * invsec, blit=True,
                                       repeat=False)

        if filename is not None:
            if progress:
                print(filename + ', saving progress')
                pb = display.ProgressBar(len(t) // spacing)
                pb_iter = iter(pb)
                anim.save(filename, fps=30, codec=codec,
                          progress_callback=lambda i, n: next(pb_iter))
                display.clear_output(wait=True)
            else:
                anim.save(filename, fps=30, codec=codec)
        plt.close(fig)
    return anim
