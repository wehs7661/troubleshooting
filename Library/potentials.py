import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib import gridspec

rc('font', **{
    'family': 'sans-serif',
    'sans-serif': ['DejaVu Sans'],
    'size': 10
})
# Set the font used for MathJax - more on this later
rc('mathtext', **{'default': 'regular'})
plt.rc('font', family='serif')

class DoubleWellPotential:
    def __init__(self, **kwargs):
        defaults = {"a": 1, "b": 6, "c": 1, "d": 1}
        if not len(kwargs):
            self.params = defaults
        else:
            self.params = kwargs
        for key in self.params:
            setattr(self, key, self.params[key])

        # stationary pts: solve 4ax^{3} - 2bx + c=0 (y = 0)
        self.x_roots = np.roots([4 * self.a, 0, -2 * self.b, self.c])
        self.y_roots = np.zeros(3)
        self.stationary_pts = np.array([[self.x_roots[i], self.y_roots[i]] for i in range(3)])

        # classify the stationary pts into minima and saddle point
        # solve 12ax^{2} - 2b = 0
        if self.a * self.b >= 0:  # ensure that there are two inflection pts
            for i in range(len(self.x_roots)):
                if self.x_roots[i] > np.sqrt(self.b / 6 * self.a):
                    self.min_right = np.array([self.x_roots[i], 0])
                if self.x_roots[i] < - np.sqrt(self.b / 6 * self.a):
                    self.min_left = np.array([self.x_roots[i], 0])
                else:
                    self.saddle = np.array([self.x_roots[i], 0])
        
        # energy barrier and energy difference
        e_right = self.get_energy(self.min_right)
        e_left = self.get_energy(self.min_left)
        e_saddle = self.get_energy(self.saddle)
        self.e_diff = e_right - e_left 
        self.barrier = e_saddle - e_left

    def get_energy(self, coords):
        x = coords[0]
        y = coords[1]
        E = self.a * x ** 4 - self.b * x ** 2 + self.c * x + self.d * y ** 2
        return E

    def plot_section(self, **kwargs):
        
        if len(kwargs) != 1:
            print('Error! x or y should fixed.')
        
        plt.grid()
        if 'x' in kwargs:  # x1 is fixed
            x = kwargs['x']
            y  = np.linspace(-7, 7, 100)
            E = self.get_energy([x, y])
            plt.plot(y, E)
            plt.xlabel('$ x_{2} $')
            plt.ylabel('$ Energy $')
            plt.title('Section of the double-well potential at $ x_{1} $ = %s' % x)

        elif 'y' in kwargs:  # x2 is fixed
            y = kwargs['y']
            x = np.linspace(-3, 3, 100)
            E = self.get_energy([x, y])
            plt.plot(x, E)
            plt.xlabel('$ x_{1} $')
            plt.ylabel('Potential energy $u({x_{1}})$')
            plt.title('Cross section of the double-well potential at $ x_{2} $ = %s' % y)

    def plot_FES(self):
        x = np.linspace(-5, 5, 100)
        y = np.linspace(-8, 8, 100)
        X, Y = np.meshgrid(x, y)
        E = self.get_energy([X, Y])

        plt.contourf(X, Y, E, 20)
        plt.xlabel('$ x_{1} $')
        plt.ylabel('$ x_{2} $')
        plt.title('Contour plot of the double-well potential')
        clb = plt.colorbar()
        clb.ax.set_title(r'$H(x)$')

    def plot_samples(self, samples=None,  fig=None, color='green'):
        nofig = False
        if fig is None:
            nofig = True
            fig = plt.figure(figsize=(12, 3))

        plt.subplot(1, 2, 1)
        if nofig is True:
            self.plot_FES()
        if samples is not None:
            plt.scatter(samples[:, 0], samples[:, 1], color=color, s=0.5)
        
        plt.subplot(1, 2, 2)
        if nofig is True:
            self.plot_section(y=0)
        if samples is not None:
            plt.scatter(samples[:, 0], self.get_energy(samples.T), color=color, s=0.5)

        return fig

