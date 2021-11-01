import numpy as np

from matplotlib import pyplot as plt

class Checker(object):
    def __init__(self, resolution, tile_size):
        if not isinstance(tile_size, int):
            raise ValueError("tile_size must be an integar")
        if not isinstance(resolution, int):
            raise ValueError("resolution must be an integar")
        if (resolution % (2*tile_size)):
            raise ValueError("resolution must be divisible by 2*tile_size")

        self.tile_size = tile_size
        self.resolution = resolution
        self.output = np.ones((resolution, resolution), dtype=np.uint8)
        self.black = np.zeros((tile_size, tile_size), dtype=np.uint8)
        self.white = np.ones((tile_size, tile_size), dtype=np.uint8)

    def draw(self):
        reps = int(self.resolution/(2*self.tile_size))
        first_row = np.tile(np.hstack((self.black, self.white)), (1,reps))
        second_row = np.tile(np.hstack((self.white, self.black)), (1,reps))
        board = np.tile(np.vstack((first_row, second_row)), (reps,1))
        self.output = board
        return self.output.copy()

    def show(self, headless=False):
        plt.imshow(self.output, cmap='gray')
        if headless:
            plt.savefig("plot_checkers.png")
        else:
            plt.show()

class Circle(object):
    def __init__(self, resolution, radius, center):
        if not isinstance(radius, int):
            raise ValueError("radius must be an integar")
        if not isinstance(resolution, int):
            raise ValueError("resolution must be an integar")
        if not isinstance(center, tuple):
            raise ValueError("center position must be a tuple")
        if not self._are_circle_coords_valid(resolution,radius, center):
            raise ValueError("circle coords are not valid")
        self.resolution = resolution
        self.radius = radius
        self.center = center
        self.output = np.zeros((resolution, resolution), dtype=np.uint8)

    def _are_circle_coords_valid(self,resolution, radius, center):
        if radius < 0:
            return False
        if center[0] < 0:
            return False
        if center[1] < 0:
            return False
        if (radius + center[0]) > resolution or (center[0] - radius) < 0:
            return False
        if (radius + center[1]) > resolution or (center[1] - radius) < 0:
            return False
        return True
        
    def draw(self):
        center = self.center
        radius = self.radius
        x = np.arange(self.resolution)
        xx , yy = np.meshgrid(x,x)
        rad_sq = (xx - center[0])**2 + (yy - center[1])**2
        self.output = np.logical_and(rad_sq < (radius**2), True)
        return self.output.copy()

    def show(self, headless=False):
        plt.imshow(self.output, cmap='gray')
        if headless:
            plt.savefig("plot_circle.png")
        else:
            plt.show()


class Spectrum(object):
    def __init__(self, resolution):
        if not isinstance(resolution, int):
            raise ValueError("resolution must be an integar")
        self.resolution = resolution
        self.output = np.zeros((self.resolution, self.resolution, 3), dtype=np.float64)

    def draw(self):
        self.output[:,:,0] = np.linspace(0, 1, self.resolution)
        self.output[:,:,2] = np.linspace(1, 0, self.resolution)
        self.output[:,:,1] = np.linspace(0, 1, self.resolution).reshape(self.resolution, 1)
        return self.output.copy()

    def show(self, headless=False):
        plt.imshow(self.output)
        if headless:
            plt.savefig("plot_spectrum.png")
        else:
            plt.show()
