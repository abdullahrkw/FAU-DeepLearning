from pattern import Checker, Circle, Spectrum
from generator import ImageGenerator

if __name__=="__main__":
    ch = Checker(250, 25)
    ch.draw()
    ch.show(headless=True)

    cir = Circle(16, 2, (2, 4))
    cir.draw()
    cir.show(headless=True)

    spec = Spectrum(1024)
    spec.draw()
    spec.show(headless=True)

    ig = ImageGenerator("src/exercise_data", "src/Labels.json", 10, [200,200,3], shuffle=True, rotation=True, mirroring=True)
    ig.show(headless=True)
