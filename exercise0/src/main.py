from pattern import Checker, Circle, Spectrum

if __name__=="__main__":
    ch = Checker(250, 25)
    ch.draw()
    ch.show()

    cir = Circle(1024, 200, (512, 256))
    cir.draw()
    cir.show()

    spec = Spectrum(1024)
    spec.draw()
    spec.show()
