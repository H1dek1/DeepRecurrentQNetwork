#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

class LinearScheduler:
    def __init__(self, init_val, final_val, init_epi, final_epi):
        self._init_val = init_val
        self._final_val = final_val
        self._init_epi = init_epi
        self._final_epi = final_epi

    def __call__(self, id_epi):
        if id_epi < self._init_epi:
            return self._init_val
        elif id_epi > self._final_epi:
            return self._final_val
        else:
            return self.__func(id_epi)

    def __func(self, id_epi):
        return (self._final_val-self._init_val) / (self._final_epi-self._init_epi) * (id_epi - self._init_epi) + self._init_val




class ExponentialScheduler:
    def __init__(self, init_val, final_val, init_epi, final_epi):
        self._init_val = init_val
        self._final_val = final_val
        self._init_epi = init_epi
        self._final_epi = final_epi

    def __call__(self, id_epi):
        if id_epi < self._init_epi:
            return self._init_val
        elif id_epi > self._final_epi:
            return self._final_val
        else:
            return self.__func(id_epi)

    def __func(self, id_epi):
        return self._init_val * self._final_val ** ((id_epi-self._init_epi) / (self._final_epi-self._init_epi))


def main():
    schdlr = ExponentialScheduler(1.0, 0.01, 100, 500)

    epi = np.arange(0, 600, 1)
    eps = []
    for e in epi:
        eps.append( schdlr(e) )

    fig, ax = plt.subplots(1, 1)
    ax.set_yscale('log')
    ax.plot(epi, eps)
    plt.show()

if __name__ == '__main__':
    main()
