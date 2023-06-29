'''
This script trains the reservoir to detect speech emotion on the spike-encoded
Crema dataset.
'''

from reservoirs import SpikeReservoir
import numpy as np

def main():
    shape = (20, 50)
        # M X N 
        # M neurons to match size of input layer.
        # N feature maps will convolve frequency information

    shape = (2,2)
    res = SpikeReservoir(shape)

    res.W = np.asarray([
        [0, 1, 2, 3],
        [0, 0, 2, 3],
        [0, 1, 0, 2],
        [0, 1, 2, 0],
    ])

    res.S = np.asarray([
       [0, 1], 
       [0, 1], 
    ])

    res.step()
    print(res.V)
if __name__ == '__main__':
    main()

