'''
This script trains the reservoir to detect speech emotion on the spike-encoded
Crema dataset.
'''

from reservoirs import Empath 
import numpy as np

def main():
    shape = (20, 50)
        # M X N 
        # M neurons to match size of input layer.
        # N feature maps will convolve frequency information

    empath = Empath(shape)
    empath.draw_shared_input_weights(n_local_segments=5)

if __name__ == '__main__':
    main()

