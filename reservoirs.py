# CONVOLUTION

# 1. integrate and fire.
#       V(t) = V(t-1) + W^T S(t - 1)
#       reset potential to zero afterwards
# 2. Feature maps; sublayers convolve different times of input
# 3. share weights among spatially nearby neurons, but separate sets of weights
#   for different time periods.
# 4. lateral inhibition; after a neuron fires, all neurons in the same position
# in other feature map are inhibited until the next sample appears; there is only ever one spike
# allowed in each position.

# STDP
# 1. intitialise weights from Gaussian
# 2. update weights during training folowing
#       delta =     a_{p} w_{ij}(1 - w_{ij}) if t_j < t_i
#               or
#                   - a_{n} w_{ij} (1 - w_{ij}) otherwise
# where w is weight of synapse from jth input neuron to ith conv neuron
# t_i and t_j are firing times
# a_{p} and a_{n} are learning rates.
# 3. stop learning when delta is less than 0.01
# 4. STDP not allowed after one neuron updates in the same position across 
# feature maps until next sample appears

# POOLING (readout)
#1. each neuron in pooling layer integrates inputs from one section in the 
# corresponding feature map in the conv layer.
# 2. these do not fire; final membrane potentials are used as training data
# for a linear classifier
# 3. weights are fixed to one (readout is basicaly the spike number from corresponding
# section)
# 4. reset membrane potential after a sample.

# linear classifier
#1. First train SNN on training set, following STDP.
#2. Fix weights (turn off plasticity), run the training set again but to train
# linear classifier on membrane potentials of pooling layer and corresponding labels
#3. evaluate fixed network using test set with trained classifier

import numpy as np
import sys
import pickle

class SpikeReservoir:
    '''
    Integrate and fire spiking neural network.
    ---
    '''

    def __init__(self, shape):
        '''
        V: membrane potentials.
        W: synaptic weights (W[ij] stores neuron j -> neuron i).
        S: occurence of spikes in last time step.
        threshold: membrane potential threshold.     
        '''
        self.shape = shape
        self.V = np.zeros(shape)
        self.W = np.zeros((np.prod(shape), np.prod(shape)))
        self.S = np.zeros(shape)
        self.threshold = 23 

    def step(self, plasticity=False):
        '''
        Step according to
        V(t) = V(t-1) + W^T S(t - 1)
        where V should be a vector of membrane potentials,
        W should be a matrix of synaptic weights,
        S should be a vector of firing states.
        '''
        flat_V = self.V.flatten()
        flat_S = self.S.flatten()
        flat_V = flat_V + np.dot(self.W, flat_S)
        self.V = np.reshape(flat_V, self.shape)
        self.S = (self.V > self.threshold).astype(int)
        self.V = np.where(self.V > self.threshold, 0, self.V)

class Empath(SpikeReservoir):
    '''
    SNN architecture designed to detect speech emotion.
    Reservoir only convolves input, does not connect to itself (but there is a
    notion of lateral inhibition).
    Shape M X N.
    M should match the size of input (in one time window). 
    N is the number of time-frequency feature maps to use.
    '''

    def draw_shared_input_weights(self,n_local_segments=10):
        '''Draw from Gaussian input weights shared inside each segmented feature
        map.

        Params
        ---
        n_local_segments: `int` Number of feature map segments to use
        '''
       
        assert n_local_segments < self.shape[0], "Too many local segments."
        err_msg = "M must be multiple of n_local_segments"
        assert self.shape[0] % n_local_segments == 0, err_msg

        #find segment start indices.
        segment_width = self.shape[0] // n_local_segments
        segment_starts = np.asarray([
            i for i in range(0, self.shape[0], segment_width)
        ]) 

        #draw shared weights from gaussian
        prng = np.random.default_rng()
        shared_weights = prng.normal(
            loc=0.0, scale=1.0, size=(n_local_segments, self.shape[1])) 

        #input will be fed to different segments of the feature map over time
        n_windows = n_local_segments

        #4D input weight matrix
        #(segment_width x M x N x n_windows)
        input_W = np.zeros(
            (segment_width, self.shape[0], self.shape[1], n_local_segments))

        for i in range(n_windows): 
            for j in range(self.shape[1]):
                input_W[:,:,j,i] = shared_weights[i][j]

        self.input_W = input_W

    def stimulate(self, input_S):
        pass

def main():
    print('This module contains classes for computation via reservoirs.') 
if __name__ == "__main__":
    main()
