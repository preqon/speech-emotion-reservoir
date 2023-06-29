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
        self.V = np.reshape(flat_V, self.V.shape)
        self.S = (self.V > self.threshold).astype(int)
        self.V = np.where(self.V > self.threshold, 0, self.V)

class Empath(SpikeReservoir):
    '''
    SNN architecture designed to detect speech emotion. 
    '''
    def draw_weights():
       pass 

def main():
    pass
if __name__ == "__main__":
    main()
