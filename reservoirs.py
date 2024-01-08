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
import copy

class SpikeReservoir:
    '''
    Integrate and fire spiking neural network.
    ---
    '''

    def __init__(self, shape, threshold=23):
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
        self.threshold = threshold 

    def step(self, plasticity=False):
        '''
        Step according to
        V(t) = V(t-1) + W^T S(t - 1)
        where V should be a vector of membrane potentials,
        W should be a matrix of synaptic weights,
        S should be a vector of firing states.
        no refractory period.
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
    Does have refractory period.
    '''

    def __init__(
            self, 
            shape, 
            threshold=23,
            positive_learning_rate=0.004,
            negative_learning_rate=0.003
            ):

        super().__init__(shape, threshold=threshold)
        self.input_W = None
        self.pool = None
        self.pos_lr = positive_learning_rate
        self.neg_lr = negative_learning_rate
        self.refractory = np.zeros(shape)

    def step(self, plasticity=False):
        self.refractory = np.where(
            self.refractory > 0, self.refractory - 1, self.refractory
        )
        super().step()
    
    def draw_shared_input_weights(self,n_local_segments=10):
        '''Draw from Gaussian input weights shared inside each segmented feature
        map. Creates input weight matrix for this instance. 

        Params
        ---
        n_local_segments: `int` Number of feature map segments to use

        Returns
        ---
        `None`
        '''
       
        assert n_local_segments < self.shape[0], "Too many local segments."
        err_msg = "M must be multiple of n_local_segments"
        assert self.shape[0] % n_local_segments == 0, err_msg

        segment_width = self.shape[0] // n_local_segments

        #draw shared weights from gaussian
        prng = np.random.default_rng()
        shared_weights = prng.normal(
            loc=0.8,
            scale=0.1,
            size=(n_local_segments, self.shape[0], self.shape[1])) 
        assert not (shared_weights < 0).any(), "Negative input weights found."

        #4D input weight matrix
        #(segment_width x M x N x n_windows)
        n_windows = n_local_segments
        input_W = np.zeros(
            (segment_width, self.shape[0], self.shape[1], n_windows))

        #broadcast 3D shared weight values to 4D input weight matrix
        for i in range(n_local_segments):
            input_W[:,:,:,i] = shared_weights[i]

        self.input_W = input_W

    def stimulate(self, input_S, time_window=0):
        '''
        Integrate spikes arriving from input neurons. 
        
        Params
        ---
        input_S: `ndarray` vector of input neuron states.
        time_window: `int` index of time windows i.e. also the segment index.

        Returns
        ---
        `None`
        '''

        segment_idx = time_window

        segment_over_thresh = np.zeros(self.shape[1]).astype(bool)
        max_neuron_idx = np.zeros(self.shape[1]).astype(int)
        max_neuron = np.zeros(self.shape[1])

        for feature_idx in range(self.shape[1]):
            segment = self.get_local_segment(segment_idx, feature_idx)
            seg_refractory = self.get_local_segment_refractory(segment_idx,
                                                               feature_idx)
            
            segment += np.dot(
                self.input_W[:,:,feature_idx,segment_idx], 
                input_S)
            
            #refractory period: hold potential to zero where refractory > 0.
            segment = np.where(seg_refractory > 0, 0, segment)
            
            segment_over_thresh[feature_idx] = (segment > self.threshold).any()
            max_neuron_idx[feature_idx] = np.argmax(segment)
            max_neuron[feature_idx] = np.amax(segment)

            self.set_local_segment(segment_idx, feature_idx, segment)        

        # max pot over threshold inhibits other features and neighbourhood
        max_feature_idx = np.argmax(max_neuron) 
        if segment_over_thresh[max_feature_idx]:
            for feature_idx in range(self.shape[1]):
                segment = self.get_local_segment(segment_idx, feature_idx)
                segment = np.where(segment > self.threshold, 0, segment)
                seg_refractory = self.get_local_segment_refractory(segment_idx,
                                                                   feature_idx)

                if feature_idx == max_feature_idx:
                    winner = max_neuron_idx[feature_idx]
                    segment[winner] = max_neuron[feature_idx]
                    #refractory period: winner cannot spike until refractory
                    # ticks down to zero. 
                    seg_refractory[winner] = 4 
                self.set_local_segment(segment_idx, feature_idx,segment)
                self.set_local_segment_refractory(segment_idx, feature_idx,
                                                  seg_refractory)

        self.step()

    
    def update_input_weights_stdp(self, input_S, time_window=0):
        '''Spike-timing dependent plasticity'''

        segment_idx = time_window
        for feature_idx in range(self.shape[1]):
            segment_state = self.get_local_segment_state(
                segment_idx, feature_idx)
            #updates whole segment negatively and then winner positively.
            neuron_idx = np.argmax(segment_state)
            winner_saved_weights = copy.copy(self.input_W[
                        neuron_idx,
                        :,
                        feature_idx,
                        time_window]) 
            neg_delta = self.input_W[
                        :,
                        :,
                        feature_idx,
                        time_window] * np.logical_not(input_S).astype(int)
            neg_delta = - (self.neg_lr * neg_delta * (1 - neg_delta))
            self.input_W[:,
                        :,
                        feature_idx,
                        time_window] += neg_delta
            if segment_state.any():
                pos_delta = self.input_W[
                            neuron_idx,
                            :,
                            feature_idx,
                            time_window] * input_S 
                pos_delta = self.pos_lr * pos_delta * (1 - pos_delta)
                self.input_W[neuron_idx,
                                :,
                                feature_idx,
                                time_window] = winner_saved_weights + pos_delta 


    def reset_potentials(self):
        '''
        Reset all membrane potentials to 0.
        '''
        self.V = 0
    
    def reset_refractory(self):
        '''
        Reset refractory period for all neurons to 0.
        '''
        self.refractory = 0
    
    def pool_segments(self):
        '''
        Pool (i.e. counts spikes from) each segment from each feature map.
        '''

        if self.pool is None:
            pool_shape = (self.input_W.shape[3], self.shape[1])
            self.pool = np.zeros(pool_shape)

        segment_width = self.shape[0] // self.input_W.shape[3]
        
        for segment_idx in range(self.input_W.shape[3]):
            segment_start = segment_idx * segment_width
            segment_end = segment_start + segment_width

            self.pool[segment_idx] += np.matmul(
                np.ones(segment_width),
                self.S[segment_start:segment_end],
            )
        
        # print(self.pool)

    def readout(self):
        return copy.copy(self.pool)
    
    def get_state(self):
        '''Returns copy of firing state'''
        return copy.copy(self.S)
    
    def get_local_segment_state(self, segment_idx, feature_idx):
        segment_width = self.shape[0] // self.input_W.shape[3]
        feature = self.S[:, feature_idx]
        segment_start = segment_idx * segment_width
        segment_end = segment_start + segment_width
        segment = feature[segment_start:segment_end] 
        return segment
    
    def get_local_segment_refractory(self, segment_idx, feature_idx):
        segment_width = self.shape[0] // self.input_W.shape[3]
        feature = self.refractory[:, feature_idx]
        segment_start = segment_idx * segment_width
        segment_end = segment_start + segment_width
        segment = feature[segment_start:segment_end] 
        return segment

    def set_local_segment_refractory(self, segment_idx, feature_idx, segment):
        segment_width = self.shape[0] // self.input_W.shape[3]
        feature = self.refractory[:, feature_idx]
        segment_start = segment_idx * segment_width
        segment_end = segment_start + segment_width
        feature[segment_start:segment_end] = segment
        self.refractory[:,feature_idx] = feature
    
    def get_local_segment(self, segment_idx, feature_idx):
        segment_width = self.shape[0] // self.input_W.shape[3]
        feature = self.V[:, feature_idx]
        segment_start = segment_idx * segment_width
        segment_end = segment_start + segment_width
        segment = feature[segment_start:segment_end] 
        return segment

    def set_local_segment(self, segment_idx, feature_idx, segment):
        segment_width = self.shape[0] // self.input_W.shape[3]
        feature = self.V[:, feature_idx]
        segment_start = segment_idx * segment_width
        segment_end = segment_start + segment_width
        feature[segment_start:segment_end] = segment
        self.V[:,feature_idx] = feature


def main():
    print('This module contains classes for computation via reservoirs.') 
if __name__ == "__main__":
    main()
