import numpy as np
import sys
import pickle
import copy

class SpikeReservoir:
    '''
    Basic integrate and fire spiking neural network. No refractory period.
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

    def step(self):
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
    notion of lateral inhibition). W contains only 0 forever. input_W updates.
    Shape M X N.
    M should match the size of input (in one time window). 
    N is the number of time-frequency feature maps to use.
    A "segment" is several neurons inside one feature map; these share initial
    weights.
    A neuron only spikes once per sample. 

    ---
    IAF as per SpikeReservoir.
    Segments of each feature convolve different time windows of input.
    Lateral inhibition: after a neuron fires, all other neurons in this
    row should be inhibited for the sample. 
    Spike-timing Dependent Plasticity: if input neuron spike occurs before 
    target neuron spike, that weight is updated positively and negatively
    otherwise.
    STDP should only be allowed once per sample inside a row.
    STDP should only be allowed once per sample inside a segment.
    Readout is really just the spike number from each segment. Achieved via
    pooling layer.
    Membrane potentials should be reset to zero after each sample. 
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
        self.inhibited = np.ones(shape) #1 for disinhibited, 0 for inhibited
        self.allow_stdp = np.ones(shape) #1 for allow stdp, 0 for disallow stdp 
        self.total_delta = 0

    def step(self):
        super().step()
    
    def set_shared_input_weights(self, input_W):
        self.input_W = input_W
        if np.isnan(input_W).any():
            sys.stderr.write(
                "Warning: Empath found NaN in shared input weights.\n")
    
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
        #find this segment in each feature and convolve input.
        for feature_idx in range(self.shape[1]):
            segment = self.get_local_segment(segment_idx, feature_idx)
            seg_inhibition = self.get_local_segment_inhibition(segment_idx,
                                                               feature_idx)
            
            segment += seg_inhibition * np.dot(
                self.input_W[:,:,feature_idx,segment_idx], 
                input_S)
            
            self.set_local_segment(segment_idx, feature_idx, segment)        
        
        #lateral inhibition
        segment_width = self.shape[0] // self.input_W.shape[3]
        segment_start = segment_idx * segment_width
        segment_end = segment_start + segment_width
        for row_idx in range(segment_start, segment_end):
            #calculate max in row over thresh, inhibit the rest of row  
            #i.e. just set their potential + inhibition to 0.
            # they should stay inhibited until end of sample
            if row_idx >= self.shape[0]:
                break
            row_max = copy.copy(np.amax(self.V[row_idx,:]))
            row_max_idx = np.argmax(self.V[row_idx,:])
            if row_max > self.threshold:
                self.inhibited[row_idx,:] = 0
                self.V[row_idx,:] = 0 
                self.V[row_idx, row_max_idx] = row_max
                # print(
                # "time window: ", time_window, 
                # "row/col: ", row_idx, row_max_idx)

        self.step()

    
    def update_input_weights_stdp(self, input_S, time_window=0):
        '''
        Spike-timing dependent plasticity
        ---
        Update weights during training folowing
            delta =     a_{p} w_{ij}(1 - w_{ij}) if t_j < t_i
                or
                    - a_{n} w_{ij} (1 - w_{ij}) otherwise
        
        where w is weight of synapse from jth input neuron to ith conv neuron.
        t_i and t_j are firing times.
        a_{p} and a_{n} are learning rates.
        Stop learning when delta is less than 0.01
        '''

        #iterate over current segment in each feature.
        segment_idx = time_window
        segment_width = self.shape[0] // self.input_W.shape[3]
        segment_start = segment_idx * segment_width
        segment_end = segment_start + segment_width

        for row_idx in range(segment_start, segment_end):
            if row_idx >= self.shape[0]:
                break
            for col_idx in range(self.shape[1]):
                if self.allow_stdp[row_idx, col_idx]:
                    #update weight
                    if self.S[row_idx,col_idx]:
                        delta = self.input_W[
                            row_idx - segment_start,
                            :,
                            col_idx,
                            time_window] * input_S 
                        delta = self.pos_lr * delta * (1 - delta) 

                    else:
                        delta = self.input_W[
                        row_idx - segment_start,
                        :,
                        col_idx,
                        time_window] * np.logical_not(input_S).astype(int)
                        delta = - (self.neg_lr * delta * (1 - delta)) 

                    self.input_W[row_idx - segment_start,
                            :,
                            col_idx,
                            time_window] += delta 

                    # print("stdp delta: ", delta)
                    #disallow stdp in row
                    self.allow_stdp[row_idx,:] = 0
                    #disallow stdp in segment
                    self.allow_stdp[segment_start:segment_end,col_idx] = 0
                    #update stop crit
                    self.total_delta += np.sum(abs(delta))
                    break #no point checking rest of row

    def check_stop_criterion(self):
        return self.total_delta < 0.01

    def reset_potentials(self):
        '''
        Reset all membrane potentials to 0.
        '''
        self.V = np.zeros(self.shape) 
    
    def reset_inhibition(self):
        '''
        Reset inhibition for all neurons to 1.
        '''
        self.inhibited = np.ones(self.shape) 
    
    def reset_pool(self):
        '''
        Reset pooling layer to None.
        '''
        self.pool = None
    
    def reset_allow_stdp(self):
        '''
        Reset allowing stdp for all neurons. 
        '''
        self.allow_stdp = np.ones(self.shape)
        self.total_delta = 0 
    
    def pool_segments(self):
        '''
        Pool (i.e. counts spikes from) each segment from each feature map.

        ---
        Each neuron convolves a segment from every feature.
        These neurons don't fire: final membrane potential trains a 
        linear classifier.
        Weights are fixed to one.
        So readout is really just the spike number from each segment.
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
    
    def get_local_segment_inhibition(self, segment_idx, feature_idx):
        segment_width = self.shape[0] // self.input_W.shape[3]
        feature = self.inhibited[:, feature_idx]
        segment_start = segment_idx * segment_width
        segment_end = segment_start + segment_width
        segment = feature[segment_start:segment_end] 
        return segment
    
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
    
    def save_input_weights(self, path):
        with open(path, 'wb+') as f:
            pickle.dump(self.input_W, f)

def main():
    print('This module contains classes for computation via reservoirs.') 
if __name__ == "__main__":
    main()
