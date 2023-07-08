'''
This script trains the reservoir to detect speech emotion on the spike-encoded
Crema dataset.
'''

from reservoirs import Empath 
import numpy as np
import pickle
import sys
import copy

n_signal_epochs = 5
epoch_len = 50
time_step_bin_width = 30 

def main():
    shape = (20, 50)

    empath = Empath(shape, threshold=5)
    empath.draw_shared_input_weights(n_local_segments=n_signal_epochs)

    input_spike_trains = None
    with open('preprocessed/crema_spikes/1001_DFA_DIS_XX_spks.pk', 'rb') as f:
        input_spike_trains = pickle.load(f)
    assert not np.isnan(input_spike_trains).any()

    stride_len = input_spike_trains.shape[1] // n_signal_epochs
    assert epoch_len > stride_len , "choose a longer signal epoch length"

    # build stride start indices
    stride_starts = np.zeros(n_signal_epochs) 
    stride_starts += stride_len
    stride_starts[:input_spike_trains.shape[1] % n_signal_epochs] += 1
    stride_starts = (np.cumsum(stride_starts) - stride_starts).astype(int)
    
    #signal epochs are windows that slide over the input
    #note Empath is still stimulated one time step at a time.
    #the signal epoch affects which segment of the feature map is stimulated.

    readouts = []
    recorded_spikes = []
    epoch_idx = 0
    for start in stride_starts:
        epoch = input_spike_trains[:,start:start+epoch_len]
        for time_step in range(epoch.shape[1]):       
            latencies = epoch[:,time_step]
            latencies = latencies - (start+time_step) #undo cumsum
            latencies = latencies * time_step_bin_width # [0,time_bin_width] 
            for time_step_bin in range(time_step_bin_width):
                if time_step_bin == 0:
                    S = ((latencies >= time_step_bin) &
                     (latencies <= time_step_bin+1))
                else:
                    S = ((latencies > time_step_bin) &
                     (latencies <= time_step_bin+1))
                if S.any():
                    empath.stimulate(S.astype(int), time_window=epoch_idx)
                else:
                    empath.step()

                empath.update_input_weights_stdp(S.astype(int),
                                                 time_window=epoch_idx)
                empath.pool_segments()
                readouts.append(empath.readout())
                recorded_spikes.append(empath.get_state())
        epoch_idx += 1
    
    with open('logs/debug/readouts.pk', 'wb+') as f:
        pickle.dump(readouts, f)
    
    recorded_spikes = np.asarray(recorded_spikes)
    with open('logs/debug/reservoir_spikes.pk', 'wb+') as f:
        pickle.dump(recorded_spikes, f)
if __name__ == '__main__':
    main()

