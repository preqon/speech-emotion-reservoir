'''
This script trains the reservoir to detect speech emotion on the spike-encoded
Crema dataset.
Can choose to use fixed weights (no STDP) instead; this script then simply 
generates readouts for the spike-encoded Crema dataset.
'''

from reservoirs import Empath 
import numpy as np
import pickle
import sys
import copy
import glob
import time

# ==== command line options
# N_SAMPLES = 5000
# FIXED_WEIGHTS = False
# dataset = "crema"
#===

# ==== script options (check before running)
log_name = "readout-fsdd"
dstring = "2024-04-19"
n_signal_epochs = 4 #(set to 5 for crema)
epoch_len = 8 #(set to 50 for crema)
time_step_bin_width = 30 
#====

def main():
    log = open(f'logs/{log_name}.log', 'w+')
    err_log = open(f'logs/{log_name}.err', 'w+')
    sys.stderr = err_log
    sys.stdout = log

    N_SAMPLES = 5000
    FIXED_WEIGHTS = False
    if len(sys.argv) > 1:
        N_SAMPLES = int(sys.argv[1])
        if sys.argv[2] == "fixed":
            FIXED_WEIGHTS = True
        dataset = sys.argv[3]

    shape = (20, 50)
    empath = Empath(shape, threshold=5)
    if not FIXED_WEIGHTS:
        empath.draw_shared_input_weights(n_local_segments=n_signal_epochs)
        empath.save_input_weights(
            'logs/debug/learned_weights/initial_weights.pk')
    else:
        input_W = None
        with open(f"final_weights/{dstring}-weights.pk", 'rb') as f:
            input_W = pickle.load(f)
        empath.set_shared_input_weights(input_W)

    input_spike_trains = None

    X_count = 0
    print("Empath initialised. Iterating over input spike trains.")
    for pre_spikes_file in glob.glob(f"preprocessed/{dataset}_spikes/*pk"):

        start_time = time.time()
        with open(pre_spikes_file, 'rb') as f:
            input_spike_trains = pickle.load(f)
        
        #signal epochs are windows that slide over the input
        #note Empath is still stimulated one time step at a time.
        #the signal epoch affects which segment of the feature map is stimulated

        stride_len = input_spike_trains.shape[1] // n_signal_epochs
        try:
            assert not np.isnan(
                input_spike_trains).any(), f"NaN in {pre_spikes_file}"
            assert epoch_len > stride_len, \
            f"{pre_spikes_file}: choose a longer signal epoch length"
        except AssertionError as e:
            sys.stderr.write(str(e) + '\n')
            continue

        # build stride start indices
        stride_starts = np.zeros(n_signal_epochs) 
        stride_starts += stride_len
        stride_starts[:input_spike_trains.shape[1] % n_signal_epochs] += 1
        stride_starts = (np.cumsum(stride_starts) - stride_starts).astype(int)

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
                        #stimulate calls step()
                    else:
                        empath.step()
                    if not FIXED_WEIGHTS:
                        empath.update_input_weights_stdp(S.astype(int),
                                                    time_window=epoch_idx)
                    empath.pool_segments()
                    readouts.append(empath.readout())
                    recorded_spikes.append(empath.get_state())
            epoch_idx += 1

        end_time = time.time()
        debug_file_name = pre_spikes_file.split('/')[-1].split('.')[0] 
        print(f"{X_count} ({debug_file_name}): {end_time - start_time} secs") 

        if X_count % 1000 == 0 or X_count == N_SAMPLES-1:
            with open(f'logs/debug/readouts/{debug_file_name}.pk', 'wb+') as f:
                pickle.dump(readouts, f)
            recorded_spikes = np.asarray(recorded_spikes)
            with open(f'logs/debug/reservoir_spikes/{debug_file_name}.pk',
                    'wb+') as f:
                pickle.dump(recorded_spikes, f)
            if not FIXED_WEIGHTS:
                print(f"learned weights from {X_count+1} samples")
                empath.save_input_weights(
                    f'logs/debug/learned_weights/{X_count+1}_rounds_weights.pk') 

        if FIXED_WEIGHTS:
            with open(
                f"final_readouts/{dstring}/{debug_file_name}_readout.pk",
                'wb+') as f:
                pickle.dump(empath.readout(), f)

        X_count += 1
        if X_count == N_SAMPLES:
           break 
        if not FIXED_WEIGHTS and empath.check_stop_criterion():
            print("stopping criterion met")
            break
   
        empath.reset_potentials()
        empath.reset_inhibition()
        empath.reset_allow_stdp()
        empath.reset_pool() 

    if not FIXED_WEIGHTS:
        empath.save_input_weights(f'final_weights/{dstring}-weights.pk')
        print("finished learning weights")

    log.close()
    err_log.close()
if __name__ == '__main__':
    main()