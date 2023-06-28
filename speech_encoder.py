'''
Encodes speech using spikes under a biologically plausible coding scheme.
---
1. Obtain cochlear-like frequency filters reconstructed in the time domain.
    - Constant Q filters -> real-valued impulse response (wavelets) 
2. Perform time domain convolution of each wavelet with input signal to yield
a signal decomposed into time-frequency domain.
    - Notice by using wavelet transforms, we could generate a spike for each 
    convolution in real time in a neuromorphic implementation.
4. Window the decomposed signal and find logarithmic energies in each window. 
5. Convert energies to spike times using latency code

---
reference: doi: 10.1109/IJCNN.2018.8489434
'''

import librosa
import numpy as np
import sys
import glob
import pickle

def read_wav(path: str):
    '''
    Given path to a single channel wav file,
    return numpy array of frames and file's frame rate.
    '''
    frames_array, frame_rate = librosa.load(path, sr=None)
    return (frames_array, frame_rate)

def cochlear_wavelets(sampling_rate):
    '''
    Obtain cochlear-like frequency filters *reconstructed in time domain*.
    1. Construct cochlear-like filterbank. Using constant Q bases.
    2. Find impulse response of each filter in time domain. Using ifft.
    3. Return time-domain wavelets and wavelet lengths.

    ---
    Constant Q transform is similar to short-time Fourier transform, only 
    frequency bins are logarithmically spaced, and lower frequency bins have a
    wider sample range than the higher frequency bins. Relative powers and 
    temporal resolution from each bin therefore better approximate the 
    perception of pitch by human ear. 
    '''

    n_filters = 20
    bins_per_octave=2.1 #covers 15Hz - 8kHz
    fmin=15

    center_frequencies = librosa.cqt_frequencies(
        n_bins=n_filters,
        fmin = fmin,
        bins_per_octave=bins_per_octave,
        tuning=0.0 #adjusts f_min. don't care about matching notes.
        )
    
    cq_filterbank, cq_filter_lengths = librosa.filters.wavelet(
        freqs=center_frequencies,
        sr = sampling_rate,
        window='hann',#smoothen using Hann window function.
        filter_scale=1,#scale each filter length (N_k) i.e. vary time resolution
        pad_fft=True,#Center-pad filters with zeroes.
        norm=1,#normalise each filter as required by CQT.
        gamma=0, #constant Q
        dtype=np.complex64
    )

    # spectral_widths = (2 ** (1/bins_per_octave) - 1) * center_frequencies
    print("Cochlear filterbank central frequencies range from " + 
          f'{center_frequencies[0]} to {center_frequencies[-1]:.2f} Hz')

    cq_filter_lengths = np.ceil(cq_filter_lengths).astype(int) #N_k's 
    print("Cochlear filterbank shape: ", cq_filterbank.shape)

    '''
    Apply inverse fft to each cq filter. This is to obtain wavelets in the time
    domain.
    '''
    time_domain_wavelets = [None]*n_filters
    for k in range(20):
        start = cq_filterbank.shape[1] // 2 - (
            np.ceil(cq_filter_lengths[k] / 2).astype(int))
        # with this start index
        # even-length filters look like: [0ccc0]
        # odd-length filters look like: [0ccc]
        # print(cq_filterbank[k][start],
            #   cq_filterbank[k][start+cq_filter_lengths[k]-1],
            #   cq_filter_lengths[k])

        F = cq_filterbank[k][start:start+cq_filter_lengths[k]]

        # enforce symmetry in the filter about the max value
        # symm obtains real-valued impulse response from the filter after ifft 
        center = np.ceil(len(F)/2).astype(int)
        assert(max(F)== F[center]) 
        F = F[:center+1]
        F = np.concatenate((F, np.flip(np.conj(F[1:-1]))))
        f = np.fft.ifft(F)
        time_domain_wavelets[k] = np.real_if_close(f)
        assert time_domain_wavelets[k].dtype == np.float64 #checks real-valued

    # odd-length filters after enforcing symmetry now have +1 length 
    wavelet_lengths = np.where(cq_filter_lengths % 2 == 0,
                               cq_filter_lengths,
                               cq_filter_lengths+1) 
    print(f"Wavelet lengths range from",
          f"{wavelet_lengths[0]} to {wavelet_lengths[-1]}")

    return time_domain_wavelets, wavelet_lengths 

def cochlear_convolution(signal, wavelets, wavelet_lengths):
    '''
    Obtain a time-domain convolution between input speech signal
    and cochlear  wavelets. 
    '''
    max_len = np.max(wavelet_lengths)
    padded_signal = np.pad(signal, ((0,max_len)))

    time_domain_convolution = np.zeros((20, len(signal)))

    for k in range(20):
        for n in range(len(signal)):
            #Extract N_k samples from the signal.
            signal_window = padded_signal[n:n+wavelet_lengths[k]]
            #Convolve i.e. dot product signal window and wavelet.
            time_domain_convolution[k,n] = np.dot(
                signal_window, wavelets[k])

    print(f"Signal shape: {signal.shape}",
           f"Convolution shape: {time_domain_convolution.shape}")
    return time_domain_convolution 

def windowed_energies(y):
    '''
    Window each filtered signal and find energies in each window.
    Can be thought of as downsampling. There is room for this since cochlear
    filters' freq ranges are already limited compared to 16kHz sr.
    '''
    window_len = 512 
    stride_len = window_len // 2
    #With this stride, we downsample to 32 / window_len kHz
    #window_len 512 comes out to 0.0625 kHz 
    #Later, since one spike per window, encoding spike rate = 62.5 spikes/sec
      
    sig_len = y.shape[1]
    n_windows = np.ceil(sig_len / stride_len).astype(int)
    energies = np.zeros((20, n_windows))

    for k in range(20):
        for i in range(0, sig_len ,stride_len):
            window = y[k, i:i+window_len] 
            energy = np.log(np.sum(np.square(window)))
            idx = i // stride_len
            energies[k, idx] = energy
    print(f"Windowed energies shape: {energies.shape}")
    return energies

def spike_latency_coding(energies):
    '''
    Obtain a spike train for each windowed decomposed signal using latency 
    coding.
    ---
    Each window is one time step. There is one spike per time step. 
    Fractional energy inside a window is the inverse of the spike time's latency 
    relative to the beginning of each time step.
    '''

    # normalise each window and place between 0 and 1
    energies = energies + np.abs(np.min(energies,axis=0))
    energies = np.divide(energies, np.max(energies, axis=0))

    # latency in each encoding window will be the complement of energy 
    # (place larger energies near 0, smaller energies near 1)
    energies = np.abs(energies - 1)
    
    #generate latencies
    latencies = np.zeros(energies.shape)
    latencies[:,1:] = 1 
    latencies = np.cumsum(latencies, axis=1) 
    latencies += np.round(energies, decimals = 4) #resolution: 0.1 ms

    # Each time step is (window_len / 32)/1000 sec
    # for window_len 512, this comes out to 0.016 seconds long.

    sorted_latencies = np.sort(latencies, axis=0)
    sorted_latencies = np.flip(sorted_latencies,axis=0)
    is_sorted = latencies == sorted_latencies
    print(f"Number of time steps with perfectly sorted latencies (strange?):",
          f"{is_sorted.all(axis=0).sum()}") 

    return latencies

def main():
    log = open('logs/speech_encoder.log', 'w+')
    sys.stdout = log
    wav_paths = glob.glob('data/Crema/*.wav')
    sampling_rate = 16000
    wavelets, wavelet_lengths = cochlear_wavelets(sampling_rate)
    
    for wav_path in wav_paths:

        audio_signal, wav_sampling_rate = read_wav(wav_path)
        assert sampling_rate == wav_sampling_rate, "mismatched sampling rate"

        time_domain_convolution = cochlear_convolution(
            audio_signal, 
            wavelets,
            wavelet_lengths) 
    
        energies = windowed_energies(time_domain_convolution)
        spike_trains = spike_latency_coding(energies)

        print("\nSpike times in first five windows:")
        print(spike_trains[:,:5])

        spikes_fname = wav_path.split('/')[-1].split('.wav')[0] + '_spks'
        with open(f'preprocessed/crema_spikes/{spikes_fname}.pk', 'wb+') as f:
            pickle.dump(spike_trains, f)
    
    log.close()

if __name__ == '__main__':
    main()