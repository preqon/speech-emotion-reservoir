'''
Encodes speech using spikes under a biologically plausible coding scheme.
'''
# 1. digital signal
# 2. cochlear filter bank
# 3. spectrogram
# 4. spike pattern
# 5. masked spike pattern

import librosa
import numpy as np
import sys

def read_wav(path: str):
    '''
    Given path to a single channel wav file,
    return numpy array of frames and file's frame rate.
    '''
    frames_array, frame_rate = librosa.load(path, sr=None)
    return (frames_array, frame_rate)

def constant_q_transform(audio_signal, sampling_rate):
    '''
    Constant q transform. Similar to stft, only frequency bins are 
    logarithmically spaced, and lower frequency bins have a wider range than
    the higher frequency bins. Energies in each bin therefore better approximate
    the perception of pitch by human ear.
    '''
    cqt_matrix = librosa.cqt(
        audio_signal,
        sr=sampling_rate,
        hop_length=38,
        fmin=15,
        n_bins=20, 
        bins_per_octave=3,
        window='hann',
        tuning = 0.0,
        sparsity=0,
        pad_mode='constant',
        )
    wavelet_size = cqt_matrix.shape[1] 
    #I think usually wavelet size varies for each freq bin, but librosa's
    #implementation keeps them consistent.
    return cqt_matrix, wavelet_size

def ifft_each_filter(cqt_matrix, n):
    '''
    Apply inverse fft to each filter produced by cqt. This is to obtain
    real valued wavelets in the time domain.
    Number of samples n in each reconstructed signal is kept to wavelet size (?) 
    '''
    decomposed_signal = np.zeros((20, n))
    for idx, freq_bin in enumerate(cqt_matrix):
        decomposed_signal[idx] = np.fft.ifft(freq_bin, n=n)
    return decomposed_signal

def cochlear_wavelets(audio_signal, sampling_rate):
    '''
    Obtain wavelets of cochlear filters.
    1. Construct cochlear-like filter bank. Using CQT.
    2. Find impulse response (wavelets) of each filter. Using ifft.
    '''
    cqt_matrix, wavelet_size = constant_q_transform(audio_signal, sampling_rate)
    decomposed_signal = ifft_each_filter(cqt_matrix, n=wavelet_size) 
    return decomposed_signal, wavelet_size

def cochlear_convolution(signal, decomposed_signal, window_len):
    '''
    Obtain a time-domain convolution between input speech signal
    and cochlear filter wavelets. 
    '''
    padded_signal = np.pad(signal, ((0,window_len)))
    time_domain_convolution = np.zeros((20, len(signal)))
    for k in range(20):
        for n in range(len(signal)):
            signal_window = padded_signal[n:n+window_len]
            decomposed_window = decomposed_signal[k]
            time_domain_convolution[k,n] = np.dot(
                signal_window, decomposed_window)
    return time_domain_convolution 

def windowed_energies(y):
    '''
    Window the time domain convolution and find energies in each window.
    '''
    window_len = 960 #60 possible spikes per second 
    energies = np.zeros((20, y.shape[1]//2)) #stride length l/2
    for k in range(20):
        for i in range(0,y.shape[1],window_len//2):
            window = y[k, i:i+window_len] 
            e = np.log(np.sum(np.square(window)))
            idx = i // (window_len // 2)
            energies[k, idx] = e
    return energies

def spike_latency_coding(energies):
    '''
    Obtain spike train for each filter using latency coding.
    '''
    n_time_steps = 30 #30 seconds per 1000 possible spikes.
    #place between 0 and 1.
    energies = energies + np.abs(np.min(energies,axis=0))
    energies = np.divide(energies, np.max(energies, axis=0), where=energies>0)
    #invert axis (larger energies near 0, smaller energies near 1)
    energies = np.abs(energies - 1)
    #place between 0 and 30
    energies = energies * n_time_steps
    #generate latencies
    latencies = np.zeros(energies.shape)
    latencies[:,1:] = n_time_steps
    latencies = np.cumsum(latencies, axis=1)
    #resolution: 0.1 ms
    latencies += np.round(energies, decimals = 4)
    return latencies

def main():
    path = "data/Crema/1001_DFA_ANG_XX.wav"

    audio_signal, sampling_rate = read_wav(path)

    decomposed_signal, wavelet_size = cochlear_wavelets(
        audio_signal,
        sampling_rate)

    time_domain_convolution = cochlear_convolution(
        audio_signal, 
        decomposed_signal,
        wavelet_size)

    energies = windowed_energies(time_domain_convolution)
    spike_trains = spike_latency_coding(energies)

    print("Spike times in first five windows.")
    print(spike_trains[:,:5])

if __name__ == '__main__':
    main()