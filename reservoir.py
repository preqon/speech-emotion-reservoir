#INPUT ENCODING

#1.  Mel-Frequency Spectral Coefficients
#2. M X N array of neurons
# each row is a time frame, each column a frequency band
    # time frame m, freq band n

 #3. time to first spike encoding
    # shut off the neuron as soon as it fires a spike.
    # i.e., time of first spike of neuron (m,n) encodes intensity of frequency 
    #band n in the time frame m.
        #(higher intensity, earlier neuron fires)

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

import wave
import numpy as np
import sys
import librosa

def read_wav(path):
    '''
    Given path to a single channel wav file.
    Return numpy array of frames and file's frame rate.
    '''
    # using librosa which does some type conversion of each frame from my wavs' 
    # int16 to float, don't think there should be any resulting issues.
    frames_array, frame_rate = librosa.load(path, sr=None)
    return (frames_array, frame_rate)

def mel_freq_spectral_coeffs(frames_array, frame_rate):
    '''
    Equivalent to MFCC without the final discrete cosine transform.
    1. Find short time fourier transform.
    2. Find power spectrogram.
    3. Map power spectrogram to the mel scale.
    4. Return logarithm of mel-scale spectrogram.
    '''
    # references:
    # https://medium.com/@tanveer9812/mfccs-made-easy-7ef383006040
    # http://practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/

    # pad signal so varying n_frames can be chosen such that it is close to 
    # 512 for average signal length and is divisible by 4.
    remainder = len(frames_array) % 240
    n_zeroes = 240 - remainder if remainder else 0
    frames_array = np.pad(frames_array, (0, n_zeroes), 'constant')
    
    #short time fourier transform
        #applies discrete fft in short overlapping windows.
    n_fft = len(frames_array) // 60 #number of points in each fft i.e. length of each window.
        #varying n_fft to lead to fixed input size
    stft = librosa.stft(
        y=frames_array,
        n_fft = n_fft, 
        hop_length= n_fft//4, #number of points to slide each window by
        win_length= n_fft, #number of points sampled in each window.
            #if smaller than n_fft, the rest of the window is padded with zeros.
        window = 'hann', #windowing function. (smooths each window somehow)
        center = False,
        #         If ``True``, the signal ``y`` is padded so that frame
        #         ``D[:, t]`` is centered at ``y[t * hop_length]``.
        #         If ``False``, then ``D[:, t]`` begins at ``y[t * hop_length]``
        #Don't think having the first few windows consistently zero padded 
        #adds any value. Last few windows will be zero padded in any case.
        pad_mode='constant' #y is padded on both ends with zeros if center=True
    )

    #power spectrogram
    power_spectrogram = np.abs(stft)**2

    # map spectrogram to mel-scale (equal diffs in pitch sound equally distant)
        #create set of mel triangular filters
    mel_filter_bank = librosa.filters.mel(sr=frame_rate, n_fft=n_fft)
        #multiply each filter by the power spectrogram and sum the coefficients
        #gives mel energies over time
    mel_spectrogram = np.einsum(
        "...ft,mf->...mt", power_spectrogram, mel_filter_bank, optimize=True)
    # Compress large energies with log, such that loudness makes it harder to 
    # tell pitch apart.
    mel_log_spectrogram = np.log10(mel_spectrogram)
    print(mel_log_spectrogram.shape)

def main():
    paths = [
         "data/Crema/1001_DFA_ANG_XX.wav",
         "data/Crema/1001_IEO_HAP_LO.wav",
         "data/Crema/1001_IEO_ANG_HI.wav"
    ]
    for path in paths:
        frames_array, frame_rate = read_wav(path)
        mel_freq_spectral_coeffs(frames_array, frame_rate)

if __name__ == "__main__":
    main()
