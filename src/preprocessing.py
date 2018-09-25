from os import listdir
from os.path import isfile, join
import os
import numpy as np
import madmom

import time

from madmom.audio.filters import LogarithmicFilterbank
from madmom.features.beats import RNNBeatProcessor
from madmom.features.downbeats import RNNDownBeatProcessor, DBNDownBeatTrackingProcessor
from madmom.features.onsets import SpectralOnsetProcessor, RNNOnsetProcessor, CNNOnsetProcessor, spectral_flux, superflux, complex_flux
from madmom.audio.stft import ShortTimeFourierTransform
from madmom.audio.signal import FramedSignal, Signal
from madmom.audio.spectrogram import LogarithmicSpectrogram, FilteredSpectrogram, Spectrogram
from madmom.audio.cepstrogram import MFCC

from librosa.feature import spectral_centroid, spectral_bandwidth, spectral_contrast, spectral_flatness, spectral_rolloff, mfcc, rmse, zero_crossing_rate

import pdb

na = np.newaxis

NORMALIZE_CHANNELS = False

def crop_image_patches(X, h, w, hstride=1, wstride=1, return_2d_patches=False):
    N, H, W, D =  X.shape
    
    assert(h <= H and w <= W)
    
    num_patches_h = (H - h) // hstride + 1
    num_patches_w = (W - w) // wstride + 1
    
    patches = []
    for h_idx in range(num_patches_h):
        hstart = h_idx * hstride
        
        patches_w = []
        for w_idx in range(num_patches_w):
            wstart = w_idx * wstride
            
            patches_w.append(X[:,hstart:hstart + h, wstart:wstart + w, :])
            
        patches.append(patches_w)
            
    patches = np.array(patches)
    
    patches = patches.transpose(2, 0, 1, 3, 4, 5)
    
    if return_2d_patches:
        return patches.reshape(N, num_patches_h, num_patches_w, h, w, D)
    else:
        return patches.reshape(N, num_patches_h * num_patches_w, h, w, D)

def mean_pool(X, h, w):
    N, H, W, D = X.shape
    
    assert(H % h == 0 and W % w == 0)
    
    NH = H // h
    NW = W // w
    
    return X.reshape(N, NH, h, NW, w, D).mean(axis=(2, 4))

def get_spectrogram(path, sample_rate=None, fps=None, window=np.hanning, fft_sizes=[1024], filtered=True, filterbank=LogarithmicFilterbank, num_bands=12, fmin=30, fmax=17000):
    ''' 
        path: single file path
        filtered: generate FilteredSpectrogram or normal one
        
        return numpy array shaped (Frequencies, Timeframes, Channels)
        (log-spaced (Filtered)Spectrogram from madmom)
    '''
    spectros = []
    max_fft_size = np.max(fft_sizes)
    # sample_rate=None takes original sample_rate
    signal = Signal(path, sample_rate=sample_rate)
    frames = FramedSignal(signal, fps=fps)
    channel_num = 0
    for fft_size in fft_sizes:
        stft = ShortTimeFourierTransform(frames, window=window, fft_size=fft_size)
        spectro = LogarithmicSpectrogram(stft)
        if filtered:
            filtered_spectro = FilteredSpectrogram(spectro, filterbank=filterbank, num_bands=num_bands, fmin=fmin, fmax=fmax)
            spectros.append(filtered_spectro)
        else:
            spectros.append(spectro)

    # bring all spectros to the same shape, concat them and return them
    num_frequencies = max([spectro.shape[1] for spectro in spectros])
    num_channels = len(spectros)
    num_timestamps = spectros[0].shape[0]

    final_spectro = np.zeros([num_frequencies, num_timestamps, num_channels])
    for channel, spectro in enumerate(spectros):
        final_spectro[:spectro.shape[1], :, channel] = spectro.T
    return final_spectro


def get_dir_spectrograms(audio_dir, num_samples = -1, **kwargs):
    '''
        audio_dir: directory path
        num_samples: number of tracks sampled from directory
        
        return numpy array of (Frequencies, Timeframes, Channels)
    '''
    # TODO:
    # add STFT options to the spectrogram (window size etc)
    # add possibility to use different options at the same time (add depth dimension, is there a problem with the resulting shape?)

    audio_files = [f for f in listdir(audio_dir) if isfile(join(audio_dir, f))]
    if num_samples>0:
        audio_files = audio_files[:num_samples]

    # calc spectrogram for all files in the folder
    spectrograms = []
    for i, af in enumerate(audio_files):
        print("Load file {}/{} in {}".format(i+1, len(audio_files), audio_dir))
        spectrograms.append(get_spectrogram(join(audio_dir, af), **kwargs))
    spectrograms = np.array(spectrograms)
    
    return spectrograms



def get_dataset(music_dir, speech_dir, hpool=16, wpool=15, shuffle=True,
                num_samples = -1, reload=False, process_dir=get_dir_spectrograms, file_suffix="_raw",
                **kwargs):

    file_name = (music_dir + speech_dir).replace("/", "-").replace(".", "")+ file_suffix

    try:
        assert not reload
        X, Y = load_db_from_disk(file_name)
        print("loaded from disk:", file_name)
        return X, Y
    except (FileNotFoundError, AssertionError):
        print("generate dataset")
    
    music_spectros  = process_dir(music_dir, num_samples, **kwargs)
    speech_spectros = process_dir(speech_dir, num_samples, **kwargs)
    
    #print(music_spectros.shape)
    X = np.concatenate([music_spectros, speech_spectros], axis=0)#[:,:,:,:,na]
    # create labels, 1 for music, -1 for speech
    Y = ((np.arange(X.shape[0]) < music_spectros.shape[0]) - .5) * 2
    
    if hpool > 0 and wpool > 0:
        X = mean_pool(X, hpool, wpool)
    
    if shuffle:
        I = np.random.permutation(X.shape[0])
        X, Y = X[I], Y[I]

    save_to_disk(X, Y, file_name)
    return X, Y


def spectro_mini_db_patches(music_dir, speech_dir, patch_width, hpool = 16, wpool = 15, hstride=10, wstride=1, shuffle=True, max_samples = -1):
    
    X, Y = spectro_mini_db(music_dir, speech_dir, hpool=hpool, wpool=wpool, shuffle=False, max_samples = max_samples)
        
    return patch_augment(X, Y, patch_width, shuffle, max_samples)

def patch_augment(X, Y, patch_width, patch_stride = 1, shuffle=True, max_samples = -1):
        
    N, H, W, D = X.shape
    
    pos_idxs = Y > 0
    neg_idxs = np.logical_not(pos_idxs)
    
    # crop patches from the images
    X_patched_pos = crop_image_patches(X[pos_idxs], H, patch_width, wstride=patch_stride)
    X_patched_neg = crop_image_patches(X[neg_idxs], H, patch_width, wstride=patch_stride)
    
    X_patched_pos = X_patched_pos.reshape(-1, *X_patched_pos.shape[2:])
    X_patched_neg = X_patched_neg.reshape(-1, *X_patched_neg.shape[2:])

    num_pos = X_patched_pos.shape[0]
    
    X_patched = np.concatenate([X_patched_pos, X_patched_neg])
    Y_patched = ((np.arange(X_patched.shape[0]) < num_pos) - .5) * 2
    
    if shuffle:
        I = np.random.permutation(X_patched.shape[0])
        return X_patched[I], Y_patched[I]
    
    else:
        return X_patched, Y_patched

def stride_pad_multiply(signal, multiplier):
    
    if len(signal.shape) == 1:
        signal = signal[:,na]
    
    return (signal[:, na,:] * np.ones(multiplier)[na,:,na]).reshape(signal.shape[0] * multiplier, signal.shape[1])

def mean_pool_signal(signal, factor):
    '''
    mean pool (L, D) signal assuming it is a multiple of factor
    '''

    if len(signal.shape) == 1:
        signal=signal[:,na]
    L, D = signal.shape
    return signal.reshape(L//factor, factor, D).mean(axis=1)

def concatenate_and_resample(signals, sample_down=True):
    '''
    signals: list of signals, all lengths need to be multiples of the smallest length
    '''
    lengths = [len(sig) for sig in signals]
    
    if sample_down:
        min_length = min(lengths)
        resample_factors = [int(leng/min_length) for leng in lengths]
 
        downsampled_signals = [mean_pool_signal(signals[i], resample_factors[i]) for i in range(len(signals))]

        return np.concatenate(downsampled_signals, axis=1) 

    else:
        max_length = max(lengths)
        resample_factors = [int(max_length/leng) for leng in lengths]
 
        upsampled_signals = [stride_pad_multiply(signals[i], resample_factors[i]) for i in range(len(signals))]

        return np.concatenate(upsampled_signals, axis=1) 
     

       # upsampled_signals = [stride_pad_multiply(signals[i], upsample_factors[i]) for i in range(len(signals))]

def save_to_disk(X, y, file_name):
    base = "../data/processed"
    os.makedirs(base, exist_ok=True)
    x_path = join(base, file_name+"_X")
    y_path = join(base, file_name+"_y")
    np.save(x_path, X)
    np.save(y_path, y)

def load_db_from_disk(file_name):
    base = "../data/processed"
    x_path = join(base, file_name+"_X.npy")
    y_path = join(base, file_name+"_y.npy")
    return np.load(x_path), np.load(y_path)

processors = [SpectralOnsetProcessor(),
    RNNOnsetProcessor(),
    CNNOnsetProcessor(),
    SpectralOnsetProcessor(onset_method='superflux', fps=200, filterbank=LogarithmicFilterbank, num_bands=24, log=np.log10),
    RNNDownBeatProcessor(),
    lambda sig: np.array(RNNBeatProcessor(post_processor=None)(sig)).T
    ]

def rhythm_features_for_signal(signal):
    rhythm_features = [process(signal) for process in processors]
    return concatenate_and_resample(rhythm_features)

def load_and_rhythm_preprocess(audio_dir, max_samples=-1):
    print('...load and preprocess files from folder')
    audio_files = [f for f in listdir(audio_dir) if isfile(join(audio_dir, f))]

    if max_samples > 0:
        audio_files = audio_files[:max_samples]

    num_files = len(audio_files)

    processed = []
    for ind, file_name in enumerate(audio_files):
        strt = time.time()
        
        path = join(audio_dir, file_name)
        signal = Signal(path)
        processed.append(rhythm_features_for_signal(signal))
       
        stp = time.time()
        print('finished treating file {}/{} in {:4.3f}s'.format(ind+1, num_files, stp-strt))
        
    return processed

def load_rhythm_feature_db(music_dir, speech_dir, num_samples=-1, reload=False):
    file_name = (music_dir + speech_dir).replace("/", "-").replace(".", "")

    try:
        assert not reload
        return load_db_from_disk(file_name)
        print("loaded from disk")
    except (FileNotFoundError, AssertionError):
        print("generate dataset")

    music = load_and_rhythm_preprocess(music_dir, num_samples)
    music_labels = [1] * len(music)
    speech = load_and_rhythm_preprocess(speech_dir, num_samples)
    speech_labels = [-1] * len(music)

    X = np.array(music + speech)
    y = np.array(music_labels + speech_labels)
    perm = np.random.permutation(len(y))
    X, y = X[perm], y[perm]
    print("X", X.shape)
    print("y", y.shape)
    save_to_disk(X, y, file_name)

    return X, y

def normalize_channels(X):
    # divide each channel by its std dev
    if NORMALIZE_CHANNELS:
        X /= np.std(X, axis=(0,1,2))[na,na,na,...] # hehe
    return X


class RhythmData():
    def __init__(self, music_dir, speech_dir):
        X, Y = load_rhythm_feature_db(music_dir, speech_dir, num_samples=-1)

        # change -1, 1 labels to 0,1
        Y = (Y + 1) / 2 

        # X is in (N,L,D) format

        X = X[:,na,:,:] # dont conv over the number of models

        X = normalize_channels(X)

        self.X, self.Y = X, Y 

        self.num_frequencies = X.shape[1]
        self.num_timesteps   = X.shape[2]
        self.num_channels    = X.shape[3]
        self.input_shape = X[0].shape


class SpectroData():
    def __init__(self, music_dir, speech_dir):
        max_samples = -1

        X, Y = get_dataset(music_dir, speech_dir, process_dir=get_dir_spectrograms,
                           hpool=0, wpool=0, 
                           num_samples=max_samples, shuffle=True, reload=False,
                           window=np.hanning, fps=100, num_bands=3, fmin=30, fmax=17000,
                           fft_sizes=[1024, 2048, 4096]
                          )

        X = normalize_channels(X)

        Y = (Y + 1) / 2 
        self.X, self.Y = X, Y

        self.num_frequencies = X.shape[1]
        self.num_timesteps   = X.shape[2]
        self.num_channels    = X.shape[3]
        self.input_shape = X[0].shape


def get_mir(audio_path):

    hop_length = 200
    # Spectral Flux/Flatness, MFCCs, SDCs
    spectrogram = madmom.audio.spectrogram.Spectrogram(audio_path, frame_size=2048, hop_size=hop_length, fft_size=4096)
    audio = madmom.audio.signal.Signal(audio_path, dtype=float)
   
    all_features = []

    #print(spectrogram.shape)
    #print(audio.shape)
    #print('signal sampling rate: {}'.format(audio.sample_rate))
    
    # madmom features
    all_features.extend([spectral_flux(spectrogram), superflux(spectrogram), complex_flux(spectrogram)]) #, MFCC(spectrogram)])
    
    # mfcc still wrong shape as it is a 2 array
    
    # librosa features
    libr_features = [spectral_centroid(audio, hop_length=hop_length), spectral_bandwidth(audio,hop_length=hop_length), spectral_flatness(audio,hop_length=hop_length), spectral_rolloff(audio,hop_length=hop_length), rmse(audio, hop_length=hop_length), zero_crossing_rate(audio, hop_length=hop_length)]#, mfcc(audio)])
    for libr in libr_features:
        all_features.append(np.squeeze(libr, axis=0))
    # for feature in all_features:
    #     print(feature.shape)
    X = np.stack(all_features, axis=1)[na,:,:]
    return X

def get_dir_mir(audio_dir, num_samples = -1):
    '''
        audio_dir: directory path
        num_samples: number of tracks sampled from directory
        
        return numpy array of (Frequencies, Timeframes, Channels)
    '''
    # TODO:
    # add STFT options to the spectrogram (window size etc)
    # add possibility to use different options at the same time (add depth dimension, is there a problem with the resulting shape?)

    audio_files = [f for f in listdir(audio_dir) if isfile(join(audio_dir, f))]
    if num_samples>0:
        audio_files = audio_files[:num_samples]

    # calc spectrogram for all files in the folder
    mirs = []
    for i, af in enumerate(audio_files):
        print("Load file {}/{} in {}".format(i+1, len(audio_files), audio_dir))
        mirs.append(get_mir(join(audio_dir, af)))
    mirs = np.array(mirs)
    
    return mirs


class MIRData():
    def __init__(self, music_dir, speech_dir):

        max_samples = -1

        # 24 bands for superflux https://madmom.readthedocs.io/en/latest/modules/features/onsets.html?highlight=spectral_flux#madmom.features.onsets.superflux

        X, Y = get_dataset(music_dir, speech_dir, process_dir=get_dir_mir, file_suffix="_mir", num_samples=max_samples, hpool=0, wpool=0)
        X = normalize_channels(X)

        Y = (Y + 1) / 2 
        self.X, self.Y = X, Y

        self.spectral_flux = X[0]
        self.super_flux = X[1]
        self.complex_flux = X[2]
        #self.mfcc = X[3]
        # Maxi: this selects the first sample as spectral flux, the second audio file as super flux, ...

        #self.num_frequencies = X.shape[1]
        #self.num_timesteps   = X.shape[2]
        #self.num_channels    = X.shape[3]
        self.input_shape = X[0].shape


