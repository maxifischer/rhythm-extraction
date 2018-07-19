from os import listdir
from os.path import isfile, join

import numpy as np
import madmom

na = np.newaxis

def crop_image_patches(X, h, w, hstride=1, wstride=1, return_2d_patches=False):
    N, H, W, D =  X.shape
    
    assert(h <= H and w <=W)
    
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
    
    return X.reshape(N, NH * h, NW * w, D).reshape(N, NH, h, NW, w, D).mean(axis=(2, 4))

def spectros_from_dir(audio_dir, max_samples = -1):

    # TODO:
    # add STFT options to the spectrogram (window size etc)
    # add possibility to use different options at the same time (add depth dimension, is there a problem with the resulting shape?)
    
    audio_files = [f for f in listdir(audio_dir) if isfile(join(audio_dir, f))]
    
    if max_samples > 0:
        audio_files = audio_files[:max_samples]
    
    # TODO: see above, could be handled here
    spectro_function = lambda path: madmom.audio.spectrogram.Spectrogram(path).log()

    # calc spectrogram for all files in the folder
    spectrograms = np.array([spectro_function(join(audio_dir, af)) for af in audio_files])

    # transorm to N, H, W shape
    spectrograms = spectrograms.transpose(0, 2, 1) 
    
    return spectrograms

def spectro_mini_db(music_dir, speech_dir, hpool=16, wpool=15, shuffle=True, max_samples = -1):
    
    music_spectros  = spectros_from_dir(music_dir, max_samples)
    speech_spectros = spectros_from_dir(speech_dir, max_samples)
    
    X = np.concatenate([music_spectros, speech_spectros], axis=0)[:,:,:,na]
    
    # create labels, 1 for music, -1 for speech
    Y = ((np.arange(X.shape[0]) < music_spectros.shape[0]) - .5) * 2
    
    if hpool > 0 and wpool > 0:
        X = mean_pool(X, hpool, wpool)
    
    if shuffle:
        I = np.random.permutation(X.shape[0])
        
        return X[I], Y[I]
    else:
        return X, Y

def spectro_mini_db_patches(music_dir, speech_dir, patch_width, hpool = 16, wpool = 15, hstride=10, wstride=1, shuffle=True, max_samples = -1):
    X, Y = spectro_mini_db(music_dir, speech_dir, hpool=hpool, wpool=wpool, shuffle=False, max_samples = max_samples)
        
    N, H, W, D = X.shape
    
    pos_idxs = Y > 0
    neg_idxs = np.logical_not(pos_idxs)
    
    # crop patches from the images
    X_patched_pos = crop_image_patches(X[pos_idxs], H, patch_width)
    X_patched_neg = crop_image_patches(X[neg_idxs], H, patch_width)
    
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