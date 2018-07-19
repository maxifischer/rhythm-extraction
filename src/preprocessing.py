import numpy as np

def crop_image_patches(X, h, w, hstride=1, wstride=1, return_2d_patches=False):
    N, H, W, D =  X.shape
    
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