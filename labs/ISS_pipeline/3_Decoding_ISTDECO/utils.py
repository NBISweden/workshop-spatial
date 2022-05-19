import numpy as np
from itertools import product
from typing import Union, Tuple
from scipy.ndimage import grey_dilation, gaussian_filter


def gaussian_filter(image,sigma):
    out = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            out[i,j] = gaussian_filter(image[i,j], sigma)
    return out

def nonmaxsuppress2D(X,radius=1.5):
    from scipy.ndimage import grey_dilation
    footprint = disk_strel(radius)
    # Small petrubation to avoid adjacent maxima of equal height.
    n = np.random.rand(*X.shape) * 1e-9
    mask = (grey_dilation((X+n), footprint=footprint) == (X+n))
    return mask

def nonmaxsuppress1D(X,radius=1.5):
    from scipy.ndimage import grey_dilation
    footprint = np.ones((1,1,2*int(radius)+1))
    # Small petrubation to avoid adjacent maxima of equal height.
    n = np.random.rand(*X.shape) * 1e-9
    mask = (grey_dilation((X+n), footprint=footprint) == (X+n))
    return mask



def __truncated_gaussian(im_shape: Tuple[int,int], yx, sigma):
    trunc_radius = np.ceil(3*np.max(np.array(sigma)))
    r = np.arange(-trunc_radius,trunc_radius+1,1) 
    if im_shape[0] == 1:
        x, y = np.meshgrid(r+ np.floor(yx[1]), np.zeros_like(r))
    else:
        x, y = np.meshgrid(r+ np.floor(yx[1]), r  + np.floor(yx[0]))

    t1 = np.divide(x-yx[1], sigma)**2
    t2 = np.divide(y-yx[0], sigma)**2
    v = np.exp(-0.5*(t1 + t2))
    return v.ravel(), y.ravel().astype('int'), x.ravel().astype('int')

def random_codebook(n_codes=50, n_rounds=4, n_channels=3):
    '''
        Generates a random codebook of shape (m, r, c).
        m - number of codes,
        r - number of rounds,
        c - number of channels

    '''

    assert n_codes > 0, 'Number of codes must be greater than 0'
    assert n_rounds > 0, 'Number of sequencing rounds must be greater than 0'
    assert n_channels > 0, 'Number of channels must be greater than 0'
    words = np.eye(n_channels, dtype='uint8')
    codebook_full = np.array([v for v in product(words, repeat = n_rounds)])\
        .reshape((n_channels**n_rounds, n_rounds, n_channels))
    ind = np.random.choice(np.arange(n_channels**n_rounds), n_codes, replace=False)
    codebook = codebook_full[ind]
    return codebook

def random_image_stack(
    codebook: np.ndarray,
    n_signals: int,
    ndim: int,
    signal_std:float = 1.2,
    im_shape:Tuple[int,int]=(50,50), 
    avg_signal_intensity:int=50,
    signal_variation:float=0.2,
    avg_photon_background:int=2,
    background_electron_std:float=0.5,
    ):

    '''
        Creates a random image stack of shape (r, c, ny, nx)


        Parameters:
        ----------
            Codebook : np.array, 
                Codebook of shape (m, r, c)

            n_signals : int
                Number of signals in the stack

            
            im_shape : tuple(int, int)
                Shape of the image stack

            avg_signal_intensity : float
                Average intensity of signals

            signal_variation : float
                Signal intensities are sampled from 
                avg_signal_intensity * (1+r), 
                r ~ U(-avg_signal_intensity,avg_signal_intensity)

            avg_photon_background : int
                Rate of Poisson noise

            background_electron_std : float
                Standard deviation of gausian noise

    '''

    assert codebook.ndim == 3, 'Codebook must be of shape (n_codes, n_rounds, n_channels)'
    assert n_signals > 0, 'Number of signals rounds must be greater than 0'
    assert avg_signal_intensity > 0, 'Average signal intensity must be greater than 0'
    assert signal_variation >= 0 and signal_variation <= 1, 'Signal variation must be between 0 and 1'
    assert avg_photon_background > 0, 'Average number of background photons must be greater than 0'
    assert background_electron_std >= 0, 'Background electrons must be non negative'
    'Left over signal scale interval must be a tuple on the form (LOW, HIGH)'

    assert ndim == 1 or ndim == 2, 'Number of dimensions must be 1 or 2'

    # Parse stuff
    if ndim == 1:
        im_shape = (1, np.max(np.array(im_shape))) 
        spatial_dims = np.array([0,1])
    else:
        spatial_dims = np.array([1,1])

    n_codes, n_rounds, n_channels = codebook.shape
    spatial_shape = im_shape
    spectral_shape = (n_rounds, n_channels)
        
    image_data = np.zeros(spectral_shape + spatial_shape, 'float32')
    image_noise_data = np.zeros(spectral_shape + spatial_shape, 'float32')
    barcode_ind = np.random.choice(np.arange(0,n_codes), size=n_signals)   
    padding = np.ceil(3*signal_std) * spatial_dims

    yx = np.array([np.random.uniform(low=padding[i],
        high=s-padding[i]-1, 
        size=n_signals) for i,s in enumerate(im_shape)]).T

    print(yx.max())
    gt = np.append(np.fliplr(yx), barcode_ind.reshape((-1,1)), axis=1)
    yx = np.kron(yx,np.ones((n_rounds,1)))
    signal_barcodes = codebook[barcode_ind]
    _, r, ch = np.where(signal_barcodes)

    n_spots = n_signals*n_rounds
    high = avg_signal_intensity + signal_variation*avg_signal_intensity
    low = avg_signal_intensity - signal_variation*avg_signal_intensity
    intensities = np.random.uniform(high=high,low=low, size=n_spots)
    for i in range(len(intensities)):
        gaussian, ii, jj = __truncated_gaussian(im_shape, yx[i], signal_std)
        image_data[r[i],ch[i],ii,jj] = image_data[r[i],ch[i],ii,jj] + gaussian*intensities[i]

    image_noise_data = image_noise_data + np.random.poisson(lam=avg_photon_background, size=image_noise_data.shape)
    image_noise_data = image_noise_data + np.random.normal(scale=background_electron_std, size=image_noise_data.shape)
    image_data = image_data + image_noise_data
    image_data[image_data<0] = 0
    return image_data, gt

