import os, shutil
import numpy as np
import pandas as pd

from PIL import Image

from starfish.types import Axes

import matplotlib.pyplot as plt

def normalize(X, q):
    X_scaled = (X - np.percentile(X,q)) / (np.percentile(X,(100-q)) - np.percentile(X,q))
    X_scaled[X_scaled<0] = 0
    X_scaled[X_scaled>1] = 1
    return X_scaled

def qc_csv(experiment, spot_intensities, output_path):

    """
    Create a CSV file from a starfish experiments compatible with the TissUUmaps "Spot Insepector" plugin

    Parameters:
        experiment (starfish Experiment): containing the codebook
        spot_intensities (starfish DecodedIntensityTable): result of the decoding process
        output_path (string): directory where you wish to save the images
    """
    
    output_path = os.path.abspath(output_path)
    csv_path = os.path.join(output_path, 'pixel_decoding.csv')
    rounds_gt = []
    channels_gt = []

    x = spot_intensities.x.values
    y = spot_intensities.y.values
    target_names = spot_intensities.target.values

    for name in target_names:
        idx = np.where(experiment.codebook.target.values==name)[0][0]
        rs = np.where(experiment.codebook.values[idx]>0)[0]
        rounds_gt.append(';'.join([str(r) for r in rs]))
        chans = []
        for r in np.unique(rs):
            ch = experiment.codebook.values[idx][r]
            chans.append(np.argwhere(ch == np.amax(ch)))
        chs = np.concatenate(chans)
        channels_gt.append(';'.join([str(c[0]) for c in chs]))

    df = pd.DataFrame(np.stack([x,y,target_names,rounds_gt,channels_gt]).transpose(), columns=['x','y','target_name','rounds','channels'])
    df.to_csv(csv_path)

    return csv_path

def qc_images(filtered_imgs, dapi_imgs, output_path):

    """
    Creates the images from a starfish experiments compatible with the TissUUmaps "Spot Insepector" plugin

    Parameters:
        filtered_imgs (starfish ImageStack): image stack after filtering and deconvolving the data
        experiment (starfish Experiment): containing the the nuclei images
        output_path (string): directory where you wish to save the images
    """

    image_names = []
    output_path = os.path.abspath(output_path)

    for r in range(filtered_imgs.num_rounds):
        for c in range(filtered_imgs.num_chs):
            im = np.squeeze(filtered_imgs.sel({Axes.CH: c, Axes.ROUND: r}).xarray.values)
            im = normalize(im, 1)
            im = np.uint8(255*im)
            im = Image.fromarray(im)
            image_name = 'Round{}_{}.tif'.format(r,c)
            image_path = os.path.join(output_path, image_name)
            im.save(image_path)
            image_names.append(image_path)
        try:
            dapi = np.squeeze(dapi_imgs.sel({Axes.ROUND: r}).xarray.values)
        except:
            dapi = np.squeeze(dapi_imgs.sel({Axes.ROUND: 0}).xarray.values)
        dapi = normalize(dapi, 1)
        dapi = np.uint8(255*dapi)
        dapi = Image.fromarray(dapi)
        dapi_name = 'Round{}_DAPI.tif'.format(r)
        dapi_path = os.path.join(output_path, dapi_name)
        dapi.save(dapi_path)
        image_names.append(dapi_path)

    return image_names

def compare_images(imgs_1, imgs_2, n=1):
    for i in range(n):
        c, r = np.random.randint(0,imgs_1.num_chs), np.random.randint(0,imgs_1.num_rounds)
        
        plt.figure(figsize=(12,5))
        plt.subplot(1,2,1)
        im = np.squeeze(imgs_1.sel({Axes.CH: c, Axes.ROUND: r}).xarray.values)
        plt.imshow(im[int(im.shape[0]/2-200):int(im.shape[0]/2+200),int(im.shape[1]/2-200):int(im.shape[1]/2+200)])
        plt.title(f'Round {r} - Channel {c} - Original Image')
        plt.axis('off')
        plt.colorbar()
        
        plt.subplot(1,2,2)
        im = np.squeeze(imgs_2.sel({Axes.CH: c, Axes.ROUND: r}).xarray.values)
        plt.imshow(im[int(im.shape[0]/2-200):int(im.shape[0]/2+200),int(im.shape[1]/2-200):int(im.shape[1]/2+200)])
        plt.title(f'Round {r} - Channel {c} - Filtered Image')
        plt.axis('off')
        plt.colorbar()
        plt.tight_layout()
        