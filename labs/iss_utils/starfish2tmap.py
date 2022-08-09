import os
import numpy as np
import pandas as pd

from PIL import Image

from starfish.types import Axes

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
        rounds_gt.append('{};{};{};{}'.format(rs[0],rs[1],rs[2],rs[3]))
        chans = []
        for r in np.unique(rs):
            ch = experiment.codebook.values[idx][r]
            chans.append(np.argwhere(ch == np.amax(ch)))
        chs = np.concatenate(chans)
        channels_gt.append('{};{};{};{}'.format(chs[0][0],chs[1][0],chs[2][0],chs[3][0]))

    df = pd.DataFrame(np.stack([x,y,target_names,rounds_gt,channels_gt]).transpose(), columns=['x','y','target_name','rounds','channels'])
    df.to_csv(csv_path)

    return csv_path



def qc_images(filtered_imgs, exp, output_path):

    """
    Creates the images from a starfish experiments compatible with the TissUUmaps "Spot Insepector" plugin

    Parameters:
        filtered_imgs (starfish ImageStack): image stack after filtering and deconvolving the data
        output_path (string): directory where you wish to save the images
    """

    image_names = []
    dapis = exp['fov_001'].get_image('nuclei')
    output_path = os.path.abspath(output_path)

    for r in range(filtered_imgs.num_rounds):
        for c in range(filtered_imgs.num_chs):
            im = np.squeeze(filtered_imgs.sel({Axes.CH: c, Axes.ROUND: r}).xarray.values)
            im = Image.fromarray(im)
            image_name = 'Round{}_{}.tif'.format(r,c)
            image_path = os.path.join(output_path, image_name)
            im.save(image_path)
            image_names.append(image_path)
        dapi = np.squeeze(dapis.sel({Axes.ROUND: r}).xarray.values)
        dapi = Image.fromarray(dapi)
        dapi_name = 'Round{}_{}.tif'.format(r,4)
        dapi_path = os.path.join(output_path, dapi_name)
        dapi.save(dapi_path)
        image_names.append(dapi_path)

    return image_names