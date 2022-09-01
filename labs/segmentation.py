from urllib.parse import urlparse
import matplotlib.pyplot as plt
#import matplotlib as mpl
#%matplotlib inline
#mpl.rcParams['figure.dpi'] = 300
from cellpose import utils, io
from cellpose import models, io
# DEFINE CELLPOSE MODEL
# model_type='cyto' or model_type='nuclei'
model = models.Cellpose(gpu=False, model_type='nuclei')
import skimage
from scipy import ndimage as ndi
from skimage import (
    io, color, feature, filters, measure, morphology, segmentation, util
)
import numpy as np
from skimage.segmentation import watershed, expand_labels
import scipy
from scipy.sparse import (
    coo_matrix, save_npz, load_npz
)
import tifffile
import pandas as pd
from skimage.filters import threshold_multiotsu
from skimage.measure import label, regionprops

def stardist_segmentation(image_path, output_path , model_name = '2D_versatile_fluo',expand_cells = True, n_tiles = (4,4), expanded_distance = 20,):
    import matplotlib.pyplot as plt
    
    import pandas as pd
    from glob import glob
    from tifffile import imread
    from csbdeep.utils import Path, normalize
    from csbdeep.io import save_tiff_imagej_compatible

    from stardist import random_label_cmap, _draw_polygons, export_imagej_rois
    from stardist.models import StarDist2D
    import tifffile as tff
    from scipy.sparse import (
    coo_matrix, save_npz, load_npz
    )
    from skimage import (
        io, color, feature, filters, measure, morphology, segmentation, util
    )
    
    model = StarDist2D.from_pretrained(model_name)
    from skimage.segmentation import watershed, expand_labels
    
    image = tff.imread(image_path)
    print(image.shape)
    print('normalize image')
    image = normalize(image, 1,99.8)
    print('predict instance')
    labels, details = model.predict_instances(image,n_tiles = n_tiles) # 4,4
    print('label image')
    labels = measure.label(labels)
    
    if expand_cells == True: 
        print('expand image')
        expanded = expand_labels(labels, distance=expanded_distance)
        coo = coo_matrix(expanded)
        print('save output')
        save_npz(output_path+'/stardist_segmentation_expanded.npz',coo, compressed=True)
        save_npz(output_path+'/stardist_segmentation_labels.npz',coo_matrix(labels), compressed=True)
    else: 
        coo = coo_matrix(labels)
        print('save output')
        save_npz(output_path+'/stardist_segmentation_labels.npz',coo_matrix(labels), compressed=True)




def cell_pose_segemenation_to_coo(image, diam, expanded_distance):
    
    '''
    function to segement nuclei and expand segemented nuclei using cell pose. 
    cellpose is a generalist algorithm for cellular segmentation. the function returns a 
    coo object which saves the outlines of the cells. 
    
    the input for the function is the dapi stained images, the diameter of the objects 
    to segment and the distance the to expand (we generally do a 10% expansion). 
    
    the segementation startegy employs the scikit-image package and watershed
    segementation. 
    from the docs: 
    "The watershed is a classical algorithm used for segmentation, that is, for separating 
    different objects in an image. Starting from user-defined markers, the watershed algorithm
    treats pixels values as a local topography (elevation).The algorithm floods basins from the 
    markers until basins attributed to different markers meet on watershed lines. In many cases, 
    markers are chosen as local minima of the image, from which basins are flooded."
    '''
    
    # run cell pose segementation on the objects 
    masks_nuclei, flows, styles, diams = model.eval(image,diameter=diam)
    
    distance = ndi.distance_transform_edt(masks_nuclei)
    local_max_coords = feature.peak_local_max(distance, min_distance=7)
    local_max_mask = np.zeros(distance.shape, dtype=bool)
    local_max_mask[tuple(local_max_coords.T)] = True

    # find markers
    markers = measure.label(local_max_mask)

    # run watershed segementation
    segmented_cells = segmentation.watershed(-distance, markers, mask=masks_nuclei)
    seg1 = measure.label(segmented_cells)
    expanded = expand_labels(seg1, distance=expanded_distance)
    expanded_new = expanded.astype('uint32')
    coo = coo_matrix(expanded_new)
        
    return expanded_new, coo


import time, os, sys
from urllib.parse import urlparse
import matplotlib.pyplot as plt
#import matplotlib as mpl
#%matplotlib inline
#mpl.rcParams['figure.dpi'] = 300
from cellpose import utils, io
from cellpose import models, io
# DEFINE CELLPOSE MODEL
# model_type='cyto' or model_type='nuclei'
model = models.Cellpose(gpu=False, model_type='nuclei')
import skimage
from scipy import ndimage as ndi
from skimage import (
    io, color, feature, filters, measure, morphology, segmentation, util
)
import numpy as np
from skimage.segmentation import watershed, expand_labels
import scipy
from scipy.sparse import (
    coo_matrix, save_npz, load_npz
)
import tifffile
import pandas as pd
from skimage.filters import threshold_multiotsu
from skimage.measure import label, regionprops

def segment_tile(sample_folder, 
                segment = True, 
                dapi_channel = 5, 
                diam = 40, 
                expanded_distance = 30,
                big_section = False, 
                output_file_name='cellpose_segmentation.npz', 
                 expand_tile = False
                ):
    print(sample_folder)
    output_path = sample_folder+'/cell_segmentation/'
    if not os.path.exists(output_path):
            os.makedirs(output_path)
    tiles_segmented = os.listdir(output_path)
    path = sample_folder+'/preprocessing/ReslicedTiles/Base_5_stitched-'+str(dapi_channel)+'/'
    tiles_to_segment = os.listdir(path)
    print(len(tiles_to_segment))
    if segment == True: 
        print('segmenting')
        for i, tile in enumerate(tiles_to_segment): 
            if tile.split('.')[0]+'.npz' in tiles_segmented:
                continue
            else: 
                print(tile)
                dapi = io.imread(path + str(tile))
                # segment and expand objects
                coo = cell_pose_segemenation_to_coo(image = dapi, diam = diam, expanded_distance = expanded_distance)
                # save segemenation in coo
                scipy.sparse.save_npz(output_path+str(tile.split('.')[0])+'.npz', coo[1], compressed=True)
    else: 
        print('not segmenting')
    tiles = pd.read_csv(sample_folder+"/preprocessing/ReslicedTiles/tilepos.csv", header = None)

    for i in tiles[1].unique():
        print(i)
        tiles_to_vercat = tiles[tiles[1] == i]
        str_tiles_to_vercat = list((tiles_to_vercat.index+1).astype(str))

        a = []
        for j,coo in enumerate(str_tiles_to_vercat):
                mask = load_npz(output_path+"/tile"+coo+".npz").toarray()
                if expand_tile == True: 
                    mask = expand_labels(mask, expanded_distance)
                a.append(mask)
        concatenated = np.concatenate(tuple(a), axis=1)
        coo_to_save = coo_matrix(concatenated)
        save_npz(output_path+"/tiles_"+str(i)+".npz", coo_to_save)

    top = list(tiles[1].unique())[:round(len(tiles[1].unique())/2)]
    bottom  = list(tiles[1].unique())[round(len(tiles[1].unique())/2):]

    if big_section == True: 
        print('splitting')
        top = list(tiles[1].unique())[:round(len(tiles[1].unique())/2)]
        bottom  = list(tiles[1].unique())[round(len(tiles[1].unique())/2):]

        a_top = []
        for i in top:
            print(i)
            coo_top = load_npz(output_path+"/tiles_"+str(i)+".npz")
            a_top.append(coo_top.toarray())
        concatenated_top = np.concatenate(tuple(a_top), axis=0)
        coo_to_save_top = coo_matrix(concatenated_top)

        a_bottom = []
        for i in bottom:
            print(i)
            coo_bottom = load_npz(output_path+"/tiles_"+str(i)+".npz")
            a_bottom.append(coo_bottom.toarray())
        concatenated_bottom = np.concatenate(tuple(a_bottom), axis=0)
        coo_to_save_bottom = coo_matrix(concatenated_bottom)
    else: 
        print('not splitting')
        a = []
        for i in tiles[1].unique():
            print(i)
            coo = load_npz(output_path+"/tiles_"+str(i)+".npz")
            a.append(coo.toarray())
        concatenated = np.concatenate(tuple(a), axis=0)

    concatenated_relabled = label(concatenated)
    coo_to_save = coo_matrix(concatenated_relabled)
    save_npz(output_path+output_file_name, coo_to_save)




def hex_to_rgb(hex):
    hex=hex[1:]
    return tuple(int(hex[i:i+2], 16) for i in (0, 2, 4))
    
def plot_segmentation_mask_colored(ad_sp,
                                    coo_file,
                                    color_column,
                                   positions,
                                   output_file,
                                  ):

    # import packages
    import scanpy as sc
    import pandas as pd
    from scipy.sparse import load_npz
    from scipy.sparse import coo_matrix
    import skimage
    from skimage.color import label2rgb
    import numpy as np
    import matplotlib.pyplot as plt

    # load data
    coo_file = coo_file
    coo = load_npz(coo_file)
    array = coo.toarray()

    # subset image
    image_subset=array[positions[0]:positions[1],positions[2]:positions[3]]
    rgb_label_image = label2rgb(image_subset, bg_label=0)

    Cell_Num = ad_sp.obs['CellID']#pd.DataFrame(ad_sp.obs.index)['CellID'].str.split('_',expand = True)[3].astype(int)
    ad_sp.obs['CellID'] = list(Cell_Num)
    ad_sp.obs['col_rgb'] = [hex_to_rgb(item) for item in ad_sp.obs[color_column]]
    # subset anndata object
    ad_sp_int = ad_sp[ad_sp.obs['CellID'].astype(int).isin(image_subset.flatten())]

    # color image
    
    filtered_cells = dict(zip(ad_sp_int.obs['CellID'].astype(int), ad_sp_int.obs['col_rgb']))
    values = (np.unique(image_subset.flatten()))

    colors_empty = np.zeros((values.max()+1, 3)).astype(int)
    for i in filtered_cells:
        colors_empty[i] = np.array(filtered_cells[i])

    colored_image = colors_empty[image_subset]
    with plt.rc_context({'figure.figsize': (20, 20)}):
        plt.imshow(colored_image)
        #lt.gca().invert_yaxis()
        #plt.axis('off')
        mpl.rcParams['pdf.fonttype'] = 42
        mpl.rcParams['ps.fonttype'] = 42
        plt.rcParams['svg.fonttype'] = 'none'
        plt.savefig(output_file, dpi = 600)
        plt.show()
         
    
