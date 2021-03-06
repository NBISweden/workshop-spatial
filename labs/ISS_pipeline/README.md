# In Situ Sequencing Pipeline

This repository hosts a pipeline for In Situ Sequencing _(Ke R et al; 2013 Nature Methods)_ and similar image based RNA analysis methods, going through all the steps from raw data to downstream analysis.

## 1. Deconvolution

The deconvolution step of the 3D raw data allows a sharper projection of the 3D data into 2D images. We will use Flowdec _Czech, E., Aksoy, B. A., Aksoy, P., & Hammerbacher, J. (2018). Cytokit: A single-cell analysis toolkit for high dimensional fluorescent microscopy imaging. BioRxiv. https://doi.org/10.1101/460980_

__Lab files:__
 - [1_deconvolution/flowdec.ipynb](1_deconvolution/flowdec.ipynb)
 
## 2. Tiling and registration

We will use ASHLAR (https://github.com/jmuhlich/ashlar) to stitch and register tiles.

__Lab files:__
 - [2_stitching_registration/ashlar.py](2_stitching_registration/ashlar.py) [TODO]

## 3. Decoding

We will use ISTDECO _(Andersson et al; (2021). ISTDECO: In Situ Transcriptomics Decoding by Deconvolution. 10.1101/2021.03.01.433040.)_ to decode rounds and channels of fluorescence images into markers.

__Lab files:__
 - In Situ Sequencing: [3_Decoding_ISTDECO/iss_example.ipynb](3_Decoding_ISTDECO/iss_example.ipynb)
 - Merfish: [3_Decoding_ISTDECO/merfish_example.ipynb](3_Decoding_ISTDECO/merfish_example.ipynb)
 - Synthetic example: [3_Decoding_ISTDECO/synthetic_example.ipynb](3_Decoding_ISTDECO/synthetic_example.ipynb)

__OR__

We will use starfish _(Axelrod S, Carr AJ, Freeman J, Ganguli D, Long B, Tung T, and others. Starfish: Open Source Image Based Transcriptomics and Proteomics Tools, 2018-, http://github.com/spacetx/starfish)_ to decode rounds and channels of fluorescence images into markers.

__Lab files:__
 - In Situ Sequencing: [3_Decoding_starfish/iss_example.ipynb](3_Decoding_ISTDECO/iss_example.ipynb)

## 4. Quality Control

We will use TissUUmaps to perform quality control on the resulting markers.

__Lab files:__
 - [4_QC/TissUUmaps_example.ipynb](4_QC/TissUUmaps_example.ipynb) [TODO]

## 5. Downstream Analysis

We will use scanpy and squidpy to perform some analysis of our data.

__Lab files:__
 - CSS to AnnData: [5_Analysis_tools/squidpy_example.ipynb](5_Analysis_tools/squidpy_example.ipynb)
 - Squidpy: [TODO]