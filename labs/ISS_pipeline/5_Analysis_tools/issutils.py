import numpy as np
from sklearn.cluster import MiniBatchKMeans as KMeans
from scipy.spatial import cKDTree
from scipy.sparse import csc_matrix
import pandas as pd



def create_gene_signatures(
    gene_labels, 
    gene_position_xy, 
    spatial_scale, 
    gene_labels_unqiue=None, 
    bin_stride=None, 
    bin_type='gaussian_per_point'):
    
    '''
        Computes gene expression signatures in different "empty" bins by inpainting neighbouring gene expression markers.
        The inpainting is done by:
            1. 
                Each "empty" bin is connected to its neighbouring, "known", gene expression bins through edges.
                A known gene expression bin is a one-hot-encoded vector that labels a particular gene.
                Each edge has a weight that is given by exp(-0.5 d_ij^2 / sigma^2 ), where d_ij is the distance
                between the empty node i and the known node j.
            2. The gene expression of the the empty bin is obtained by a weighted sum of the neighbouring known bins.

        The empty bins can either be placed on a rectangular grid or ontop of each known node.

        Input:
            - gene_labels: The categorical label of each gene, (#genes long np.array).
            - gene_position_xy: Position of each gene (#genes x 2 size np.array)
            - Spatial scale, i.e., the sigma of a Gaussian kernel. Scalar
            - gene_labels_unique: unique list of labels. If not provided, it is automagically computed using np.unique(gene_labels)
            - bin_stride: Stride between bins. If not provided it is set to spatial scale. 
            - bin_type: option for setting the type of bins. If set to gaussian_grid it will use bins on a grid. If set to gaussian_per_point
                it will compute bins centered around each point in gene_position_xy.
        Output:
            Gene vectors (#genes x #gene_types) gene profile vectors
            Coord       (#genes x 2) spatial coordinate of each vector.
    '''

    ok_bin_types = ['gaussian_grid', 'gaussian_per_point']

    if bin_type not in ok_bin_types:
        raise ValueError(f'bin_type must be one of the following: {ok_bin_types}')
        
    if bin_stride is None:
        bin_stride = spatial_scale
    
    if gene_labels_unqiue is None:
        gene_labels_unqiue = np.unique(gene_labels)

    # Create unique numeric labels
    gene_labels_unique_map = {str(k) : i for i,k in enumerate(gene_labels_unqiue)}
    gene_labels_numeric = np.array([gene_labels_unique_map[str(k)] for k in gene_labels])

    n_genes = len(gene_labels_unqiue)
    n_pts = gene_position_xy.shape[0]
    if bin_type == 'gaussian_grid':
        # Compute grid start and stop
        start = gene_position_xy.min(axis=0); stop = gene_position_xy.max(axis=0)
        # Compute location of each bin
        y,x = np.meshgrid(np.arange(start[1],stop[1], bin_stride),
                np.arange(start[0],stop[0], bin_stride))         
        bin_coords = np.array([x.ravel(), y.ravel()]).T
        # Compute gene vectors
        vectors = np.eye(n_genes)
        p =  cKDTree(gene_position_xy).query_ball_point(bin_coords, spatial_scale*3, p=2)
        q = [np.ones(len(l))*i for i,l in enumerate(p) if len(l) > 0]
        p = np.concatenate(p).astype('int')
        q = np.concatenate(q).astype('int')
        # Affinities
        aff = np.sum((gene_position_xy[p,:] - bin_coords[q,:])**2, axis=1)
        aff = np.exp(- 0.5 * aff  / (spatial_scale**2))     
        # Sum neighbours   
        gene_each_point_one_hot = np.array([vectors[:,gene_labels_numeric[i]] for i in range(n_pts)])
        gene_vectors = csc_matrix((aff, (q, p)), shape=(bin_coords.shape[0], n_pts)) @ gene_each_point_one_hot
    elif bin_type == 'gaussian_per_point':
        bin_coords = gene_position_xy.copy()
        vectors = np.eye(n_genes)
        p =  cKDTree(gene_position_xy).query_ball_point(bin_coords, spatial_scale*3, p=2)
        q = [np.ones(len(l))*i for i,l in enumerate(p) if len(l) > 0]
        p = np.concatenate(p).astype('int')
        q = np.concatenate(q).astype('int')
        aff = np.sum((gene_position_xy[p,:] - bin_coords[q,:])**2, axis=1)
        aff = np.exp(- 0.5 * aff  / (spatial_scale**2))        
        gene_each_point_one_hot = np.array([vectors[:,gene_labels_numeric[i]] for i in range(n_pts)])
        gene_vectors = csc_matrix((aff, (q, p)), shape=(bin_coords.shape[0], n_pts)) @ gene_each_point_one_hot
    return gene_vectors, bin_coords, gene_labels_unqiue

def preprocess_vectors(gene_vectors, logtform=True, normalize=True, ord=1):
    '''
        The parameter ord is the order of the normalization
        and must be one of the following:

        inf    max(abs(x))
        0      sum(x != 0)
        1      sum(abs(x))
        2      sqrt(sum(x^2))
    '''
    if logtform:
        gene_vectors = np.log(gene_vectors + 1)
    if normalize:
        gene_vectors = gene_vectors / np.linalg.norm(gene_vectors, ord=ord, axis=1,keepdims=True)
    return gene_vectors

def cluster_signatures(gene_vectors, n_clusters=6, threshold=None, logtform=True, normalize=True, ord=1):
    '''
        Cluster gene vectors (#ngenes x #gene types).
    '''
    kmeans =  KMeans(n_clusters).fit(gene_vectors)
    labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_
    return labels, cluster_centers

def create_adata(data,
    spatial_scale: float,
    spatial_columns: list =['x','y'],
    bin_type: str = 'gaussian_per_point',
    gene_type_column: str ='label',
    obs_columns: list = None,
    unique_labels: list=None,
):
    from anndata import AnnData
    
    labels = data[gene_type_column].to_numpy()
    xy = data[spatial_columns].to_numpy()
    gene_vectors, bin_coords, labels_unique = create_gene_signatures(labels, xy, spatial_scale, unique_labels, bin_type=bin_type)
    var = pd.DataFrame(labels_unique)
    var.columns = ['gene_name']

    if bin_type == 'gaussian_per_point':
        if obs_columns is None:
            obs_columns = [gene_type_column]
        else:
            obs_columns = obs_columns.insert(0, gene_type_column)

        if bin_coords is not None:
            adata = AnnData(gene_vectors, obsm={"spatial": bin_coords}, var=var, obs=data[obs_columns])
        else:
            adata = AnnData(gene_vectors, var=var, obs=data[obs_columns])
    else:
        if bin_coords is not None:
            adata = AnnData(gene_vectors, obsm={"spatial": bin_coords}, var=var)
        else:
            adata = AnnData(gene_vectors, var=var)
    adata.var_names = labels_unique
    return adata


if __name__ == "__main__":
    import pandas as pd
    import matplotlib.pyplot as plt
    np.random.seed(42)

    print('Loading data ...')
    data = pd.read_csv('example_data.csv')
    xy = data[['x','y']].to_numpy()
    labels = data['name'].to_numpy()
    
    # Cell radius is approx 2.5 micormeters. 
    # Pixel size is approx 0.32 micrometers per pixel
    sigma = 2.5 / 0.32
    
    print('Clustering ...')
    gene_vectors, bin_coords, gene_labels_unique = create_gene_signatures(labels,xy,sigma, bin_type='gaussian_per_point')
    gene_vectors = preprocess_vectors(gene_vectors)
    cluster, centroids = cluster_signatures(gene_vectors, n_clusters=8)
    
    print('Plotting ...')
    plt.figure(figsize=(16, 16)); 
    plt.scatter(bin_coords[:,0], bin_coords[:,1], s=1, c=cluster); 
    plt.legend()
    plt.show()
