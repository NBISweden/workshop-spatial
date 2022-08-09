import os, errno
import pandas as pd

from slicedimage import ImageFormat
from starfish.experiment.builder import format_structured_dataset


def fstr(template, **kwargs):
    return eval(f"f'{template}'", kwargs)

def force_symlink(src, dst):
    try:
        os.symlink(src, dst)
    except OSError as e:
        if e.errno == errno.EEXIST:
            os.remove(dst)
            os.symlink(src, dst)

def format_data(base_dir,
                out_dir,
                n_rounds, 
                n_channels,
                n_zplanes,
                fov_names,
                dapi_channel,
                file_format):
    """
    
    """
    primary_channels = set(range(n_channels))^set([dapi_channel])

    # primary ACTG
    primary_dir = os.path.join(out_dir, "primary")
    if not os.path.isdir(primary_dir):
        os.mkdir(primary_dir)
        
    for r in range(n_rounds):
        for chi, ch in enumerate(primary_channels):
            for fov in fov_names:
                for z in range(n_zplanes):
                    src=os.path.join(base_dir, fstr(file_format, r=r, ch=ch, fov=fov, z=z))
                    dst=os.path.join(primary_dir,fstr("primary-f{fov}-r{r}-c{chi}-z{z}.tif", r=r, chi=chi, fov=fov, z=z))
                    force_symlink(src=os.path.abspath(src),dst=os.path.abspath(dst))

    # nuclei
    nuclei_dir = os.path.join(out_dir, "nuclei")
    if not os.path.isdir(nuclei_dir):
        os.mkdir(nuclei_dir)

    for r in range(n_rounds):
        for ch in [dapi_channel]: #only one channel, the dapi channel
            for fov in fov_names:
                for z in range(n_zplanes):
                    src=os.path.join(base_dir, fstr(file_format, r=r, ch=ch, fov=fov, z=z))
                    dst=os.path.join(nuclei_dir,fstr("nuclei-f{fov}-r{r}-c{ch}-z{z}.tif", r=r, ch=0, fov=fov, z=z))
                    force_symlink(src=os.path.abspath(src),dst=os.path.abspath(dst))
    
    return(primary_dir, nuclei_dir)


def create_coordinates(primary_dir,
                       nuclei_dir,
                       n_rounds,
                       n_channels,
                       n_zplanes,
                       fov_names,
                       xy_max,
                       z_max,
                       fov_coordinates):

    n_fovs = len(fov_names)
    n_channels = n_channels - 1 # separate DAPI from the rest of the channels

    fovs_primary = [[(r, ch, z) for r in range(n_rounds) for ch in range(n_channels) for z in range(n_zplanes)]]*n_fovs
    fovs_nuclei = [[(r, ch, z) for r in range(n_rounds) for ch in [0] for z in range(n_zplanes)]]*n_fovs

    # conversion dictionary to make sure all the formats are correct
    convert_dict = {
                'fov': int,
                'round': int,
                'ch': int,
                'zplane': int,
                'xc_min': float,
                'xc_max': float,
                'yc_min': float,
                'yc_max': float,
                'zc_min': float,
                'zc_max': float}

    # primary metadata
    primary_df = pd.DataFrame(columns=['fov','round','ch','zplane','xc_min','yc_min','zc_min','xc_max','yc_max','zc_max'])
    for f, fov in enumerate(range(n_fovs)):
        x_pos = fov_coordinates[f][0]
        y_pos = fov_coordinates[f][1]
        z_pos = fov_coordinates[f][2]
        fov_info = fovs_primary[fov]
        for i in range(len(fov_info)):
            primary_df = pd.concat([primary_df, pd.DataFrame.from_records([{
                'fov': fov_names[fov],
                'round': fov_info[i][0],
                'ch': fov_info[i][1],
                'zplane': fov_info[i][2],
                'xc_min': x_pos,
                'xc_max': x_pos + xy_max,
                'yc_min': y_pos,
                'yc_max': y_pos + xy_max,
                'zc_min': z_pos,
                'zc_max': z_pos + z_max
            }])], ignore_index=True)

    primary_df = primary_df.astype(convert_dict)
    primary_df.to_csv(os.path.join(primary_dir, 'coordinates.csv'), index=None)
        

    # nuclei metadata
    nuclei_df = pd.DataFrame(columns=['fov','round','ch','zplane','xc_min','yc_min','zc_min','xc_max','yc_max','zc_max'])
    for f, fov in enumerate(range(n_fovs)):
        x_pos = fov_coordinates[f][0]
        y_pos = fov_coordinates[f][1]
        z_pos = fov_coordinates[f][2]
        fov_info = fovs_nuclei[fov]
        for i in range(len(fov_info)):
            nuclei_df = pd.concat([nuclei_df, pd.DataFrame.from_records([{
                'fov': fov_names[fov],
                'round': fov_info[i][0],
                'ch': fov_info[i][1],
                'zplane': fov_info[i][2],
                'xc_min': x_pos,
                'xc_max': x_pos + xy_max,
                'yc_min': y_pos,
                'yc_max': y_pos + xy_max,
                'zc_min': z_pos,
                'zc_max': z_pos + z_max
            }])], ignore_index=True)


    nuclei_df = nuclei_df.astype(convert_dict)
    nuclei_df.to_csv(os.path.join(nuclei_dir, 'coordinates.csv'), index=None)

    return(primary_df, nuclei_df)


def space_tx(output_dir,
             primary_dir,
             nuclei_dir):

    primary_out = os.path.join(output_dir, "primary")
    nuclei_out = os.path.join(output_dir, "nuclei")

    format_structured_dataset(
        primary_dir,
        os.path.join(primary_dir, "coordinates.csv"),
        primary_out,
        ImageFormat.TIFF
    )
    
    format_structured_dataset(
        nuclei_dir,
        os.path.join(nuclei_dir, "coordinates.csv"),
        nuclei_out,
        ImageFormat.TIFF
    )

    with open(os.path.join(primary_out, "experiment.json"), "r+") as fh:
        contents = fh.readlines()
        contents[3] = ",".join([contents[3].strip("\n"),"\n"])
        contents.insert(4, '\t"nuclei": "../nuclei/nuclei.json"\n')  # new_string should end in a newline
        fh.seek(0)  # readlines consumes the iterator, so we need to start over
        fh.writelines(contents)  # No need to truncate as we are increasing filesize
        fh.seek(0)

    return
