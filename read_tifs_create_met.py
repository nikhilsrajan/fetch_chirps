import os
import pandas as pd
import geopandas as gpd
import rasterio
import rasterio.merge
import tqdm
import numpy as np
import multiprocessing as mp
import functools

import rsutils.utils as utils
import fetch_missing_chirps_files as fmcf


COL_METHOD = 'method'


class LoadTIFMethod:
    READ_AND_CROP = 'read and crop'
    READ_NO_CROP = 'read no crop'
    COREGISTER_AND_CROP = 'coregister and crop'
    

def coregister_and_maybe_crop(
    tif_filepath:str,
    reference_tif_filepath:str,
    working_folderpath:str = None,
    resampling=rasterio.merge.Resampling.nearest,
    nodata=None,
    shapes_gdf:gpd.GeoDataFrame = None,
):
    os.makedirs(working_folderpath, exist_ok=True)

    _filename = os.path.split(tif_filepath)[1]
    zero_tif_filepath = utils.add_epochs_prefix(
        filepath = reference_tif_filepath,
        prefix = f'zero+{_filename}_',
        new_folderpath = working_folderpath,
    )

    utils.create_zero_tif(
        reference_tif_filepath = reference_tif_filepath,
        zero_tif_filepath = zero_tif_filepath,
    )

    coregistered_tif_filepath = utils.add_epochs_prefix(
        filepath = tif_filepath,
        prefix = f'coregistered+{_filename}_',
        new_folderpath = working_folderpath
    )

    utils.coregister(
        src_filepath = tif_filepath,
        dst_filepath = coregistered_tif_filepath,
        reference_zero_filepath = zero_tif_filepath,
        resampling = resampling,
        nodata = nodata,
    )

    if shapes_gdf is not None:
        out_image, out_meta = utils.crop_tif(
            src_filepath = coregistered_tif_filepath,
            shapes_gdf = shapes_gdf,
        )
    else:
        with rasterio.open(coregistered_tif_filepath) as src:
            out_image = src.read()
            out_meta = src.meta.copy()
    
    os.remove(zero_tif_filepath)
    os.remove(coregistered_tif_filepath)

    return out_image, out_meta


def load_tif(
    tif_filepath:str,
    working_folderpath:str = None,
    shapes_gdf:gpd.GeoDataFrame = None,
    reference_tif_filepath:str = None,
    method:str = LoadTIFMethod.READ_NO_CROP,
    resampling = rasterio.merge.Resampling.nearest,
    nodata = None,
):
    if method == LoadTIFMethod.READ_NO_CROP:
        with rasterio.open(tif_filepath) as src:
            out_image = src.read()
            out_meta = src.meta.copy()

    elif method == LoadTIFMethod.READ_AND_CROP:
        if shapes_gdf is None:
            raise ValueError(f'shapes_gdf can not be None for method={method}')
        out_image, out_meta = utils.crop_tif(
            src_filepath=tif_filepath,
            shapes_gdf=shapes_gdf,
        )

    elif method == LoadTIFMethod.COREGISTER_AND_CROP:
        if shapes_gdf is None:
            raise ValueError(f'shapes_gdf can not be None for method={method}')
        if reference_tif_filepath is None:
            raise ValueError(f'reference_tif_filepath can not be None for method={method}')
        out_image, out_meta = coregister_and_maybe_crop(
            tif_filepath = tif_filepath,
            reference_tif_filepath = reference_tif_filepath,
            resampling = resampling,
            nodata = nodata,
            shapes_gdf = shapes_gdf,
            working_folderpath = working_folderpath,
        )

    return out_image, out_meta


def get_centre_value(ndarray:np.ndarray):
    return np.take(ndarray, ndarray.size // 2)


AGGREGATION_DICT = {
    'mean': np.nanmean,
    'median': np.nanmedian,
    'centre': get_centre_value
}


def read_tif_get_agg_value(
    filepath:str,
    filetype:str,
    method:str,
    multiplier:float,
    aggregation:str,
    shapes_gdf:gpd.GeoDataFrame,
    working_folderpath:str,
    reference_tif_filepath:str=None,
):
    aggregation_func = AGGREGATION_DICT[aggregation]

    if filetype == fmcf.EXT_TIF:
        tif_filepath = filepath
    elif filetype == fmcf.EXT_TIF_GZ:
        gzip_file = utils.GZipTIF(
            gzip_tif_filepath = filepath
        )
        tif_filepath = gzip_file.decompress_and_load()
    else:
        raise NotImplementedError(f'New filetype: {filetype}')

    out_image, out_meta = load_tif(
        tif_filepath = tif_filepath,
        shapes_gdf = shapes_gdf,
        reference_tif_filepath = reference_tif_filepath,
        method = method,
        working_folderpath = working_folderpath,
    )

    out_image = out_image * multiplier

    # CHIRPS NODATA value = -9999
    out_image[out_image == -9999] = np.nan

    value = aggregation_func(out_image)

    del out_image, out_meta

    if filetype == fmcf.EXT_TIF_GZ:
        gzip_file.delete_tif()
        del gzip_file

    return value


def read_tif_get_agg_value_by_tuple(
    filepath_filetype_method_multiplier:tuple[str,str,str,float],
    shapes_gdf:gpd.gpd.geopandas,
    working_folderpath:str,
    aggregation:str = 'mean',
    reference_tif_filepath:str = None,
):
    filepath, filetype, method, multiplier = filepath_filetype_method_multiplier
    return read_tif_get_agg_value(
        filepath = filepath,
        working_folderpath = working_folderpath,
        filetype = filetype,
        method = method,
        multiplier = multiplier,
        aggregation = aggregation,
        shapes_gdf = shapes_gdf,
        reference_tif_filepath = reference_tif_filepath,
    )


def read_tifs_get_agg_value(
    catalogue_df:pd.DataFrame,
    shapes_gdf:gpd.geopandas,
    val_col:str,
    working_folderpath:str,
    method_col:str = COL_METHOD,
    tif_filepath_col:str = fmcf.COL_TIF_FILEPATH,
    filetype_col:str = fmcf.COL_FILETYPE,
    multiplier_col:str = fmcf.COL_MULTIPLIER,
    aggregation:str = 'mean',
    reference_tif_filepath:str = None,
    njobs:int = mp.cpu_count() - 2,
):  
    if aggregation not in AGGREGATION_DICT.keys():
        raise ValueError(f'Invalid aggregation={aggregation}. Valid aggregations: {AGGREGATION_DICT.keys()}')

    updated_catalogue_df = catalogue_df.copy(deep=True)

    if aggregation == 'centre':
        """
        - alter the geometry to a box [DONE]
        - get the centroid pixel coordinate [TO DO]
        """
        shapes_gdf['geometry'] = shapes_gdf.envelope

    read_tif_get_agg_value_by_tuple_partial = functools.partial(
        read_tif_get_agg_value_by_tuple,
        shapes_gdf = shapes_gdf,
        aggregation = aggregation,
        reference_tif_filepath = reference_tif_filepath,
        working_folderpath = working_folderpath,
    )

    filepath_filetype_method_multiplier_tuples = list(zip(
        catalogue_df[tif_filepath_col],
        catalogue_df[filetype_col],
        catalogue_df[method_col],
        catalogue_df[multiplier_col],
    ))

    with mp.Pool(njobs) as p:
        values = list(tqdm.tqdm(
            p.imap(
                read_tif_get_agg_value_by_tuple_partial, 
                filepath_filetype_method_multiplier_tuples,
            ), 
            total=len(filepath_filetype_method_multiplier_tuples)
        ))
        
    updated_catalogue_df[val_col] = values
    
    return updated_catalogue_df
