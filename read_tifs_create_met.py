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


METHOD_COL = 'method'


class LoadTIFMethod:
    READ_AND_CROP = 'read and crop'
    READ_NO_CROP = 'read no crop'
    COREGISTER_AND_CROP = 'coregister and crop'
    

def coregister_and_maybe_crop(
    tif_filepath:str,
    reference_tif_filepath:str,
    resampling=rasterio.merge.Resampling.nearest,
    nodata=None,
    shapes_gdf:gpd.GeoDataFrame = None,
):
    filename = os.path.split(tif_filepath)[1]
    zero_tif_filepath = utils.add_epochs_prefix(
        filepath = reference_tif_filepath,
        prefix = f'zero+{filename}_',
    )

    utils.create_zero_tif(
        reference_tif_filepath = reference_tif_filepath,
        zero_tif_filepath = zero_tif_filepath,
    )

    coregistered_tif_filepath = utils.add_epochs_prefix(
        filepath = tif_filepath,
        prefix = f'coregistered+{filename}_',
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
        )

    return out_image, out_meta


AGGREGATION_DICT = {
    'mean': np.mean,
    'median': np.median,
}


def read_tif_get_agg_value(
    filepath:str,
    filetype:str,
    method:str,
    aggregation:str,
    shapes_gdf:gpd.GeoDataFrame,
    reference_tif_filepath:str=None,
):
    aggregation_func = AGGREGATION_DICT[aggregation]

    if filetype == fmcf.TIF_EXT:
        tif_filepath = filepath
    elif filetype == fmcf.TIF_GZ_EXT:
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
    )

    value = aggregation_func(out_image)

    del out_image, out_meta

    if filetype == fmcf.TIF_GZ_EXT:
        gzip_file.delete_tif()
        del gzip_file

    return value


def read_tif_get_agg_value_by_tuple(
    filepath_filetype_method:tuple[str,str,str],
    shapes_gdf:gpd.gpd.geopandas,
    aggregation:str = 'mean',
    reference_tif_filepath:str = None,
):
    filepath, filetype, method = filepath_filetype_method
    return read_tif_get_agg_value(
        filepath = filepath,
        filetype = filetype,
        method = method,
        aggregation = aggregation,
        shapes_gdf = shapes_gdf,
        reference_tif_filepath = reference_tif_filepath,
    )


def read_tifs_get_agg_value(
    catalogue_df:pd.DataFrame,
    shapes_gdf:gpd.geopandas,
    val_col:str,
    method_col:str = METHOD_COL,
    tif_filepath_col:str = fmcf.TIF_FILEPATH_COL,
    filetype_col:str = fmcf.FILETYPE_COL,
    aggregation:str = 'mean',
    reference_tif_filepath:str = None,
    njobs:int = mp.cpu_count() - 2,
):  
    if aggregation not in AGGREGATION_DICT.keys():
        raise ValueError(f'Invalid aggregation={aggregation}. Valid aggregations: {AGGREGATION_DICT.keys()}')

    updated_catalogue_df = catalogue_df.copy(deep=True)

    read_tif_get_agg_value_by_tuple_partial = functools.partial(
        read_tif_get_agg_value_by_tuple,
        shapes_gdf = shapes_gdf,
        aggregation = aggregation,
        reference_tif_filepath = reference_tif_filepath,
    )

    filepath_filetype_method_tuples = list(zip(
        catalogue_df[tif_filepath_col],
        catalogue_df[filetype_col],
        catalogue_df[method_col]
    ))

    with mp.Pool(njobs) as p:
        values = list(tqdm.tqdm(
            p.imap(
                read_tif_get_agg_value_by_tuple_partial, 
                filepath_filetype_method_tuples,
            ), 
            total=len(filepath_filetype_method_tuples)
        ))
        
    updated_catalogue_df[val_col] = values
    
    return updated_catalogue_df
