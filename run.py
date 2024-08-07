import sys
import geopandas as gpd
import multiprocessing as mp

import chcfetch.constants
import fetch_missing_chirps_files as fmcf
import read_tifs_create_met as rtcm


def parse_args():
    roi_shapefile = sys.argv[1]
    start_year = int(sys.argv[2])
    end_year = int(sys.argv[3])
    geoglam_chirps_data_folderpath = sys.argv[4]
    chc_chirps_v2_0_p05_download_folderpath = sys.argv[5]
    reference_tif_filepath = sys.argv[6]
    aggregation = sys.argv[7]
    njobs = int(sys.argv[8])
    export_filepath = sys.argv[9]
    working_folderpath = sys.argv[10]

    return roi_shapefile, start_year, end_year, \
        geoglam_chirps_data_folderpath, \
        chc_chirps_v2_0_p05_download_folderpath, \
        reference_tif_filepath, \
        aggregation, njobs, export_filepath, \
        working_folderpath


def check_if_any_geom_within_chirps_bounds(
    shapes_gdf:gpd.GeoDataFrame
):
    chirps_bounds_gdf = gpd.read_file(
        chcfetch.constants.CHIRPS_V2_P50_BOUNDS_GEOJSON_FILEPATH
    )

    chirps_bounds_epsg_4326 = chirps_bounds_gdf['geometry'].iloc[0]
    
    any_roi_within_chirps_bounds = shapes_gdf.to_crs(
        chirps_bounds_gdf.crs
    ).within(chirps_bounds_epsg_4326).any()

    return any_roi_within_chirps_bounds


if __name__ == '__main__':
    """
    args:
    - roi_shapefile
    - start_year
    - end_year
    - geoglam_chirps_folderpath
    - chc_chirps_download_folderpath
    - reference_tif_filepath
    - aggregation
    - njobs
    - export_filepath
    - working_folderpath
    """

    roi_shapefile, start_year, end_year, \
    geoglam_chirps_data_folderpath, \
    chc_chirps_v2_0_p05_download_folderpath, \
    reference_tif_filepath, \
    aggregation, njobs, export_filepath, \
    working_folderpath, \
    = parse_args()

    shapes_gdf = gpd.read_file(roi_shapefile)
    
    years = list(range(start_year, end_year + 1))

    if njobs == -1:
        njobs = mp.cpu_count() - 2

    VAL_COl = f'{aggregation} CHIRPS'

    catalogue_df = fmcf.fetch_missing_chirps_v2p0_p05_files(
        years = years,
        geoglam_chirps_data_folderpath = geoglam_chirps_data_folderpath,
        chc_chirps_v2_0_p05_download_folderpath = chc_chirps_v2_0_p05_download_folderpath,
        njobs = njobs,
    )

    catalogue_df[rtcm.METHOD_COL] = rtcm.LoadTIFMethod.READ_AND_CROP
    catalogue_df.loc[
        catalogue_df[fmcf.FILETYPE_COL] == fmcf.TIF_GZ_EXT,
        rtcm.METHOD_COL
    ] = rtcm.LoadTIFMethod.COREGISTER_AND_CROP

    updated_catalogue_df = rtcm.read_tifs_get_agg_value(
        shapes_gdf = shapes_gdf,
        catalogue_df = catalogue_df,
        val_col = VAL_COl,
        reference_tif_filepath = reference_tif_filepath,
        aggregation = aggregation,
        njobs = njobs,
        working_folderpath = working_folderpath,
    )

    updated_catalogue_df[[
        fmcf.DATE_COL,
        fmcf.YEAR_COL,
        fmcf.DAY_COL,
        VAL_COl,
    ]].to_csv(export_filepath, index=False)

