#! /usr/bin/env python

import numpy as np
import rasterio
from affine import Affine

def mask_by_shp(geom,array,ds,invert=False):
    """
    retrive date from input raster array falling within input polygon
    
    Parameters
    -------------
    geom: shapely.geometry 
        shapefile within which to return raster values
        if using geopandas GeoDataFrame, input should be geometry column like gdf['geometry']
    array: np.ma.array
        masked array of input raster
    ds: gdal or rasterio dataset
        dataset information for input array
        used in computing geotransform
    
    Returns
    -------------
    masked_array: np.ma.array
        input array containing non-masked values for only regions falling within input geometry
    """
    
    if (type(ds) == rasterio.io.DatasetReader):
        transform = ds.transform
    else:
        transform = Affine.from_gdal(*ds.GetGeoTransform())
    shp = rasterio.features.rasterize(geom,out_shape=np.shape(array),fill=-9999,transform=transform,dtype=float)
    shp_mask = np.ma.masked_where(shp==-9999,shp)
    if invert:
        masked_array = np.ma.array(array,mask=~shp_mask.mask)
    else:
        masked_array = np.ma.array(array,mask=shp_mask.mask)
    return np.ma.fix_invalid(masked_array)

