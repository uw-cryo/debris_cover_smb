#! /usr/bin/env python

import os,sys
import numpy as np
import rasterio
from pygeotools.lib import iolib,geolib
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

def clip_raster_by_shp_disk(r_fn,shp_fn,extent='raster',invert=False,out_fn=None):
    """
    # this is a lightweight version of directly being used from https://github.com/dshean/pygeotools/blob/master/pygeotools/clip_raster_by_shp.py
    # meant to limit subprocess calls
    """
    if not os.path.exists(r_fn):
        sys.exit("Unable to find r_fn: %s" % r_fn)
    if not os.path.exists(shp_fn):
        sys.exit("Unable to find shp_fn: %s" % shp_fn)
     #Do the clipping
    r, r_ds = geolib.raster_shpclip(r_fn, shp_fn,extent=extent,invert=invert)
    if not out_fn:
        out_fn = os.path.splitext(r_fn)[0]+'_shpclip.tif'
    iolib.writeGTiff(r, out_fn, r_ds)
    
