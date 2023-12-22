#! /usr/bin/env python
import os,sys,glob
import geopandas as gpd
import rasterio
from shapely.geometry import box
from pygeotools.lib import iolib
from debris_cover_smb import geospatial 
data_dir = iolib.get_datadir()

farinotti_thickness_dir = os.path.join(data_dir,'farinotti_2019_ice_thickness')
rounce_debris_thickness_dir = os.path.join(data_dir,'debris_thickness')
millan_thickness_dir = os.path.join(data_dir,'millan_thickness')
millan_velocity_dir = os.path.join(data_dir,'millan_velocity')
rgi_dir = os.path.join(data_dir,'rgi60/regions/rgi15/')
hma_ara_crs = '+proj=aea +lat_0=36 +lon_0=85 +lat_1=25 +lat_2=47 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs'

rgi_dicts = {'ngozumpa':'RGI60-15.03473',
             'khumbu': 'RGI60-15.03733',
             'changri_nup': 'RGI60-15.03734',
             'imja': 'RGI60-15.03743',
             'langtang': 'RGI60-15.04121',
             'lirung': 'RGI60-15.04045',
             'cn_validation':'RGI60-15.03417',
             'kaskawulsh':'RGI60-01.16201',
             'gando':'RGI60-13.19763'}

def fetch_farinotti_thickness(glac_name):
    rgi_id = rgi_dicts[glac_name]
    try:
        thick_fn = glob.glob(os.path.join(farinotti_thickness_dir,f'{rgi_id}_thickness.tif'))[0]
        return thick_fn
    except:
        print("thickness file not found") 

def fetch_rounce_debris_thickness(glac_name):
    rgi_id = rgi_dicts[glac_name].split('-')[1]
    try:
        debris_thick_fn = os.path.join(rounce_debris_thickness_dir,f'HMA_DTE_{rgi_id}_hdts_m.tif')
        if not os.path.exists(debris_thick_fn):
            print("debris thickness not found, will return extrap product")
            debris_thick_extrap_fn = glob.glob(os.path.join(rounce_debris_thickness_dir,f'HMA_DTE_{rgi_id}_hdts_m_extrap.tif'))[0]
            out = debris_thick_extrap_fn
        else:
            out = debris_thick_fn
        return out
    except:
        print("neither debris thickness nor extrap file not found")


def fetch_glac_shp(rgi_id):
    glac_shp = os.path.join(rgi_dir,f'{rgi_id}_shp.gpkg')
    if not os.path.exists(glac_shp):
        print("Shapefile not found, creating a new shapefile")
        #if rgi_id == 'RGI60-01.16201':
         #   gdf = gpd.read_file(rgi_region1_fn)
          #  mask = gdf['RGIId'] == rgi_id
           # gdf[mask].to_file(glac_shp,driver='GPKG')
        else:
            gdf = gpd.read_file(os.path.join(rgi_dir,'15_rgi60_SouthAsiaEast.shp'))
            mask = gdf['RGIId'] == rgi_id
            gdf[mask].to_file(glac_shp,driver='GPKG')
    return glac_shp

def raster_bound_geom(fn,out_crs=None):
    with rasterio.open(fn,'r') as src:
        bounds = src.bounds
        crs = src.crs
    geom = box(*bounds)
    raster_geom = gpd.GeoDataFrame({'id':[1],'geometry':geom},crs=crs)
    if out_crs:
        raster_geom = raster_geom.to_crs(out_crs)
    
    return raster_geom.geometry.values[0]

def fetch_millan_thickness(glac_name):
    rgi_id = rgi_dicts[glac_name]
    millan_thickness_fn = os.path.join(millan_thickness_dir,f'{rgi_id}_millan_thickness.tif')
    if not os.path.exists(millan_thickness_fn):
        print("thickness file does not exist, will create one")
        glac_shp_fn = fetch_glac_shp(rgi_id)
        glac_shp = gpd.read_file(glac_shp_fn).to_crs(hma_aea_crs).geometry.values[0]
        potential_thickness_val = sorted(glob.glob(os.path.join(millan_thickness_dir,'THICKNESS*.tif')))
        for thick in potential_thickness_val:
            raster_bound = raster_bound_geom(thick,hma_aea_crs)
            if raster_bound.contains(glac_shp):
                
                geospatial.clip_raster_by_shp_disk(thick,glac_shp_fn,extent='shp',out_fn=millan_thickness_fn)
    return millan_thickness_fn

def fetch_millan_thickness_error(glac_name):
    rgi_id = rgi_dicts[glac_name]
    millan_thickness_error_fn = os.path.join(millan_thickness_dir,f'{rgi_id}_millan_thickness_err.tif')
    if not os.path.exists(millan_thickness_error_fn):
        print("thickness error file does not exist, will create one")
        glac_shp_fn = fetch_glac_shp(rgi_id)
        glac_shp = gpd.read_file(glac_shp_fn).to_crs(hma_aea_crs).geometry.values[0]
        potential_thickness_err_val = sorted(glob.glob(os.path.join(millan_thickness_dir,'ERRTHICKNESS*.tif')))
        for thick_err in potential_thickness_err_val:
            raster_bound = raster_bound_geom(thick_err,hma_aea_crs)
            if raster_bound.contains(glac_shp):
                
                geospatial.clip_raster_by_shp_disk(thick_err,glac_shp_fn,extent='shp',out_fn=millan_thickness_error_fn)
    return millan_thickness_error_fn

def fetch_millan_velocity(glac_name):
    rgi_id = rgi_dicts[glac_name]
    outdir = os.path.join(millan_velocity_dir,'individual_glacier_fn')
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    millan_vx_fn = os.path.join(millan_velocity_dir,f'{rgi_id}_millan_vx.tif')
    millan_vy_fn = os.path.join(millan_velocity_dir,f'{rgi_id}_millan_vy.tif')
    millan_vm_fn = os.path.join(millan_velocity_dir,f'{rgi_id}_millan_vm.tif')

    if not os.path.exists(millan_vx_fn):
        print("velocity file does not exist, will create one")
        glac_shp_fn = fetch_glac_shp(rgi_id)
        glac_shp = gpd.read_file(glac_shp_fn).to_crs(hma_aea_crs).geometry.values[0]
        potential_vx_val = sorted(glob.glob(os.path.join(millan_velocity_dir,'VX*.tif')))
        for vx in potential_vx_val:
            raster_bound = raster_bound_geom(vx,hma_aea_crs)
            if raster_bound.contains(glac_shp):
                
                geospatial.clip_raster_by_shp_disk(vx,glac_shp_fn,extent='shp',out_fn=millan_vx_fn)
                vy = os.path.join(millan_velocity_dir,f"VY_RGI{os.path.basename(vx).split('_RGI')[1]}")
                vm = os.path.join(millan_velocity_dir,f"V_RGI{os.path.basename(vx).split('_RGI')[1]}")
                geospatial.clip_raster_by_shp_disk(vy,glac_shp_fn,extent='shp',out_fn=millan_vy_fn)
                geospatial.clip_raster_by_shp_disk(vm,glac_shp_fn,extent='shp',out_fn=millan_vm_fn)
    return millan_vx_fn, millan_vy_fn, millan_vm_fn

