#! /usr/bin/env python
import os,sys,glob,shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from pygeotools.lib import iolib,malib,warplib,geolib,timelib,filtlib
from imview import pltlib
from skimage.measure import label, regionprops
from skimage.morphology import binary_erosion, disk
from skimage.restoration import inpaint_biharmonic
import skimage.filters
import skimage.feature
import scipy.ndimage 
from velocity_proc import velocity_timeseries

def gauss_fltr_astropy_fft(dem, size=None, sigma=None, origmask=True, fill_interior=False):
    
    """
    ## Applying original mask=True is recommended to take care of bleeding along edges
    ***From pygeotools, commit to mainstream repo after full testing***
    # I use this over the direct Gaussian astropy filter  as it is way faster for large kernels
    Astropy gaussian filter properly handles convolution with NaN
    http://stackoverflow.com/questions/23832852/by-which-measures-should-i-set-the-size-of-my-gaussian-filter-in-matlab
    width1 = 3; sigma1 = (width1-1) / 6;
    Specify width for smallest feature of interest and determine sigma appropriately
    sigma is width of 1 std in pixels (not multiplier)
    scipy and astropy both use cutoff of 4*sigma on either side of kernel - 99.994%
    3*sigma on either side of kernel - 99.7%
    If sigma is specified, filter width will be a multiple of 8 times sigma 
    Alternatively, specify filter size, then compute sigma: sigma = (size - 1) / 8.
    If size is < the required width for 6-8 sigma, need to use different mode to create kernel
    mode 'oversample' and 'center' are essentially identical for sigma 1, but very different for sigma 0.3
    The sigma/size calculations below should work for non-integer sigma

    Parameters
    ----------
    dem: np.ma.array
        input array on which filter is to be applied
    size: int
        size of Gaussian filter kernel in px units (odd)
    sigma: int
        Sigma of Gaussian filter, see above for documentation
    origmask: bool
        Respect the original nodata mask after Gaussian filtering has been performed (default:False, but recommended to set True)
    fill_interior: bool
        If true, fill interior holes (default:False)
    Returns
    ----------
    out: np.ma.array
        Output array after application of Gaussian filter
    """

    #import astropy.nddata
    import astropy.convolution
    dem = malib.checkma(dem)
    #Generate 2D gaussian kernel for input sigma and size
    #Default size is 8*sigma in x and y directions
    #kernel = astropy.nddata.make_kernel([size, size], sigma, 'gaussian')
    #Size must be odd
    if size is not None:
        size = int(np.floor(size/2)*2 + 1)
        size = max(size, 3)
    #Truncate the filter at this many standard deviations. Default is 4.0
    truncate = 3.0
    if size is not None and sigma is None:
        sigma = (size - 1) / (2*truncate)
    elif size is None and sigma is not None:
        #Round up to nearest odd int
        size = int(np.ceil((sigma * (2*truncate) + 1)/2)*2 - 1)
    elif size is None and sigma is None:
        #Use default parameters
        sigma = 1
        size = int(np.ceil((sigma * (2*truncate) + 1)/2)*2 - 1)
    size = max(size, 3)
    kernel = astropy.convolution.Gaussian2DKernel(sigma, x_size=size, y_size=size, mode='oversample')

    print("Applying gaussian smoothing filter with size %i and sigma %0.3f (sum %0.3f)" % \
            (size, sigma, kernel.array.sum()))

    #This will fill holes
    #np.nan is float
    #dem_filt_gauss = astropy.nddata.convolve(dem.astype(float).filled(np.nan), kernel, boundary='fill', fill_value=np.nan)
    #dem_filt_gauss = astropy.convolution.convolve(dem.astype(float).filled(np.nan), kernel, boundary='fill', fill_value=np.nan)
    #Added normalization to ensure filtered values are not brightened/darkened if kernelsum != 1
    dem_filt_gauss = astropy.convolution.convolve_fft(dem.astype(float).filled(np.nan), kernel, boundary='fill', 
                                                      fill_value=np.nan, normalize_kernel=True,
                                                     allow_huge=True)
    #This will preserve original ndv pixels, applying original mask after filtering
    if origmask:
        print("Applying original mask")
        #Allow filling of interior holes, but use original outer edge
        if fill_interior:
            mask = malib.maskfill(dem)
        else:
            mask = np.ma.getmask(dem)
        dem_filt_gauss = np.ma.array(dem_filt_gauss, mask=mask, fill_value=dem.fill_value)
    out = np.ma.fix_invalid(dem_filt_gauss, copy=False, fill_value=dem.fill_value)
    out.set_fill_value(dem.fill_value.astype(dem.dtype))
    return out.astype(dem.dtype)

def compute_lagrangian(dem1,dem2,vx,vy,dt,annual=True,res=2,startofpath=True,return_shifted_dem=False):
    """
    Compute Lagrangian elevation change by taking into account the velocity grids
    
    Parameters
    ------------
    dem1: np.ma.array
        DEM at first timestamp
    dem2: np.ma.array
        DEM at second timestamp
    vx: np.ma.array
        velocity in x-direction in m/yr
    vy: np.ma.array
        velocity in y-direction in m/yr (positive up)
    dt: numeric
        time difference in years between the observation
    annual: bool
        If true, return annualized lagrangian elevation change
    res: numeric
        GSD of inputs
    startofpath: bool
        if True, products will be calculated at grid of timestamp 1, else timestamp 2 (endofpath)
    return_shifted_dem: bool
        if True, return the flow-corrected DEM along with the Lag Dh/Dt raster
    Returns
    ------------
    lag_dz: np.ma.array
        elevation change by following the particle in time
    dem2_orig_idx: np.ma.array
        if return_shifted_dem is True, Lagrangian flow corrected DEM2 is returned
    """
    #This returns tuples of x, y indices for all unmasked values
    #Note, these are pixel centers
    y_init, x_init = np.nonzero(~(np.ma.getmaskarray(dem1)))
    x = np.ma.array(x_init)
    y = np.ma.array(y_init)
    
    # find the total displacement
    dispx_t = malib.nanfill(vx, scipy.ndimage.map_coordinates, [y,x], order=1, mode='nearest') * dt
    dispy_t = malib.nanfill(-1*vy, scipy.ndimage.map_coordinates, [y,x], order=1, mode='nearest') * dt
    # m to px
    res = res
    if startofpath:
        x_di = x + (dispx_t/res)
        y_di = y + (dispy_t/res)
    

        ### sample dem2 at expected position during time1
        dem2_samp = malib.nanfill(dem2, scipy.ndimage.map_coordinates, [y_di,x_di], order=1, mode='nearest')
        dem2_orig_idx = np.zeros_like(dem2)
        dem2_orig_idx[y,x] = dem2_samp
        lag_dz = dem2_orig_idx - dem1
    else:
        x_di = x - (dispx_t/res)
        y_di = y - (dispy_t/res)
        
        ### sample dem1 at expected position during time2  
        dem1_samp = malib.nanfill(dem1, scipy.ndimage.map_coordinates, [y_di,x_di], order=1, mode='nearest')
        dem1_laterindex = np.zeros_like(dem1)
        dem1_laterindex[y,x] = dem1_samp
        lag_dz = dem2 - dem1_laterindex
     
    vm = np.ma.sqrt(dispx_t**2+dispy_t**2)
    mask = malib.common_mask([vx,vy,lag_dz])
    lag_dz = np.ma.array(lag_dz,mask=mask)
    if annual:
        lag_dz = lag_dz/dt
    if return_shifted_dem & startofpath:
        out = [lag_dz,dem2_orig_idx]
    else:
        out = lag_dz
    return out 

def compute_along_slope_flow_correction_working(dem1,vx,vy,dt,smooth=True,px_lengthscale=None,res=2,
        annual=True,lookup_indexes=None,origmask=True):
    """
    Compute expected along slope elevation change due to glacier flow
    Parameters
    ------------
    dem1: np.ma.array
        DEM at time 1
    vx: np.ma.array
        velocity in x-direction in m/yr
    vy: np.ma.array
        velocity in y-direction in m/yr (positive up)
    dt: numeric
        time difference in years between the observation
    smooth: bool
        smooth rough surface of DEM before computing along-slope dh changes
    px_lengthscale: list
        list of lengthcales over which to compute along slope flow in px (will use a Gaussian Kernel with that filter width)
        if more than one entry in list, will employ gaussian kernel based on input
    res: numeric
        grid resolution
    annual: bool
        If true, return annualized lagrangian elevation change
    lookup_indexes: list
        list of numpy arrays containing indices to place the smoothed result with a given lengthscale in the final product
    origmask: bool
        conserve original mask when applying gaussian filter
    
    Returns
    ------------
    along_slope_dh : np.ma.array
        expected elevation change due to flow along slope
    """
    
    #This returns tuples of x, y indices for all unmasked values
    #Note, these are pixel centers
    y_init, x_init = np.nonzero(~(np.ma.getmaskarray(dem1)))
    x = np.ma.array(x_init)
    y = np.ma.array(y_init)
    
    # find the total displacement
    dispx_t = malib.nanfill(vx, scipy.ndimage.map_coordinates, [y,x], order=1, mode='nearest') * dt
    dispy_t = malib.nanfill(-1*vy, scipy.ndimage.map_coordinates, [y,x], order=1, mode='nearest') * dt
    # m to px
    x_di = x + (dispx_t/res)
    y_di = y + (dispy_t/res)
    

    ### sample smooth_dem1 at t2
    # this implies that there has not been any elevation change due to surface melting or vertical glacier flow
    # the particle's elevation changed due to a change in slope
    dem1_samp = malib.nanfill(dem1, scipy.ndimage.map_coordinates, [y_di,x_di], order=1, mode='nearest')
    dem1_orig_idx = np.zeros_like(dem1)
    dem1_orig_idx[y,x] = dem1_samp
    along_slope_dh = dem1_orig_idx - dem1
    
    if smooth:
        # smooth downslope dhdt using a gausian filter of above width
        if len(px_lengthscale) == 1:
            print("Will use a constant lengthscale of {px_lengthscale[0]} px for gaussian smoothing")
            along_slope_dh = gauss_fltr_astropy_fft(along_slope_dh,size=px_lengthscale[0],origmask=origmask)
        else:
            along_slope_dh = adaptive_gaussian_smooth_alt(along_slope_dh,px_lengthscale,lookup_indexes,origmask=origmask)
    
    dem1 = None

     
    if annual:
        along_slope_dh = along_slope_dh/dt
    #limit mask by presence of velocity data
    along_slope_dh = np.ma.array(along_slope_dh,mask=np.ma.getmask(vx))
    return along_slope_dh

def compute_simple_flux_div(vx,vy,H,v_col_f=0.8,
    dx=2,dy=2,eo=1,smooth=True,px_lengthscale=None,lookup_indexes=None):
    """
    Compute simple flux divergence
    Parameters
    ------------
    vx: np.ma.array
        surface velocity in E-W direction
    vy: np.ma.array
        surface velocity in N-S direction
    H: np.ma.array
        ice thickness raster
    px_lengtscale:
        list of lengthcales over which to compute along slope flow in px (will use a Gaussian Kernel with that filter width)
        if more than one entry in list, will employ gaussian kernel based on input
    lookup_indexes: list
        list of numpy arrays containing indices to place the smoothed result with a given lengthscale in the final product
    v_col_f: float
        factor to scale surface velocity to column-averaged velocity
    dx: numeric
        x-resolution of grid in m
    dy: numeric
        y-resolution of grid in m
    eo: numeric
        edge order for gradient computation
    smooth: bool
        Smooth the flux divergence map
    Returns
    ------------
    divQ2: np.ma.array
        flux divergence (areas of emergence have negative values in this raster)
    """
    Q = H * v_col_f * np.ma.array([vx,-vy])
    divQ2 = np.gradient(Q[0],dx,axis=1,edge_order=eo) + np.gradient(Q[1],dy,axis=0,edge_order=eo)
    if smooth:
        divQ2 = adaptive_gaussian_smooth_alt(divQ2,px_lengthscale,lookup_indexes)
    return divQ2


def adaptive_gaussian_smooth_alt(ma,px_lengthscale,lookup_indexes,final_gauss_filter=False,return_stack=False,origmask=True):
    """
    Apply adaptive gaussian smoothing using an adaptive strategy (this is the latest)
    Parameters
    ------------
    ma: np.ma.array
        array to smoothen
    px_lengtscale:
        list of lengthcales over which to compute along slope flow in px (will use a Gaussian Kernel with that filter width)
        if more than one entry in list, will employ gaussian kernel based on input
    lookup_indexes: list
        list of numpy arrays containing indices to place the smoothed result with a given lengthscale in the final product
    final_gauss_filter: bool
        apply an additional filtering to smooth the edges (not used, will be phased out)
    return_stack: bool
        return the entire stack rather than composite array (not used, will be phased out)
    origmask: bool
        use the original mask when applying the Gaussian Filter
    Returns
    ------------
    smooth_ma: np.ma.array
        smoothed result at given lengthscales
    """

    # https://gis.stackexchange.com/questions/229349/buffering-pixels-in-an-array-python
    from scipy.ndimage import maximum_filter
    #maximum_filter(index,size=2*buffer_size+1,mode='constant',cval=0)
    
    n = len(px_lengthscale)
    #smooth_ma = np.zeros((ma.shape[0],ma.shape[1],n))
    smooth_ma = np.ma.dstack([np.zeros_like(ma)]*n)
    for idx,px_length in enumerate(px_lengthscale):
        # compute buffer based on expected px_lengthscale around the patch of pixels
        if px_length < 3:
            buff_dis = 3
        elif (int(px_length) % 2) != 0:
            buff_dis = int(px_length) + 1
        else:
            buff_dis = int(px_length)
        buffer_index = maximum_filter(lookup_indexes[idx],size=px_length,mode='constant',cval=0).astype(bool)
        # sample the array at the buffered locations
        
        ma_filter = np.ma.array(ma,mask=~buffer_index)

        # run gaussian filter         
        temp_gauss = gauss_fltr_astropy_fft(ma_filter,size=px_length,origmask=origmask)

        # sample pixels after gaussian filter into output array
        smooth_ma[:,:,idx][lookup_indexes[idx]] = temp_gauss[lookup_indexes[idx]]

    smooth_ma = np.ma.masked_equal(smooth_ma,0)

    if return_stack is not True:
        smooth_ma = np.ma.mean(smooth_ma,axis=2)
    # smoothen the transition between the composite bands using a final filter to remove abruptness
    if final_gauss_filter:
        smooth_ma = filtlib.gauss_fltr_astropy(smooth_ma,size=9)
    return smooth_ma

# **** Lookup table calculation ****

def thickness_division(icethick,n,factor,res=2,verbose=False):
    """
    Divide thickness distribution using equal interval classification and return indices for each division and corresponding lengthscales
    Parameters
    ------------
    icethick: np.ma.array
        ice thickness raster
    n: int
        number of bins to divide the ice thickness distribution into
    factor: list
        list of integers to be used as factor which when multiplied with the ice thickness provides length of the filter
        I use now only a single value, passed as [4]
    res: numeric
        GSD or resolution of raster products in m
    verbose: bool
        whether to print logs or not
    Returns
    ------------
    indices_table: list
        list of array containing indices (positions) for the corresponding thickness measurment
    px_lengthscale: list
        list of lengtscales which will be used for performing filtering
    """
    # iniate empty arrays
    thickness_med = []
    indexes_table = []
    # divide ice thickness using a simple equal interval classification
    # TODO: use a division in which divison contains equal number of pixels
    # Some ideas here: https://pro.arcgis.com/en/pro-app/latest/help/mapping/layer-properties/data-classification-methods.htm
    equal_inerval_class = np.ptp(icethick.compressed())/n

    # plot of the histogram distribution
    f,ax = plt.subplots()
    ax.hist(icethick.compressed(),bins=30)
    ax.set_xlabel("Ice thickness (m)")
    ax.set_ylabel("# of pixels")
    
    # interval bounds   
    intervals = np.arange(icethick.min(),icethick.max(),equal_inerval_class).tolist() + [icethick.max()]

    for idx,lowerlim in enumerate(intervals) :
        ax.axvline(x=lowerlim,c='k',label=f'{np.round(lowerlim,2)} m')
        if idx == n:
            index = icethick > lowerlim
        else:
            index = (icethick < intervals[idx+1]) & (icethick > lowerlim)
        if verbose:
            print(np.ma.median(icethick[index]))
        # for each thickness bound, we compute median of all thickness values in that bound
        # and also store the spatial index where these pixels are in the array
        thickness_med.append(np.ma.median(icethick[index]))
        indexes_table.append(index)
    if n < 10:
        ax.legend()
    thickness_med = thickness_med[:-1]
    if len(factor) == 1:
        factor = factor*n
    # we now compute lengtscales, by multiplying the median ice thickness of each bound with the input factor 
    lengthscales = [thickness_med[i]*factor[i] for i in range(n)] # this is in m units
    px_lengthscales = np.round(lengthscales)/res # this is in pixel units
    indexes_table = indexes_table[:-1]
    print(f"Lengthscales in m: {lengthscales}")
    print(f"Lengthscales in pixels: {px_lengthscales}")
    return indexes_table,px_lengthscales

################ Cliff detection using scharr edges on melt and high slopes ########################
def create_masked_boolean_array(ma,threshold,gte=True):
    """
    Helper function to map regions of interest while still preserving the masked array
    Parameters
    ------------
    ma:np.ma.array
        input array to be masked
    threshold: numeric
        cutoff to be applied on the input array
    gte: bool
        if true, the function will be used to select areas greater than or equal to threshold
        else, it will select areas which smaller than or equal to the threshold 
    Returns
    -------------
    ma_bool: np.ma.array
        output array with pixels matching the threshold condition assigned to 1, other pixels assigned 0
    """
    if isinstance(ma, np.ma.MaskedArray):
        origmask = ma.mask
        mask_input = True
    else:
        mask_input = False
    if gte:
        pot_idx = ma>=threshold
        non_pot_idx = ma<threshold
    else:
        pot_idx = ma<=threshold
        non_pot_idx = ma>threshold
    ma_bool = np.ma.empty_like(ma,dtype=np.int16)
    ma_bool[pot_idx] = 1
    ma_bool[non_pot_idx] = 0
    if mask_input:
        ma_bool = np.ma.array(ma_bool,mask=origmask)
    return ma_bool
def high_slope_high_melt(slope1,slope2,dhdt,ds,slope_cutoff=10,dhdt_cutoff=-2.5,debris_cover_area=None,
    min_pixel_count=25):
    """
    """
    steep1 = create_masked_boolean_array(slope1,slope_cutoff)
    steep2 = create_masked_boolean_array(slope2,slope_cutoff) 
    steep_union_mask = np.ma.clip(steep1 + steep2,0,1)
    high_melt_mask = create_masked_boolean_array(dhdt,dhdt_cutoff,gte=False)
    ice_cliff_mask_local = steep_union_mask*high_melt_mask

    if debris_cover_area is not None:
        ice_cliff_mask_local = np.ma.array(ice_cliff_mask_local,mask=debris_cover_area.mask)
    
    gdf = binary2shapefile(ice_cliff_mask_local,val=1,ds=ds)
    res = geolib.get_res(ds,square=True)[0]
    min_area = min_pixel_count * res * res
    big_cliffs = gdf.area > min_area
    
    return gdf[big_cliffs]

def gdal2rasterio_transform(gt_fn):
    from rasterio import Affine
    return Affine(gt_fn[1],gt_fn[2],gt_fn[0],gt_fn[4],gt_fn[5],gt_fn[3])

def binary2shapefile(binary_array,val,ds,return_gdf=True,outfn=None):
    from shapely.geometry import shape
    from rasterio import features
    #binary_array = np.ones((1,),dtype=np.int16)[0]*binary_array
    mypoly=[]
    zone = int(ds.GetProjection().split('AUTHORITY["EPSG",')[-1].split(']]')[0].replace('"',''))
    proj = f'EPSG:{zone}'
    for vec in features.shapes(binary_array,transform=gdal2rasterio_transform(
        ds.GetGeoTransform())):
        if vec[1] == val:
            mypoly.append(shape(vec[0]))
    gdf = gpd.GeoDataFrame(geometry=mypoly,crs=proj)
    if outfn is None:
        outfn = 'hotspot.gpkg'
    gdf.to_file(outfn,driver='GPKG')
    if return_gdf:
        return gdf
    

####### Inpainting code from Ross Beyer ###############################
# This needs to be checked about licensing/sharing
## From Ross Beyer

# Copyright 2022, Ross A. Beyer (rbeyer@rossbeyer.net)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

def inpaint_fill(dem, filllimit: float, filldem=None):
    """
    dem: np.ma.array
    filllimit: float
    If given, nodata blobs are characterized by the length of the "
             "minor axis of an ellipse that has the same normalized second "
             "central moments as the blob (see skimage.measure.regionprops). "
             "Blobs with a minor axis larger than --limit will not be filled. "
             "If --filldem is also given, then that dem will be used to "
             "'fill in' any nodata blobs that meet the --limit criteria.  "
             "Pixels farther than --limit from the edge of nodata blob will "
             "be identical to the --filldem  pixels, in the zone between "
             "them, fitting is performed.
    filldem: np.ma.array
    """
    
    import logging

    if filldem is not None and dem.shape != filldem.shape:
        raise ValueError("The dem and the filldem do not have the same shape.")

    # Find nodata blobs and get their properties
    nodata_blobs = label(dem.mask)
    nodata_regions = regionprops(nodata_blobs)

    if len(nodata_regions) == 0:
        raise ValueError("There are no nodata regions in dem.")

    # Make copy to write changes to
    tempdem = dem.copy()

    for props in nodata_regions:
        logging.info(
            f"{props.label}/{len(nodata_regions)}, area: {props.area}"
        )
        if props.axis_minor_length > filllimit:
            # print(props.axis_minor_length)
            logging.info(f"{props.label} filling with filldem")
            if filldem is not None:
                bbox_slice = (
                    slice(props.bbox[0], props.bbox[2]),
                    slice(props.bbox[1], props.bbox[3])
                )
                # cropfdem = filldem[bbox_slice]
                # blobmask = np.full_like(cropfdem, False)
                # for c in props.coords:
                #     blobmask[c[0] - props.bbox[0], c[1] - props.bbox[1]] = True

                # This step can be slow for large areas.
                eroded_mask = binary_erosion(
                    # blobmask, disk(filllimit, dtype=bool)
                    props.image, disk(filllimit, dtype=bool)
                )
                tempdem[bbox_slice][eroded_mask] = filldem[bbox_slice][eroded_mask]
            else:
                # The blob is larger than we want to just spline fill,
                # so remove it from the mask.
                logging.info(f"Skipping {props.label}")
                for coord in props.coords:
                    tempdem.mask[coord[0], coord[1]] = False
                continue
        # else the nodata blob is small enough to inpaint, and should be left
        # in the mask as-is.

    outdem = inpaint_biharmonic(tempdem.data, tempdem.mask)
    fill_value = dem.fill_value
    outdem = np.ma.masked_equal(outdem,fill_value)

    return outdem


######### plotting functions #######
def add_quiver_contour(ax, vx, vy, stride=5, color='dodgerblue',scale=500,levels=None,
    quiver=True,contour=True):
    """ 
    Add quiver plots with vector data
    Useful for velocity maps or to show components of velocity divergence
    from David Shean's gmbtools repository
    """
    if quiver:
        X = np.arange(0,vx.shape[1],stride)
        Y = np.arange(0,vx.shape[0],stride)
        linewidths = np.linspace(0,6)
        ax.quiver(X, Y, vx[::stride,::stride], vy[::stride,::stride], 
            color=color, pivot='mid',scale=scale,linewidths=linewidths)
    # add contour
    if contour:
        if not levels:
            print("Computing integer levels for contour, provide if intended to use otherwise")
            clim = malib.calcperc(np.ma.sqrt(vx**2+vy**2),(0.001,0.999))
            levels = np.arange(clim[0],clim[1],dtype=int).tolist()
        ax.contour(np.ma.sqrt(vx**2+vy**2),colors='k',linewidths=0.35,
            levels=levels)
        
def hist_plot_gmbtools(hotspot_dh,background_dh,clean_ice_dh,smb_dh,debris_thick,vm,z1,ds,bin_width=50):
    # digitise and other tricks picked from gmbtools repository by David
    res = geolib.get_res(ds,square=True)
    z_bin_edges, z_bin_centers = malib.get_bins(z1, bin_width)
    z1_bin_counts, z1_bin_edges = np.histogram(z1.compressed(), bins=z_bin_edges)
    z1_bin_areas = z1_bin_counts * res[0] * res[1] / 1E6
    
    #Create arrays to store output
    
    mb_bin_med_background = np.ma.masked_all_like(z1_bin_areas)
    np.ma.set_fill_value(mb_bin_med_background, np.nan)
    mb_bin_nmad_background = np.ma.masked_all_like(mb_bin_med_background)
    mb_bin_q1_background = np.ma.masked_all_like(mb_bin_med_background)
    mb_bin_q3_background = np.ma.masked_all_like(mb_bin_med_background)
    mb_bin_count_background = np.ma.masked_all_like(mb_bin_med_background)

    mb_bin_med_total = np.ma.masked_all_like(mb_bin_med_background)
    mb_bin_nmad_total = np.ma.masked_all_like(mb_bin_med_background)
    mb_bin_q1_total = np.ma.masked_all_like(mb_bin_med_background)
    mb_bin_q3_total = np.ma.masked_all_like(mb_bin_med_background)
    mb_bin_count_total = np.ma.masked_all_like(mb_bin_med_background)

    mb_bin_med_clean = np.ma.masked_all_like(mb_bin_med_background)
    mb_bin_nmad_clean = np.ma.masked_all_like(mb_bin_med_background)
    mb_bin_count_clean = np.ma.masked_all_like(mb_bin_med_background)
    mb_bin_q1_clean = np.ma.masked_all_like(mb_bin_med_background)
    mb_bin_q3_clean = np.ma.masked_all_like(mb_bin_med_background)

    
    mb_bin_med_hotspot = np.ma.masked_all_like(mb_bin_med_background)
    mb_bin_nmad_hotspot = np.ma.masked_all_like(mb_bin_med_background)
    mb_bin_count_hotspot = np.ma.masked_all_like(mb_bin_med_background)
    mb_bin_q1_hotspot = np.ma.masked_all_like(mb_bin_med_background)
    mb_bin_q3_hotspot = np.ma.masked_all_like(mb_bin_med_background)
    
    debthick_bin_med = np.ma.masked_all_like(mb_bin_med_background)
    debthick_bin_nmad = np.ma.masked_all_like(mb_bin_med_background)
    debthick_bin_q1 = np.ma.masked_all_like(mb_bin_med_background)
    debthick_bin_q3 = np.ma.masked_all_like(mb_bin_med_background)

    vm_bin_med = np.ma.masked_all_like(mb_bin_med_background)
    vm_bin_nmad = np.ma.masked_all_like(mb_bin_med_background)
    vm_bin_q1 = np.ma.masked_all_like(mb_bin_med_background)
    vm_bin_q3 = np.ma.masked_all_like(mb_bin_med_background)
    
    idx = np.digitize(z1,z_bin_edges)
    for bin_n in range(z_bin_centers.size):
        #debris background_sample
        dh_background_samp = background_dh[(idx == bin_n+1)]
        mb_bin_med_background[bin_n] = np.round(malib.fast_median(dh_background_samp),2)
        mb_bin_nmad_background[bin_n] = np.round(malib.mad(dh_background_samp),2)
        mb_bin_count_background[bin_n] = np.ma.count(dh_background_samp)
        q1,q3 = np.nanpercentile(dh_background_samp.filled(np.nan),(25,75))
        mb_bin_q1_background[bin_n] = q1
        mb_bin_q3_background[bin_n] = q3

        #debris hotspot_sample
        dh_hotspot_samp = hotspot_dh[(idx == bin_n+1)]
        mb_bin_med_hotspot[bin_n] = np.round(malib.fast_median(dh_hotspot_samp),2)
        mb_bin_nmad_hotspot[bin_n] = np.round(malib.mad(dh_hotspot_samp),2)
        mb_bin_count_hotspot[bin_n] = np.ma.count(dh_hotspot_samp)
        q1,q3 = np.nanpercentile(dh_hotspot_samp.filled(np.nan),(25,75))
        mb_bin_q1_hotspot[bin_n] = q1
        mb_bin_q3_hotspot[bin_n] = q3

        # clean ice sample
        dh_clean_samp = clean_ice_dh[(idx == bin_n+1)]
        mb_bin_med_clean[bin_n] = np.round(malib.fast_median(dh_clean_samp),2)
        mb_bin_nmad_clean[bin_n] = np.round(malib.mad(dh_clean_samp),2)
        mb_bin_count_clean[bin_n] = np.ma.count(dh_clean_samp)
        q1,q3 = np.nanpercentile(dh_clean_samp.filled(np.nan),(25,75))
        mb_bin_q1_clean[bin_n] = q1
        mb_bin_q3_clean[bin_n] = q3
        
        # total sample
        dh_total_samp = smb_dh[(idx == bin_n+1)]
        mb_bin_med_total[bin_n] = np.round(malib.fast_median(dh_total_samp),2)
        mb_bin_nmad_total[bin_n] = np.round(malib.mad(dh_total_samp),2)
        mb_bin_count_total[bin_n] = np.ma.count(dh_total_samp)
        q1,q3 = np.nanpercentile(dh_total_samp.filled(np.nan),(25,75))
        mb_bin_q1_total[bin_n] = q1
        mb_bin_q3_total[bin_n] = q3

        # debris thickness
        debthick_samp = debris_thick[(idx == bin_n+1)]
        debthick_bin_med[bin_n] = np.round(malib.fast_median(debthick_samp),2)
        debthick_bin_nmad[bin_n] = np.round(malib.mad(debthick_samp),2)
        q1,q3 = np.nanpercentile(debthick_samp.filled(np.nan),(25,75))
        debthick_bin_q1[bin_n] = q1
        debthick_bin_q3[bin_n] = q3 

        # velocity magnitude vm
        vm_samp = vm[(idx == bin_n+1)]
        vm_bin_med[bin_n] = np.round(malib.fast_median(vm_samp),2)
        vm_bin_nmad[bin_n] = np.round(malib.mad(vm_samp),2)
        q1,q3 = np.nanpercentile(vm_samp.filled(np.nan),(25,75))
        vm_bin_q1[bin_n] = q1
        vm_bin_q3[bin_n] = q3
        
        
        
        
    #area cacluclation
    mb_bin_area_clean = mb_bin_count_clean * res[0] * res[1] * 1e-6
    mb_bin_area_total = mb_bin_count_total * res[0] * res[1] * 1e-6
    mb_bin_area_background = mb_bin_count_background * res[0] * res[1] * 1e-6
    mb_bin_area_hotspot = mb_bin_count_hotspot * res[0] * res[1] * 1e-6

    stats_df = pd.DataFrame({'med_bg_dhdt': mb_bin_med_background, 'nmad_bg_dhdt':mb_bin_nmad_background,
        'q1_bg_dhdt':mb_bin_q1_background,'q3_bg_dhdt':mb_bin_q3_background,
        'area_bg':mb_bin_area_background,

        'med_hotspot_dhdt':mb_bin_med_hotspot, 'nmad_hotspot_dhdt':mb_bin_nmad_hotspot,
        'q1_hotspot_dhdt':mb_bin_q1_hotspot, 'q3_hotspot_dhdt':mb_bin_q3_hotspot,
        'area_hotspot':mb_bin_area_hotspot,

        'med_clean_dhdt':mb_bin_med_clean, 'nmad_clean_dhdt':mb_bin_nmad_clean,
        'q1_clean_dhdt':mb_bin_q1_clean, 'q3_clean_dhdt':mb_bin_q3_clean,
        'area_clean':mb_bin_area_clean,

        'med_total_dhdt':mb_bin_med_total, 'nmad_total_dhdt':mb_bin_nmad_total,
        'q1_total_dhdt':mb_bin_q1_total,'q3_total_dhdt':mb_bin_q3_total,
        'area_total':mb_bin_area_total,

        'med_deb_thick':debthick_bin_med,'nmad_deb_thick':debthick_bin_nmad,
        'q1_deb_thick':debthick_bin_q1,'q3_deb_thick':debthick_bin_q3,

        'med_vm':vm_bin_med,'nmad_vm':vm_bin_nmad,
        'q1_vm':vm_bin_q1,'q3_vm':vm_bin_q3,

        'z_area':z1_bin_areas,
        'z_bin_centers':z_bin_centers})
    
    return stats_df

def plot_meltcurves(stats_df,title,outfn=None,hardcode_lim=None):
    
    if outfn is None:
        outfn = f"{title}_meltcurve.png"
    f,ax = plt.subplots(2,1,sharex=True)
    #secy = ax.twinx()
    ax[1].bar(stats_df.z_bin_centers,stats_df.area_bg,width=100,color='gray',label='backround')
    ax[1].bar(stats_df.z_bin_centers,stats_df.area_hotspot,width=100,color='pink',label='hotspot')
    ax[1].set_ylabel('Area ($km^2$)')
    ax[1].set_xlabel('Elevation (m WGS84)')
    ax[0].plot(stats_df.z_bin_centers,stats_df.med_bg_dhdt,c='gray',label='background dhdt')
    ax[0].fill_between(stats_df.z_bin_centers,stats_df.med_bg_dhdt - stats_df.nmad_bg_dhdt,
        stats_df.med_bg_dhdt + stats_df.nmad_bg_dhdt, color='gray',alpha=0.5)
    ax[0].plot(stats_df.z_bin_centers, stats_df.med_hotspot_dhdt,c='pink',label='hotspot dhdt')
    ax[0].fill_between(stats_df.z_bin_centers, stats_df.med_hotspot_dhdt - stats_df.nmad_hotspot_dhdt,
        stats_df.med_hotspot_dhdt + stats_df.nmad_hotspot_dhdt, color='pink',alpha=0.5)
    ax[0].set_ylabel('dhdt (m/yr)')
    ax[0].axhline(y=0,c='teal')
    if hardcode_lim is not None:
        lower_lim,upper_lim = hardcode_lim
    else:
        lower_lim = np.min((stats_df.med_bg_dhdt.values-stats_df.nmad_bg_dhdt.values).tolist() +
            (stats_df.med_hotspot_dhdt.values-stats_df.nmad_hotspot_dhdt.values).tolist())
        upper_lim = np.max((stats_df.med_bg_dhdt.values+stats_df.nmad_bg_dhdt.values).tolist() +
            (stats_df.med_hotspot_dhdt.values+stats_df.nmad_hotspot_dhdt.values).tolist())

    ax[0].set_ylim(lower_lim,upper_lim)
    ax[1].set_xlim(stats_df.z_bin_centers.values[0],stats_df.z_bin_centers.values[-1])
    for axa in ax.ravel():
        axa.legend()
    ax[0].set_title(title)
    f.savefig(outfn,dpi=300,bbox_inches='tight',pad_inches=0.1)

################## Save figure #######################
def prepare_lag_smb_figure(eul_dhdt,lag_dhdt, downslope_dhdt, smb_dhdt,ds,ax,clim_perc = (5,95)):
    """
    ## TODO: Add binned plots
    Prepare four panel dhdt plot showing elevation change after each step of correction
    eul_dhdt: np.ma.array
        eulerian dhdt map
    lag_dhdt: np.ma.array
        lagrangian dhdt map
    downslope_dhdt: np.ma.array
        dhdt due to downslope movement
    smb_dhdt: np.ma.array
        residual dhdt due to SMB
    ax: list
        list of 4 axes
    """
    clim = malib.calcperc_sym(eul_dhdt,clim_perc)
    pltlib.iv(eul_dhdt,ax=ax[0],ds=ds,hillshade=True,scalebar=True,cmap='RdBu',
              clim=clim,title='Eulerian dhdt',cbar=False)
    pltlib.iv(lag_dhdt,ax=ax[1],ds=ds,hillshade=True,scalebar=False,cmap='RdBu',
              clim=clim,title='Lagrangian dhdt',cbar=False)
    pltlib.iv(lag_dhdt - downslope_dhdt,ax=ax[2],ds=ds,hillshade=True,scalebar=False,
              cmap='RdBu',clim=clim,title='dhdt due to \ndownslope flow',cbar=False)
    pltlib.iv(smb_dhdt,ax=ax[3],ds=ds,hillshade=True,scalebar=False,cmap='RdBu',
              clim=clim,title='residual dhdt \n due to SMB',label='Elevation change (m/yr)')
    plt.tight_layout()


#################### Full Lag SMB workflow function ###########################
def lag_smb_workflow(dem1_fn,dem2_fn,vx_fn,vy_fn,H_fn,deb_thick_fn,glac_shp,glac_identifier,lengthscale_factor=4,
                     num_thickness_division=5,smr_cutoff=135,timescale='year',icecliff_gpkg=None,writeout=True,saveplot=True,outdir=None,
                     conserve_mass=True):
    """
    Workflow to compute residual elevation change due to surface melting following the continuity equation
    Parameters
    -------------
    dem1_fn: str
        path to DEM 1
    dem2_fn: str
        path to DEM 2
    vx_fn: str
        path to filtered E-W velocity file (m/yr or m/day)
    vy_fn: str
        path to filtered N-S velocity file (m/yr or m/day)
    H_fn: str
        path to Farinotti ice-thickness grid
    glac_identifier: str
        name of glacier or RGI-Id
    lengthscale_factor: int
        Factor to multiply ice thickness values with for computing lengthscale
    num_thickness_division: int
        number of bins to divide the thickness values in (used for adaptive thickness gaussian filtering)
    writeout: bool
        Writeout computed maps if True
    saveplot: bool
        Writeout compute figures if True
    outdir: str
        path to output directory where to store results
        defaults to {DEM_timestamps}_lag_smb_results
    timescale: str
        the timescale for "rates" measurements (year or day) 
    Returns
    -------------
    divQ2: np.ma.array
        flux divergence map (at resolution of finer DEM)
    eul_dhdt: np.ma.array
        eulerian elevation change map
    lag_dhdt: np.ma.array
        lagrangian elevation change map
    downslope_dhdt: np.ma.array
        downslope dhdt map
    smb_dhdt: np.ma.array
        residual dhdt due to SMB
    stats_df: pd.DataFrame
        altitudnal melt statistics
    """
    # prepare prefix and outdir for file saving operations
    t1 = str(timelib.fn_getdatetime(dem1_fn)).split(" ")[0]
    t2 = str(timelib.fn_getdatetime(dem2_fn)).split(" ")[0]
    prefix = f"{glac_identifier}_{t1}_to_{t2}"
    if outdir is None:
        outdir = f"{prefix}_lag_smb_products"
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    if timescale == 'year':
        dt = timelib.get_t_factor_fn(dem1_fn,dem2_fn)
    else:
        t1 = timelib.fn_getdatetime(dem1_fn)
        t2 = timelib.fn_getdatetime(dem2_fn)
        dt = (t2 - t1).days
    print(f"dt is {dt}")
    ### Step 1: Compute flux-divergence using appropriate lengthscales
    print ("******** Step 1: computing flux divergence *************")
    ## downsample the velocity grid to resolution of ice thickness map, bring datasets to common extent
    # we use average, as the velocity products are downsampled, so finer pixels will be averaged to the coarser grid
    ds_list = warplib.memwarp_multi_fn([H_fn,vx_fn,vy_fn],r='average')
    # load as masked array
    H,vx,vy = [iolib.ds_getma(ds) for ds in ds_list]
    dem1 = iolib.ds_getma(warplib.memwarp_multi(ds_list+[iolib.fn_getds(dem1_fn)],extent='first',res='first',r='average')[-1])
    # save resolution
    H_res = geolib.get_res(ds_list[0])[0]
    # here the thickness will change to cm, velocity will change to cm/day
    # compute simple flux divergence
    """
    simple_gradH_V, simple_H_divV, simple_divQ2 = compute_flux_div(
        vx,vy,H,dx=H_res,dy=H_res,return_intermediate_products=True)
    # compute lengthscales and corresponding spatial indices
    # having high number of divisions produces better results
    #hardcoding this to 20
    num_thic_div_divq2 = 20
    px_indices,lengtscales = thickness_division(H,n=num_thic_div_divq2,factor=[lengthscale_factor],
                                                         res=H_res)
    # compute flux divergence after Gaussian filtering grad
    smooth_gradH_V, smooth_H_divV, smooth_divQ2 = compute_flux_div_adaptive_smooth_gradients(
    vx,vy,H,lengtscales,px_indices,dx=H_res,dy=H_res,return_intermediate_products=True)
    # smooth the flux divergence to get rid of residual anamolies (1 times ice thickness)
    mask_mass_con = malib.common_mask([simple_divQ2,smooth_divQ2])
    smooth_divQ2 = adaptive_gaussian_smooth_alt(np.ma.array(smooth_divQ2,
                            mask=mask_mass_con),lengtscales/[lengthscale_factor],px_indices)
    """
    simple_divQ2 = compute_simple_flux_div(vx,vy,H,v_col_f=0.8,
        dx=H_res,dy=H_res,eo=1,smooth=False)
    # having high number of divisions produces better results
    #hardcoding this to 20
    num_thic_div_divq2 = num_thickness_division
    px_indices,lengtscales = thickness_division(H,n=num_thic_div_divq2,factor=[lengthscale_factor],
                                                         res=H_res)

    simple_divQ2_smooth = compute_simple_flux_div(vx,vy,H,v_col_f=0.8,
        dx=H_res,dy=H_res,eo=1,smooth=True,px_lengthscale=lengtscales,lookup_indexes=px_indices)
    # compute factor to conserve mass after operation of Gaussian Filter
    if conserve_mass:
        mask_mass_con = malib.common_mask([simple_divQ2,simple_divQ2_smooth])
        #factor_masscon = np.ma.sum(simple_divQ2)/np.ma.sum(smooth_divQ2)
        factor_masscon = (np.ma.mean(np.ma.array(simple_divQ2,mask=mask_mass_con))/
                            np.ma.mean(np.ma.array(simple_divQ2_smooth,mask=mask_mass_con)))
    else:
        factor_masscon = 1
    print(f"Multiplier for conserving mass in flux divergence equation is {factor_masscon}")
    smooth_divQ2_masscon = simple_divQ2_smooth*factor_masscon

    # writeout flux divergence map
    fluxdiv_outfn = os.path.join(outdir,f'{prefix}_divQ2.tif')
    iolib.writeGTiff(smooth_divQ2_masscon,fluxdiv_outfn,src_ds=ds_list[0])

    print("******** Step 2: Computing Elevation change due to downslope movement *****************")


    
    # here again change the compute_along_slope_flow_correction function

    dem1_glac = velocity_timeseries.mask_by_shp(glac_shp.geometry,dem1,ds_list[0])
    downslope_dhdt_smooth = compute_along_slope_flow_correction_working(dem1_glac,vx,vy,dt,smooth=True,px_lengthscale=lengtscales,
                                                              res=H_res,annual=True,lookup_indexes=px_indices)

    downslope_dhdt_raw = compute_along_slope_flow_correction_working(dem1,vx,vy,dt,smooth=False,px_lengthscale=lengtscales,
                                                              res=H_res,annual=True,lookup_indexes=px_indices)

    mask_mass_con = malib.common_mask([downslope_dhdt_raw,downslope_dhdt_smooth])
    factor_masscon = (np.ma.mean(np.ma.array(downslope_dhdt_raw,mask=mask_mass_con))/
                        np.ma.mean(np.ma.array(downslope_dhdt_smooth,mask=mask_mass_con)))
    factor_masscon = 1
    print(f"Multiplier for conserving mass in expected slope parallel equation is {factor_masscon}")
    downslope_dhdt_mascon = downslope_dhdt_smooth * factor_masscon
    downslope_outfn = os.path.join(outdir,f'{prefix}_downslope_dhdt.tif')
    iolib.writeGTiff(downslope_dhdt_mascon,downslope_outfn,src_ds=ds_list[0])
    
    print("******** Step 3: Computing Lagrangian Elevation change *****************")
    # warp high-resolution images to common grid using cubic resampling
    ds_list_highres = warplib.memwarp_multi_fn([dem1_fn,dem2_fn,vx_fn,vy_fn],extent='first',r='cubicspline')
    #here dt is in yr terms, we will need to convert it to days
    
    dem_res = geolib.get_res(ds_list_highres[0])[0]
    # read as masked arrays
    #vx, vy will change to cm/day
    dem1,dem2,vx,vy = [iolib.ds_getma(ds) for ds in ds_list_highres]
    # warp 50 m resolution data (ice-thickness and flux divergence) and read as masked arrays
    #warplib.memwarp_multi(ds_list_highres+[iolib.fn_getds(fn) for fn in [H_fn,'divQ2_smooth_ver1.tif']],res='first',r='cubicspline')[-2:]
    #use cubicspline as these 2 products are lower res
    H,divQ2,debris_ma,downslope_dhdt = [iolib.ds_getma(ds) for ds in warplib.memwarp_multi(ds_list_highres+[iolib.fn_getds(fn) for fn in [H_fn,fluxdiv_outfn,deb_thick_fn,downslope_outfn]],
                                                                                   res='first',r='cubic',extent='first')[-4:]]

    # Eulerian elevation change
    eul_dhdt = (dem2-dem1)/dt
    # Lagrangian elevation change (start of path)
    #compute_lagrangian function will change to adapt for dt terms ?
    lag_dhdt,dem2_flow_corrected = compute_lagrangian(dem1,dem2,vx,vy,dt,return_shifted_dem=True)

    
    
    

    print("******** Step 4: Computing Elevation change due to surface melting: Continuity equation *****************")
    smb_dhdt = lag_dhdt - downslope_dhdt + divQ2

   
        
    if timescale == 'year':
        ## locate cliffs and partition melt from background
        print("***** Analysis1: Computing SMR and locating hotspots*************")
        # cutoff for locate_cliffs_and_ponds_via_smr will change to accomodate cm/day
        """
        if timescale == 'day':
            smr_cutoff = smr_cutoff * 100
        smr,cliff_mask = locate_cliffs_and_ponds_via_smr(lag_dhdt-downslope_dhdt,smr_cutoff=smr_cutoff)
        """
        # now use scharr edge detection and slope map for determining cliffs
        dem1_fill = inpaint_fill(dem1,filllimit=100)
        dem2_fill = inpaint_fill(dem2_flow_corrected,filllimit=100)
        
        slope1 = geolib.gdaldem_mem_ma(dem1_fill,ds_list_highres[0],processing='slope',
            returnma=True,computeEdges=True)
        slope2 = geolib.gdaldem_mem_ma(dem2_fill,ds_list_highres[1],processing='slope',
            returnma=True,computeEdges=True)
        
        #hotspot_binary_gdf = find_melthotspot(slope1,slope2,lag_dhdt-downslope_dhdt,ds=ds_list_highres[1],
        #   debris_cover_area=debris_ma,fill_final_mask=True,dhdt_sobelcutoff=0.60,return_sobel_layer=False,erode_iteration=1)
        if icecliff_gpkg is not None:
            print("Using user provided ice-cliff location file, will not compute ice cliff in this run")
            hotspot_binary_gdf = gpd.read_file(icecliff_gpkg)
        else:
            print("No user provided ice cliff file, will compute one from areas of high melt and high slope")
            hotspot_binary_gdf = high_slope_high_melt(slope1,slope2,lag_dhdt-downslope_dhdt,ds=ds_list_highres[1],
                debris_cover_area=debris_ma,dhdt_cutoff=-2.5,min_pixel_count=15)

        # restrict cliffs within debris only
        #cliff_debris_restrict = np.ma.array(cliff_mask,mask=debris_ma.mask)
        #smr_debris_restrict = np.ma.array(smr,mask=debris_ma.mask)
        
        ## Filter out smb pixels with anamolosly high values due to flux divergence values
        ### I will have to replace it later with a slope dependent reasoning
        #set max_smb to very high value
        max_smb = 12
        smb_clean = np.ma.masked_greater_equal(smb_dhdt,max_smb)
    
    if timescale == 'year':
        #there are edge boundary artifacts, so we will use shapefiles here to do best
        debris_temp = np.ma.array(np.ma.ones(debris_ma.shape),mask=np.ma.getmask(debris_ma),dtype=np.int16)
        debris_shp = binary2shapefile(debris_temp,1,ds=ds_list_highres[2])
        ice_shp = glac_shp.overlay(debris_shp,how='difference') # this is all bare ice
        background_shp = debris_shp.overlay(hotspot_binary_gdf,how='difference') # this is all background debris
        background_smb_dhdt = velocity_timeseries.mask_by_shp(background_shp.geometry,smb_clean,ds=ds_list_highres[1])
        hotspot_smb_dhdt = velocity_timeseries.mask_by_shp(hotspot_binary_gdf.geometry,smb_clean,ds=ds_list_highres[1])
        clean_ice_dhdt = velocity_timeseries.mask_by_shp(ice_shp.geometry,smb_clean,ds=ds_list_highres[1])
        #debris_smb_dhdt = np.ma.array(smb_clean,mask=debris_ma.mask)

        #background_smb_dhdt = np.ma.array(debris_smb_dhdt,mask=hotspot_binary_map)
        
        #hotspot_smb_dhdt = np.ma.array(debris_smb_dhdt,mask=~hotspot_binary_map)
        #clean_ice_dhdt = np.ma.array(smb_clean,mask=~debris_ma.mask)
        
    if timescale == 'year':
        print("***** Analysis2: Calcuating bin-wise melt stats *************")
        # limit elevation to glacier extent for binning
        base_elevation = np.ma.array(dem1,mask=H.mask)
        vm = np.ma.array(np.ma.sqrt(vx**2+vy**2),mask=H.mask)
        
        stats_df = hist_plot_gmbtools(hotspot_smb_dhdt,background_smb_dhdt,clean_ice_dhdt,smb_clean,debris_ma,vm,base_elevation,ds_list_highres[0])
        
        
        print("************ Creating plots ****************")
        print("******** Creating spatial map figure ************")
        f,ax = plt.subplots(1,4,figsize=(12,5))
        prepare_lag_smb_figure(eul_dhdt,lag_dhdt,downslope_dhdt,smb_dhdt,ds=ds_list_highres[0],ax=ax)
        outfig_spatial = os.path.join(outdir,f'{prefix}_smb_spatial_map.png')
        f.savefig(outfig_spatial,dpi=300,pad_inches=0.1,bbox_inches='tight')

    if timescale == 'year':
        print("******** Creating melt curve ************")
        outfig_meltcurve = os.path.join(outdir,f'{prefix}_altitudnal_meltcurve.png')
        plot_meltcurves(stats_df,prefix,hardcode_lim=(-20,10),outfn=outfig_meltcurve)
    
    print("************ Saving files *******************")
    eul_dhdt_fn = os.path.join(outdir,f'{prefix}_eulerian_dhdt.tif')
    lag_dhdt_fn = os.path.join(outdir,f'{prefix}_lagrangian_dhdt.tif')
    #downslope_dhdt_fn = os.path.join(outdir,f'{prefix}_downslope_dhdt.tif')
    smb_dhdt_fn = os.path.join(outdir,f'{prefix}_smb_dhdt.tif')
    
    if timescale == 'day':
        eul_dhdt = eul_dhdt * dt
        lag_dhdt = lag_dhdt * dt
        smb_dhdt = smb_dhdt * dt
    iolib.writeGTiff(eul_dhdt,eul_dhdt_fn,src_ds=ds_list_highres[0])
    iolib.writeGTiff(lag_dhdt,lag_dhdt_fn,src_ds=ds_list_highres[0])
    #iolib.writeGTiff(downslope_dhdt,downslope_dhdt_fn,src_ds=ds_list_highres[0])
    iolib.writeGTiff(smb_dhdt,smb_dhdt_fn,src_ds=ds_list_highres[0])
    if timescale == 'year':
        stats_fn = os.path.join(outdir,f'{prefix}_altitudnal_meltstats.csv')
        hotspot_shp_fn = os.path.join(outdir,f'{prefix}_hotspot_location.gpkg')
        stats_df.to_csv(stats_fn,index=False)
        #hotspot_gdf = binary2shapefile(hotspot_binary_map,1,ds_list_highres[0])
        hotspot_binary_gdf.to_file(hotspot_shp_fn,driver='GPKG')

    ## Prepare saving image script
    stats_df = 'none'
    return divQ2,eul_dhdt,lag_dhdt,downslope_dhdt,smb_dhdt,stats_df
