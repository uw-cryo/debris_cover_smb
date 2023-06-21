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

def gauss_fltr_astropy_fft(dem, size=None, sigma=None, origmask=False, fill_interior=False):
    
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
            mask = dem.mask
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