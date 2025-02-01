#! /usr/bin/env python
import numpy as np 
import geopandas as gpd
import matplotlib.pyplot as plt
from imview import pltlib
import pandas as pd
from pygeotools.lib import warplib,geolib,iolib,malib,filtlib,timelib
import os,sys,glob
from debris_cover_smb import glac_dyn, constants
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.metrics import peak_signal_noise_ratio
from skimage.measure import label, regionprops
from skimage.morphology import binary_erosion, disk


def nlm_scipy(ma_ar):
    # estimate the noise standard deviation from the noisy image
    sigma_est = np.mean(estimate_sigma(ma_ar))
    print(f'estimated noise standard deviation = {sigma_est}')
    patch_kw = dict(patch_size=5,      # 5x5 patches
                patch_distance=6,  # 13x13 search area
               )
    # fast algorithm, sigma provided
    denoise2_fast = denoise_nl_means(ma_ar, h=0.6 * sigma_est, sigma=sigma_est,
                                 fast_mode=True, **patch_kw)
    fill_val = ma_ar.fill_value
    return np.ma.masked_equal(denoise2_fast,fill_val)

def local_avg_filter(ma_ar,operator='mean',kernel_size=3,avg_diff_perc=30,return_ma=False):
    if operator == 'mean':
        ## Experimental directional filter Part-2
        # from astropy.convolution import Box2DKernel
        # https://waterprogramming.wordpress.com/2018/09/04/implementation-of-the-moving-average-filter-using-convolution/
        from astropy.convolution import convolve,Box2DKernel
        av_value = convolve(ma_ar,kernel=Box2DKernel(kernel_size))
    else:
        av_value = filtlib.rolling_fltr(ma_ar,f=np.nanmedian,size=kernel_size,
                                        circular=False)
    # compute percentage difference from computed local mean
    filt_diff_perc = (np.ma.abs(av_value-ma_ar)/ma_ar)*100
    filtered_ar = np.ma.masked_greater(filt_diff_perc,avg_diff_perc)
    if return_ma:
        out = [filtered_ar.mask,np.ma.array(ma_ar,mask=filtered_ar.mask)]
    else:
        out = [filtered_ar.mask]
    return out


def filter_by_rolling_statsfilter(ma_ar,operator='difference',window=5,deviation=30):
    """
    #### TODO: Compute rolling std and filter dynamically
    Filter values in an array which deviate more than the mean or median of the sorrounding *window* px values
    Parameters
    ------------
    ma_ar: np.ma.array
        array to be filtered
    operator: str
        mean or median operator for computing local stats
    window: int
        odd integer value for computing local spatial stats
    deviation: numeric
        allowed deviation from locally averaged/median value 
    Returns
    -------------
    filtered_ar: np.ma.array
        filtered array
    """
    
    if operator == 'mean':
        ## Experimental directional filter Part-2
        # from astropy.convolution import Box2DKernel
        # https://waterprogramming.wordpress.com/2018/09/04/implementation-of-the-moving-average-filter-using-convolution/
        from astropy.convolution import convolve,Box2DKernel
        av_value = convolve(ma_ar,kernel=Box2DKernel(window))
    else:
        av_value = filtlib.rolling_fltr(ma_ar,f=np.nanmedian,size=window,
                                        circular=False)
    filtered_ar = np.ma.masked_greater(np.ma.abs(ma_ar - av_value), deviation)
    return filtered_ar.mask

    def fill_gaps_local_median(ma_ar,window_size=11):
        orig_mask = ma_ar.mask
        median_ar = filtlib.rolling_fltr(ma_ar,f=np.nanmedian,size=11,
                                            circular=True)
        out = np.ma.masked_equal(ma_ar.filled(0)+np.ma.array(median_ar,mask=~orig_mask).filled(0),0)
        return out


#https://stackoverflow.com/questions/37662180/interpolate-missing-values-2d-python
from scipy import interpolate
import numpy as np

def interpolate_missing_pixels(
        image: np.ndarray,
        mask: np.ndarray,
        method: str = 'nearest',
        fill_value: int = 0
):
    """
    :param image: a 2D image
    :param mask: a 2D boolean image, True indicates missing values
    :param method: interpolation method, one of
        'nearest', 'linear', 'cubic'.
    :param fill_value: which value to use for filling up data outside the
        convex hull of known pixel values.
        Default is 0, Has no effect for 'nearest'.
    :return: the image with missing values interpolated
    """
    from scipy import interpolate

    h, w = image.shape[:2]
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))

    known_x = xx[~mask]
    known_y = yy[~mask]
    known_v = image[~mask]
    missing_x = xx[mask]
    missing_y = yy[mask]

    interp_values = interpolate.griddata(
        (known_x, known_y), known_v, (missing_x, missing_y),
        method=method, fill_value=fill_value
    )

    interp_image = image.copy()
    interp_image[missing_y, missing_x] = interp_values

    return interp_image

def interpolate_gaps(ma_ar,filllimit=19,kind='linear',verbose=False):
    #filllimit = 1
    filldem = None
    # Find nodata blobs and get their properties
    nodata_blobs = label(ma_ar.mask)
    nodata_regions = regionprops(nodata_blobs)
    ma_ar_temp = ma_ar.copy()
    for props in nodata_regions:
        if verbose:
            print(
                f"{props.label}/{len(nodata_regions)}, area: {props.area}"
            )
        if props.axis_minor_length > filllimit:
            if verbose:
                print(f"{props.label} filling with filldem")
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
                if verbose:
                    print(f"Skipping {props.label}")
                for coord in props.coords:
                    ma_ar_temp.mask[coord[0], coord[1]] = False
                continue
        # fill the ma_ar
    ma_fill = interpolate_missing_pixels(ma_ar.data,ma_ar.mask,kind,-9999)
    out = np.ma.masked_equal(np.ma.filled(ma_ar,0)+np.ma.array(np.ma.masked_equal(ma_fill,-9999),mask=~ma_ar_temp.mask).filled(0),0)
    return out

def compute_bearing_from_N(vx,vy):
    # Compute bearing of velocity
    direction_vector = np.rad2deg(np.arctan2(vx,vy))
    # correct bearing for third and fourth quadrant
    neg_idx = direction_vector < 0
    direction_vector[neg_idx] = direction_vector[neg_idx] + 360
    # this should be the bearing from North increasing in the clockwise direction
    return direction_vector

def add_quiver(ax, vx, vy, stride=10, color='dodgerblue',scale=5, quiver_key=True,
        quiver_key_units=20,quiver_key_unit_rep='m/yr',res=16):
    X = np.arange(0,vx.shape[1],stride)
    Y = np.arange(0,vx.shape[0],stride)
    Q = ax.quiver(X, Y, vx[::stride,::stride]*2, vy[::stride,::stride]*2, color=color,
     pivot='mid',units='xy',scale=scale,scale_units='xy')
    if quiver_key:
        if quiver_key_unit_rep == 'm/yr':
            label = f'{quiver_key_units}'+r'$\frac{m}{yr}$'
        elif quiver_key_unit_rep == 'm/day':
            label = f'{quiver_key_units}'+r'$\frac{m}{day}$'

        qk = ax.quiverkey(Q, 0.3, 1.02,U=quiver_key_units*scale*0.5, label=label, labelpos='E',
                   coordinates='axes')
        #X=0.3, Y=1.1, U=lkey,
            # label='Quiver key, length = '+str(lkey)+' m/s', labelpos='E')
    

def filter_by_ncc(vm,ncc,ncc_lim=0.9):
    mask = np.ma.masked_less(ncc,ncc_lim).mask
    vm_masked = np.ma.array(vm,mask=mask)
    return vm_masked

# from filtlib
# for some reaseon these two functions are not importable from filtlib
def erode_edge(dem, iterations=1):
    #https://github.com/dshean/pygeotools/blob/e780b4ced1c9fda19d20ba6c41839d9ffa1247d3/pygeotools/lib/filtlib.py#L462
    """Erode pixels near nodata
    """
    import scipy.ndimage as ndimage 
    print('Eroding pixels near nodata: %i iterations' % iterations)
    mask = np.ma.getmaskarray(dem)
    mask_dilate = ndimage.morphology.binary_dilation(mask, iterations=iterations)
    out = np.ma.array(dem, mask=mask_dilate)
    return out

def remove_islands(dem, iterations=1):
    #https://github.com/dshean/pygeotools/blob/e780b4ced1c9fda19d20ba6c41839d9ffa1247d3/pygeotools/lib/filtlib.py#L472
    """Remove isolated pixels
    """
    import scipy.ndimage as ndimage 
    print('Removing isolated pixels: %i iterations' % iterations)
    mask = np.ma.getmaskarray(dem)
    mask_dilate = ndimage.morphology.binary_dilation(mask, iterations=iterations)
    mask_dilate_erode = ~(ndimage.morphology.binary_dilation(~mask_dilate, iterations=iterations))
    out = np.ma.array(dem, mask=mask_dilate_erode)
    return out

def fillgap_wrapper(fn,method='linear',filllimit=100,erode_island=True,return_ma=True):
    """
    choose between 
    1. linear (in 2d this is bilinear),
    2. cubic (in 2d this is is bicubic),
    3. inpaint (this is based on skimage, R. Beyer's code snippet
    """
    ds = iolib.fn_getds(fn)
    ma = iolib.ds_getma(ds)
    if method != 'inpaint':
        ma_filled = interpolate_gaps(ma,filllimit=filllimit,kind=method)
    else:
        ma_filled = glac_dyn.inpaint_fill(ma,filllimit=filllimit)
    if erode_island:
        ma_filled = malib.mask_islands(ma_filled,iterations=3)
    outfn = f"{os.path.splitext(fn)[0]}_{method}_limit_{filllimit}px.tif"
    print(f"writing filled array at {outfn}")
    iolib.writeGTiff(ma_filled,outfn,src_ds=ds)
    if return_ma:
        return ma_filled

