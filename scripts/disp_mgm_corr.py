#! /usr/bin/env python

##TODO: Pack all functions in a new featuretrack helper py
"""

use ASP parallel_stereo correlator to compute disparities
convert disparities to velocities/displacement using disp2v.py (vmap.py)
filter disparities based on NCC quality metric from ASP's corr_eval program
output final velocities at a given skip_size (lower res than input imagery) using gdaladdo -r gauss
# uses functions/structure from vmap.py (dshean) and autoRIFT_vmap.py 
For Imja, Changri Nup, Khumbu, Ngozumpa, Langtang, use as 
time disp_mgm_corr_eval.py <<hs1>>.tif <<hs2>>.tif -texture_smooth -dt yr -skip_rate 1
For Lirung, use as
time disp_mgm_corr_eval.py <<hs1>>.tif <<hs2>>.tif -texture_smooth -dt yr -skip_rate 1

"""

import os,sys,glob,shutil
import argparse
import subprocess
from datetime import datetime, timedelta
from distutils.spawn import find_executable
try:
    from osgeo import gdal
except:
    import gdal
import numpy as np

from pygeotools.lib import warplib, geolib, iolib,timelib,malib
from pygeotools.lib.malib import calcperc  

from debris_cover_smb import velocity_filter

## Utility functions
# we can resample the products at 8m producing products at skip size of 4 
# this is done using gdaladdo -r gauss operation
def resample_by_skip_size(fn,skip_size=4,return_ma=True):
    """
    skip size should be multiple of 4
    # adding padding for odd dimension arrays was given by David
    """
    src_ds = iolib.fn_getds(fn)
    ns = src_ds.RasterXSize
    nl = src_ds.RasterYSize
    # now pad with max overview scale
    p_olist = np.power(2,np.arange(1,10))
    # having overviews in power of 2 ensures correct application of 3x3 Gaussian kernel
    olist = list(p_olist[p_olist<=skip_size])
    ns_pad = int((np.floor(ns/skip_size)+1)*skip_size)
    nl_pad = int((np.floor(nl/skip_size)+1)*skip_size)
    dtype = src_ds.GetRasterBand(1).DataType
    ns_pad_diff = ns_pad - ns
    nl_pad_diff = nl_pad - nl
    xres,yres = geolib.get_res(src_ds)
    # need to maintain geotransform so pad extent properly
    xshift = ns_pad_diff*xres # this will be added to xmax
    yshift = nl_pad_diff*yres # this will be subtracted from ymin
    init_extent = geolib.ds_extent(src_ds)
    final_extent = [init_extent[0],init_extent[1]-yshift,init_extent[2]+xshift,init_extent[3]]
    projwin_extent = [final_extent[0],final_extent[3],final_extent[2],final_extent[1]]
    # get padded in-memory vrt with gdal_translate
    # due to a bug, need to write vrt to disk using the src_fn file path and then re-read the vrt to get overviews
    # see here: # http://osgeo-org.1560.x6.nabble.com/gdal-dev-Python-bindings-BuildOverviews-not-supported-for-VRT-dataset-td5429453.html
    # Latest gdal should have this fixed
    tgt_vrt = os.path.splitext(fn)[0]+'_gdal.vrt'
    mem_vrt = gdal.Translate(tgt_vrt,fn,width=ns_pad,height=nl_pad,format='VRT',projWin = projwin_extent)
    
    
    # close the mem_vrt dataset and then re-read
    mem_vrt = None
    mem_vrt = iolib.fn_getds(tgt_vrt)
    # now we build the overviews
    gdal.SetConfigOption('COMPRESS_OVERVIEW','LZW')
    gdal.SetConfigOption('BIGTIFF_OVERVIEW','YES')
    print("Building Overviews")
    mem_vrt.BuildOverviews("gauss",olist)
    
    
    # now read the bands and the corresponding dataset
    ma_sub,ma_sub_ds = iolib.ds_getma_sub(mem_vrt,bnum=1,scale=skip_size,return_ds=True)
    out_fn = f"{os.path.splitext(fn)[0]}_skiprate_{skip_size}.tif"
    ds_ndv = iolib.get_ndv_ds(ma_sub_ds,bnum=1)
    ds_gt = ma_sub_ds.GetGeoTransform()
    ds_proj = ma_sub_ds.GetProjection()
    iolib.writeGTiff(ma_sub,out_fn,src_ds=None,bnum=1,ndv=ds_ndv,gt=ds_gt,proj=ds_proj)
    ma_sub_ds = None
    os.remove(tgt_vrt)
    
    if return_ma:
        return ma_sub



#Generate and execute stereo commands (from vmap.py)
def run_cmd(bin, args, **kw):
    #Note, need to add full executable
    binpath = find_executable(bin)
    if binpath is None:
        msg = ("Unable to find executable %s\n" 
        "Install ASP and ensure it is in your PATH env variable\n" 
        "https://ti.arc.nasa.gov/tech/asr/intelligent-robotics/ngt/stereo/" % bin)
        sys.exit(msg)
    call = [binpath,]
    call.extend(args)
    print(' '.join(call))
    try:
        code = subprocess.call(call, shell=False)
    except OSError as e:
        raise Exception('%s: %s' % (binpath, e))
    if code != 0:
        raise Exception('Stereo step ' + kw['msg'] + ' failed')

def gen_d_sub(d_sub_fn, dx, dy, pad_perc=0.30, ndv=-9999):
    #from vmap.py
    nl = dx.shape[0]
    ns = dx.shape[1]
    #Use GDT_Byte or GDT_Int16 to save space?
    dtype = gdal.GDT_Int32
    opt = iolib.gdal_opt
    d_sub_ds = iolib.gtif_drv.Create(d_sub_fn, ns, nl, 3, dtype, opt)
    d_sub_ds.GetRasterBand(1).WriteArray(np.rint(dx.filled(ndv)).astype(np.int32))
    d_sub_ds.GetRasterBand(2).WriteArray(np.rint(dy.filled(ndv)).astype(np.int32))
    d_sub_ds.GetRasterBand(3).WriteArray((~dx.mask).astype(np.int32))
    for n in range(1, d_sub_ds.RasterCount+1):
        band = d_sub_ds.GetRasterBand(n)
        band.SetNoDataValue(float(ndv))
    d_sub_ds = None

    #Now write D_sub_spread.tif - defines spread around D_sub values
    d_sub_ds = iolib.fn_getds(d_sub_fn)
    d_sub_spread_fn = os.path.splitext(d_sub_fn)[0]+'_spread.tif'
    d_sub_spread_ds = iolib.gtif_drv.CreateCopy(d_sub_spread_fn, d_sub_ds, 0)
    dx_spread = np.ma.abs(dx * pad_perc)
    dy_spread = np.ma.abs(dy * pad_perc)
    d_sub_spread_ds.GetRasterBand(1).WriteArray(np.rint(dx_spread.filled(ndv)).astype(np.int32))
    d_sub_spread_ds.GetRasterBand(2).WriteArray(np.rint(dy_spread.filled(ndv)).astype(np.int32))
    d_sub_spread_ds.GetRasterBand(3).WriteArray((~dx_spread.mask).astype(np.int32))
    for n in range(1, d_sub_spread_ds.RasterCount+1):
        band = d_sub_spread_ds.GetRasterBand(n)
        band.SetNoDataValue(float(ndv))
    d_sub_spread_ds = None
    #Copy proj/gt to D_sub and D_sub_spread?


def getparser():
    parser = argparse.ArgumentParser(description="Generate velocity map via feature-tracking")
    parser.add_argument('-outdir', default=None, help='Output directory')
    parser.add_argument('-threads', type=int, default=iolib.cpu_count(), help='Number of threads to use(default: %(default)s)')
    parser.add_argument('-tr', default='min', help='Output resolution (default: %(default)s)')
    #Set correlator kernel size
    corr_kernel_choices = [3,5,7,9]
    parser.add_argument('-corr_kernel', choices=corr_kernel_choices,type=int, 
        default=9, 
        help='Correlator kernel size. Smaller kernels offer more detail but are prone to more noise. Odd integers required (~3-9 px recommended). (default: %(default)s)')
    # choices=corr_kernel_choices
    parser.add_argument('-rfne_kernel',type=int, 
        default=15, 
        help='Refinement (for subpixel displacement estimation) kernel. Smaller kernels offer more detail but are prone to more noise. Odd integers required (~3-9 px recommended). (default: %(default)s)')
    refinement_choices = list(range(12))
    parser.add_argument('-texture_smooth',action='store_true')
    parser.add_argument('-refinement', type=int, default=9, 
        help='Sub-pixel refinement type (see ASP doc): 0 -no subpixel refinement, 1 - parabola fitting, 2 - affine adaptive window, Bayes EM weighting, 3 - affine window, 4 - phase correlation, 5 - Lucas-Kanade method (experimental), 6 - affine adaptive window, Bayes EM with Gamma Noise Distribution (experimental), 7 - SGM None, 8 - SGM linear, 9 - SGM Poly4, 10 - SGM Cosine, 11 - SGM Parabola, 12 - SGM Blend (default: %(default)s)')
    #Integer correlator seeding
    #D_sub is low-resolution correlation (default), which works well for most situations
    #sparse_disp will use sparse seeding from full-res chips, useful for ice sheets with limited low-frequency texture (not supported here)
    #existing_velocity will accept existing vx and vy rasters.  Useful for limiting search range and limiting blunders.  Measures products are useful for ice sheets.
    seedmode_choices = ['D_sub','existing_velocity']
    parser.add_argument('-seedmode', type=str, choices=seedmode_choices, default='D_sub', help='Seeding option (default: %(default)s)')
    parser.add_argument('-vx_fn', type=str, default=None, help='Seed E-W velocity map filename')
    parser.add_argument('-vy_fn', type=str, default=None, help='Seed N-S velocity map filename')
    
    #Numer of gaussian pyramids to use
    #Can look at texture in GDAL overviews to make a decision
    #If you can see plenty of texture at 1/32 resolution, go with 5 
    #For featureless areas, limiting to 2 can help, or even 0
    parser.add_argument('-pyramid_levels', type=int, default=5, 
        help='Number of pyramid levels for correlation (default: %(default)s)')
    #This helps get rid of bogus "islands" in the disparity maps
    parser.add_argument('-erode', type=int, default=1024, 
        help='Erode isolated blobs smaller than this many pixels. Set to 0 to disable (default: %(default)s)')

    parser.add_argument('-remove_offsets', action='store_true', 
        help='Remove median horizontal and vertical offsets over stable control surfaces')
    parser.add_argument('-dt', type=str, choices=['yr','day','m','px'], default='yr', 
        help='Time increment (default: %(default)s)')
 
    parser.add_argument('-ncc_cutoff',default=None,type=float,
        help='filter values with NCC quality metric less than the input value here')
    skip_rate_choice = [1,2,4,8,16,32,64]
    parser.add_argument('-skip_rate',default=4,type=int,help='produce final grids at skip_rate*input_res(default: %(default)s)')
    fill_gap_choice = ['linear','cubic','inpaint']
    parser.add_argument('-fill_gap_method',default=None,choices=fill_gap_choice,
        help='fill gaps using the selected method (default: %(default)s)')
    parser.add_argument('-fillimit',default=100,type=int,
        help='do not fill gaps if the semi-minor axis of data gap is larger than filllim pixels (default: %(default)s)')
    parser.add_argument('-overwrite',action='store_true',help='overwrite currently existing files if true')
    parser.add_argument('-pleiades',action='store_true',help='use pleiades specific thread settings, otherwise use default settings')
    #Inputs can be images, DEMs, shaded relief maps
    #Personal experience suggests multi-directional hillshades with identical illumination work well
    #Only 2 input datsets allowed for this - want to stay modular
    parser.add_argument('fn1', type=str, help='Raster filename 1')
    parser.add_argument('fn2', type=str, help='Raster filename 2')
    return parser

def disp2v(dx,dy,xres,yres,t_factor,dt='day',remove_offsets=False,mask_list=['glaciers'],src_fn=None,src_ds=None):
    """
    python function for vmap's disp2v program
    Parameters
    ------------
    Returns
    ------------
    """
    if dt == 'day':
        t_factor = t_factor*365.25
    if dt == 'px':
        h_myr = dx
        v_myr = dy
        t_factor = 1
    elif dt == 'm':
        h_myr = dx * xres
        v_myr = dy * yres
        t_factor = 1
    else:
        h_myr = (dx * xres)/t_factor
        v_myr = (dy * yres)/t_factor

    m = np.ma.sqrt(h_myr**2+v_myr**2)
    malib.print_stats(m)
    if remove_offsets:
        print("\nUsing demcoreg to prepare mask of stable control surfaces\n")
        #mask = get_lulc_mask(src_ds, mask_glaciers=True, filter='rock+ice+water')
        # just have glaciers for now
        #TODO:
        # Probably best to predefine mask over static low sloped area and pass via mask_fn option
        from demcoreg import dem_mask
        mask = dem_mask.get_mask(src_ds,mask_list=mask_list,dem_fn=src_fn)
        #these are not tested
        
        # do the actual offset correction
        print("\nRemoving median x and y offset over static control surfaces")
        h_myr_count = h_myr.count()
        h_myr_static_count = np.ma.array(h_myr,mask=mask).count()
        h_myr_med = malib.fast_median(np.ma.array(h_myr,mask=mask))
        v_myr_med = malib.fast_median(np.ma.array(v_myr,mask=mask))
        h_myr_mad = malib.mad(np.ma.array(h_myr,mask=mask))
        v_myr_mad = malib.mad(np.ma.array(v_myr,mask=mask))
        h_spread_init = malib.calcperc(np.ma.array(h_myr,mask=mask),(16,84))
        v_spread_init = malib.calcperc(np.ma.array(v_myr,mask=mask),(16,84))
        print("Static pixel count: %i (%0.1f%%)" % (h_myr_static_count, 100*float(h_myr_static_count)/h_myr_count))
        print("median (+/-NMAD)")
        print("x velocity offset: %0.2f (+/-%0.2f) m/%s" % (h_myr_med, h_myr_mad, dt))
        print("y velocity offset: %0.2f (+/-%0.2f) m/%s" % (v_myr_med, v_myr_mad, dt))
        h_myr -= h_myr_med
        v_myr -= v_myr_med
        offset_str = '_offsetcorr_h%0.2f_v%0.2f' % (h_myr_med, v_myr_med)
        h_spread_final = malib.calcperc(np.ma.array(h_myr,mask=mask),(16,84))
        v_spread_final = malib.calcperc(np.ma.array(v_myr,mask=mask),(16,84))
        #Velocity Magnitude
        m = np.ma.sqrt(h_myr**2+v_myr**2)
        rad_med = malib.fast_median(np.ma.array(m,mask=mask))
        rad_mad = malib.mad(np.ma.array(m,mask=mask))
        rad_spread_fn = malib.calcperc(np.ma.array(m,mask=mask),(16,84))
        json_dict = {'count':int(h_myr_static_count),'x_med':h_myr_med*t_factor,'y_med':v_myr_med*t_factor,
            'x_nmad':h_myr_mad*t_factor,'y_nmad':v_myr_mad*t_factor,'x_16p_init':h_spread_init[0]*t_factor,
            'x_84p_init':h_spread_init[1]*t_factor,'y_16p_init':v_spread_init[0]*t_factor,'y_84p_init':v_spread_init[1]*t_factor,
            'x_16p_fin':h_spread_final[0]*t_factor,'x_84p_fin':h_spread_final[1]*t_factor,'y_16p_fin':v_spread_final[0]*t_factor,
            'y_84p_fin':v_spread_final[1]*t_factor,'rad_med_fin':rad_med*t_factor,'rad_mad_fin':rad_mad*t_factor,
            'rad_16p_fin':rad_spread_fn[0]*t_factor,'rad_84p_fin':rad_spread_fn[1]*t_factor}
        print("Velocity Magnitude stats after correction")
        malib.print_stats(m)
        out = [h_myr,v_myr,m,offset_str,json_dict]
    else:
        offset_str = ''
        out = [h_myr,v_myr,m,offset_str]
    return out

def get_correlator_opt(corr_kernel=(9,9),nlevels=5,spr=9,
    rfne_kernel=(15,15),erode=0,align='None',txm_size=50,entry_point=0,stop_point=None,texture_smooth=False,
    median_filter_size=41,pleiades=False):
    correlator_opt = []
    correlator_opt.extend(['--correlator-mode'])
    correlator_opt.extend(['--ip-per-tile', str(2000)])
    correlator_opt.extend(['--stereo-algorithm', 'asp_mgm'])
    correlator_opt.extend(['--corr-kernel', str(corr_kernel[0]), str(corr_kernel[1])])
    #correlator_opt.extend(['--cost-mode', str(4)])
    
    correlator_opt.extend(['--subpixel-mode',str(spr)])
    correlator_opt.extend(['--subpixel-kernel', str(rfne_kernel[0]), str(rfne_kernel[1])])
    if texture_smooth:
        correlator_opt.extend(['--texture-smooth-scale',str(0.5)])
        correlator_opt.extend(['--texture-smooth-size',str(txm_size)])

        correlator_opt.extend(['--median-filter-size', str(median_filter_size)])
    correlator_opt.extend(['-e', str(entry_point)])
    correlator_opt.extend(['--corr-sub-seed-percent',str(0.40)])
    if stop_point is not None:
        correlator_opt.extend(['--stop-point', str(stop_point)])

    if pleiades:
        # copy settings from stereo processing
        corrtilesize=1024
        corrmemlim=5000
        pstereo_proc=18
        
        # 4 = ternary census transform - use for SGM/MGM
        costmode=3

        # bro has 56 total cores (28 phyiscal + 28 threading)
        # 25 correlation processes, using 2 threads each
        thread_parallel=2 
        # bro has 28 physical cores, use 24 ?
        thread_single=24
        correlator_opt.extend(['--corr-tile-size',str(corrtilesize)])
        correlator_opt.extend(['--corr-memory-limit-mb', str(corrmemlim)])
        correlator_opt.extend(['--cost-mode',str(costmode)])
        correlator_opt.extend(['--processes', str(pstereo_proc)])
        correlator_opt.extend(['--threads-multiprocess', str(thread_parallel)])
        correlator_opt.extend(['--threads-singleprocess', str(thread_single)])
        #correlator_opt.extend(['--keep-only', '-D.tif', '-RD.tif', '-F.tif'])
    return correlator_opt

def gen_d_sub(d_sub_fn, dx, dy, pad_perc=0.1, ndv=-9999):
    nl = dx.shape[0]
    ns = dx.shape[1]
    #Use GDT_Byte or GDT_Int16 to save space?
    dtype = gdal.GDT_Int32
    opt = iolib.gdal_opt
    d_sub_ds = iolib.gtif_drv.Create(d_sub_fn, ns, nl, 3, dtype, opt)
    d_sub_ds.GetRasterBand(1).WriteArray(np.rint(dx.filled(ndv)).astype(np.int32))
    d_sub_ds.GetRasterBand(2).WriteArray(np.rint(dy.filled(ndv)).astype(np.int32))
    d_sub_ds.GetRasterBand(3).WriteArray((~dx.mask).astype(np.int32))
    for n in range(1, d_sub_ds.RasterCount+1):
        band = d_sub_ds.GetRasterBand(n)
        band.SetNoDataValue(float(ndv))
    d_sub_ds = None

    #Now write D_sub_spread.tif - defines spread around D_sub values
    d_sub_ds = iolib.fn_getds(d_sub_fn)
    d_sub_spread_fn = os.path.splitext(d_sub_fn)[0]+'_spread.tif'
    d_sub_spread_ds = iolib.gtif_drv.CreateCopy(d_sub_spread_fn, d_sub_ds, 0)
    dx_spread = np.ma.abs(dx * pad_perc)
    dy_spread = np.ma.abs(dy * pad_perc)
    d_sub_spread_ds.GetRasterBand(1).WriteArray(np.rint(dx_spread.filled(ndv)).astype(np.int32))
    d_sub_spread_ds.GetRasterBand(2).WriteArray(np.rint(dy_spread.filled(ndv)).astype(np.int32))
    d_sub_spread_ds.GetRasterBand(3).WriteArray((~dx_spread.mask).astype(np.int32))
    for n in range(1, d_sub_spread_ds.RasterCount+1):
        band = d_sub_spread_ds.GetRasterBand(n)
        band.SetNoDataValue(float(ndv))
    d_sub_spread_ds = None
    #Copy proj/gt to D_sub and D_sub_spread?


def main():
    parser = getparser()
    args = parser.parse_args()
    if args.seedmode == 'existing_velocity':
        if args.vx_fn is None or args.vy_fn is None:
            parser.error('"-seedmode existing_velocity" requires "-vx_fn" and "-vy_fn"')

    print('\n%s' % datetime.now())
    print('%s UTC\n' % datetime.utcnow())

    corr_kernel = (args.corr_kernel,args.corr_kernel)
    spr = args.refinement
    rfne_kernel = (args.rfne_kernel,args.rfne_kernel)
    seedmode = args.seedmode
    res = args.tr 
    erode = args.erode
    #Open input files
    fn1 = args.fn1
    fn2 = args.fn2 

    if not iolib.fn_check(fn1) or not iolib.fn_check(fn2):
        sys.exit("Unable to locate input files")

    dt_list = [timelib.fn_getdatetime(x) for x in [fn1,fn2]]
    t_factor = timelib.get_t_factor(dt_list[0],dt_list[1])
    
    prefix =  f"{os.path.splitext(os.path.split(fn1)[1])[0]}__{os.path.splitext(os.path.split(fn2)[1])[0]}_mgm_disp_{spr}spm_spm_ker{rfne_kernel[0]}_corr_kernel{corr_kernel[0]}px_res{res}"
    if args.texture_smooth:
        prefix = prefix+"_txm"
        
    if args.outdir is not None:
        outdir = args.outdir
    
    else:
        outdir = prefix
    
    #Note, can encounter filename length issues in boost, just use vmap prefix
    outprefix = f'{outdir}/vmap' 
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    #Check to see if inputs have geolocation and projection information
    ds1 = iolib.fn_getds(fn1)
    ds2 = iolib.fn_getds(fn2)

    if geolib.srs_check(ds1) and geolib.srs_check(ds2):
        ds1_clip_fn = os.path.join(outdir, os.path.splitext(os.path.basename(fn1))[0]+'_warp.tif')
        ds2_clip_fn = os.path.join(outdir, os.path.splitext(os.path.basename(fn2))[0]+'_warp.tif')

        if not os.path.exists(ds1_clip_fn) or not os.path.exists(ds2_clip_fn):
            #This should write out files to new subdir
            ds1_clip, ds2_clip = warplib.diskwarp_multi_fn([fn1, fn2], extent='intersection', res=res, r='average', outdir=outdir)
            ds1_clip = None
            ds2_clip = None
            #However, if inputs have identical extent/res/proj, then link to original files
            if not os.path.exists(ds1_clip_fn):
                os.symlink(os.path.abspath(fn1), ds1_clip_fn)
            if not os.path.exists(ds2_clip_fn):
                os.symlink(os.path.abspath(fn2), ds2_clip_fn)
   

    else:
        ds1_clip_fn = fn1
        ds2_clip_fn = fn2
    align = 'None'
    ds1 = None
    ds2 = None

    corr_args = [ds1_clip_fn,ds2_clip_fn,outprefix]
    txm_size = 15
    # this block below computes disparity measurements using ASP MGM correlator
    if os.path.exists(outprefix+"-F.tif"):
        if args.overwrite:
            run_asp_corr = True
        else:
            run_asp_corr = False 
    else:
        run_asp_corr = True
    print (f"run asp corr is {run_asp_corr}")
    t1 = timelib.fn_getdatetime(fn1)
    t2 = timelib.fn_getdatetime(fn2)
    if (t1 == timelib.strptime_fuzzy('20161106')) and (t2 == timelib.strptime_fuzzy('20171222')):
        print("this run is for Lirung Glacier, will use a smaller median filter")
        median_filter_size = 21
    else:
        median_filter_size = 41
    if run_asp_corr: 
        if seedmode == "D_sub":
            correlator_opt = get_correlator_opt(corr_kernel=corr_kernel,nlevels=args.pyramid_levels,
            spr=spr, rfne_kernel=rfne_kernel,erode=erode,align=align,entry_point=0,
            txm_size=txm_size,texture_smooth=args.texture_smooth,median_filter_size=median_filter_size,pleiades=args.pleiades)
            print("No seed velocity provided, D_sub will be caluclated by IP matching")
            run_cmd('parallel_stereo',correlator_opt+corr_args,msg='Full correlation')

        
        elif seedmode == 'existing_velocity':
            pprc_opt = get_correlator_opt(corr_kernel=corr_kernel,nlevels=args.pyramid_levels,
            spr=spr, rfne_kernel=rfne_kernel,erode=erode,align=align,entry_point=0,stop_point=1,pleiades=args.pleiades)
            print("Running stereo preprocessing")
            run_cmd('parallel_stereo',pprc_opt+corr_args,msg='0:stereo pprc')

            # now prepare D_sub from existing velocity maps
            print("preparing D_sub from input seed velocities")
            vx_fn = args.vx_fn 
            vy_fn = args.vy_fn
            if os.path.exists(vx_fn) and os.path.exists(vy_fn):
                    ds1_clip = iolib.fn_getds(ds1_clip_fn)
                    ds1_res = geolib.get_res(ds1_clip, square=True)[0]
                #Compute L_sub res - use this for output dimensions
                    L_sub_fn = outprefix+'-L_sub.tif' 
                    L_sub_ds = gdal.Open(L_sub_fn)
                    L_sub_x_scale = float(ds1_clip.RasterXSize) / L_sub_ds.RasterXSize
                    L_sub_y_scale = float(ds1_clip.RasterYSize) / L_sub_ds.RasterYSize
                    L_sub_scale = np.max([L_sub_x_scale, L_sub_y_scale])
                    L_sub_res = ds1_res * L_sub_scale

                    #Since we are likely upsampling here, use cubicspline
                    vx_ds_clip, vy_ds_clip = warplib.memwarp_multi_fn([vx_fn, vy_fn], extent=ds1_clip, \
                            t_srs=ds1_clip, res=L_sub_res, r='cubicspline')
                    ds1_clip = None

                    #Get vx and vy arrays
                    vx = iolib.ds_getma(vx_ds_clip)
                    vy = iolib.ds_getma(vy_ds_clip)

                    #Determine time interval between inputs
                    #Use to scaling of known low-res velocities
                    t_factor_seed = timelib.get_t_factor_fn(ds1_clip_fn, ds2_clip_fn, ds=vx_ds_clip)
                    #Compute expected offset in scaled pixels 
                    dx = (vx*t_factor_seed)/L_sub_res
                    dy = (vy*t_factor_seed)/L_sub_res
                    #This is relative to the D_sub scaled disparities
                    d_sub_fn = L_sub_fn.split('-L_sub')[0]+'-D_sub.tif' 
                    gen_d_sub(d_sub_fn, dx, dy)
                    

                    ## Now we have D_sub, we will proceed with full resolution correlation
                    correlator_opt = get_correlator_opt(corr_kernel=corr_kernel,nlevels=args.pyramid_levels,
                        spr=spr, rfne_kernel=rfne_kernel,erode=erode,align=align,entry_point=1,
                        txm_size=txm_size,texture_smooth=args.texture_smooth,median_filter_size=median_filter_size,pleiades=args.pleiades)
                    correlator_opt.extend(['--skip-low-res-disparity-comp'])

                    print("Running full correlation")
                    run_cmd('parallel_stereo',correlator_opt+corr_args,msg='0:correlation')
                    
            else:
                print("Seed velocity files could not be located, exiting")
        


    # Now we should have filtered disparities, we would like to convert to velocity
    
    ds_disp = iolib.fn_getds(outprefix+"-F.tif")
    
    res_x,res_y = geolib.get_res(ds_disp,square=True)
    dx = iolib.ds_getma(ds_disp,bnum=1)
    dy = -1*iolib.ds_getma(ds_disp,bnum=2)

    if args.ncc_cutoff is not None:
        print("will run corr_eval to compute NCC quality metric")
        corr_eval_opt = []
        corr_eval_opt.extend(['--prefilter-mode', str(2)])
        #corr_eval_opt.extend(['--kernel-size', str(txm_size), str(txm_size)])
        if args.texture_smooth:
            corr_eval_opt.extend(['--kernel-size', str(txm_size), str(txm_size)])
        else:
            corr_eval_opt.extend(['--kernel-size', str(corr_kernel[0]), str(corr_kernel[1])])
            #corr_eval_opt.extend(['--kernel-size', str(rfne_kernel[0]), str(rfne_kernel[1])])
        corr_eval_opt.extend(['--metric','ncc'])
        corr_eval_opt.extend(['--threads',f'{iolib.cpu_count()}'])
        corr_eval_args = [outprefix+'-L.tif',outprefix+'-R.tif',outprefix+'-F.tif',outprefix]
        if (os.path.exists(outprefix+'-ncc.tif') is False) | (args.overwrite):
            run_cmd('corr_eval',corr_eval_opt+corr_eval_args,msg='Corr eval')
        ncc_ma = iolib.fn_getma(outprefix+'-ncc.tif')
        ncc_mask = np.ma.masked_less_equal(ncc_ma,args.ncc_cutoff).mask
        ncc_str = f'_ncc_cutoff'
        dx = np.ma.array(dx,mask=ncc_mask)
        dy = np.ma.array(dy,mask=ncc_mask)
        # now remove isolated pixels by erode edge function
        dx = velocity_filter.erode_edge(dx,iterations=4)
        dy = velocity_filter.erode_edge(dy,iterations=4)
        #dx = malib.mask_islands(dx, iterations=1)
        #dy = malib.mask_islands(dy, iterations=1)
    else:
        ncc_str = ''
    
    
    if args.remove_offsets:
        vx,vy,vm,offset_str,json_dict = disp2v(dx,dy,res_x,np.abs(res_y),
        t_factor,args.dt,remove_offsets=args.remove_offsets,
        src_fn=outprefix+"-F.tif",src_ds=ds_disp)
        import json
        json_fn = f'{outprefix}_static_stats.json'
        with open(json_fn, 'w') as f:
            json.dump(json_dict, f)

    else:
        vx,vy,vm,offset_str = disp2v(dx,dy,res_x,np.abs(res_y),t_factor,args.dt,
        remove_offsets=args.remove_offsets,
        src_fn=outprefix+"-F.tif",src_ds=ds_disp)

    # skip rate part is very clumsy right now
    skip_rate = args.skip_rate 
    vm_fn = f"{outdir}/{prefix}_vm{offset_str}{ncc_str}.tif"
    vx_fn = f"{outdir}/{prefix}_vx{offset_str}{ncc_str}.tif"
    vy_fn = f"{outdir}/{prefix}_vy{offset_str}{ncc_str}.tif"
    
    
    ds_gt = ds_disp.GetGeoTransform()
    ds_proj = ds_disp.GetProjection()
    iolib.writeGTiff(vx,vx_fn,ndv=-9999.0,src_ds=None,proj=ds_proj,gt=ds_gt,bnum=1)
    iolib.writeGTiff(vy,vy_fn,ndv=-9999.0,src_ds=None,proj=ds_proj,gt=ds_gt,bnum=1)
    iolib.writeGTiff(vm,vm_fn,ndv=-9999.0,src_ds=None,proj=ds_proj,gt=ds_gt,bnum=1)

    fvm_fn = f"{outdir}/{prefix}_vm{offset_str}{ncc_str}_skiprate_{skip_rate}.tif"
    fvx_fn = f"{outdir}/{prefix}_vx{offset_str}{ncc_str}_skiprate_{skip_rate}.tif"
    fvy_fn = f"{outdir}/{prefix}_vy{offset_str}{ncc_str}_skiprate_{skip_rate}.tif"
    vm = None
    vx = None
    vy = None
    ds_disp = None 
    if skip_rate == 1:
        print("Skip rate is just 1, copying")
        shutil.copy2(vm_fn,fvm_fn)
        shutil.copy2(vx_fn,fvx_fn)
        shutil.copy2(vy_fn,fvy_fn)
    else:
        resample_by_skip_size(vm_fn,skip_size=skip_rate,return_ma=False)
        resample_by_skip_size(vx_fn,skip_size=skip_rate,return_ma=False)   
        resample_by_skip_size(vy_fn,skip_size=skip_rate,return_ma=False) 
    os.remove(vm_fn)
    os.remove(vy_fn)
    os.remove(vx_fn)
    if args.fill_gap_method is not None:
        
        velocity_filter.fillgap_wrapper(fvx_fn,method=args.fill_gap_method,
            filllimit=args.fillimit,erode_island=True,return_ma=False)
        velocity_filter.fillgap_wrapper(fvy_fn,method=args.fill_gap_method,
            filllimit=args.fillimit,erode_island=True,return_ma=False)
        velocity_filter.fillgap_wrapper(fvm_fn,method=args.fill_gap_method,
            filllimit=args.fillimit,erode_island=True,return_ma=False)

if __name__ == "__main__":
    main()
