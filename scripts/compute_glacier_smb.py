#! /usr/bin/env python

import argparse
import os,sys,glob
from debris_cover_smb import glac_dyn,constants
from datetime import datetime
import geopandas as gpd 

def getparser():
    parser = argparse.ArgumentParser(description="Compute dhdt from DEMs and ice-thickness rasters")
    parser.add_argument('-dem1_fn', default=None, required=True, type=str, help='path to DEM 1')
    parser.add_argument('-dem2_fn', default=None, required=True, type=str, help='path to DEM 2')    
    parser.add_argument('-vx_fn', default=None, required=True, type=str, help='path to E-W velocity map (m/yr) or m/day')
    parser.add_argument('-vy_fn', default=None, required=True, type=str, help='path to N-S velocity map (m/yr) or m/day')
    #parser.add_argument('-H_fn', default=None, required=True, type=str, help='path to Farinotti ice-thickness grid')
    thickness_choices = ['farinotti','milan']
    parser.add_argument('-H_ver',default='farinotti', choices=thickness_choices, type=str, 
        help='wether to use farinotti or milan thickness values (default: %(default)s)')
    glacier_choices = ['ngozumpa','khumbu','changri_nup','imja','langtang','lirung','cn_validation']
    parser.add_argument('-smb_uncertainty_fn',type=str,default=None,
                        help='path to user-provided smb uncertainty file, will not compute these stats in not provided (default: %(default)s)')
    parser.add_argument('-glac_identifier', default=None, required=True, type=str, choices=glacier_choices, help='name of glacier')
    parser.add_argument('-lengthscale_factor', default=5, type=int, 
        help='Factor to multiply ice thickness values with for computing lengthscale (default: %(default)s)')
    timescale_opts = ['day','year']
    parser.add_argument('-timescale',type=str,default='year',choices=timescale_opts, help='Computation is in m/yr or m/day, (default: %(default)s)')
    parser.add_argument('-num_thickness_divison', default=20, type=int, 
        help='number of bins to divide the thickness values in (used for adaptive thickness gaussian filtering) (default: %(default)s)')
    parser.add_argument('-icecliff_gpkg',type=str,default=None,help='Path to user-provided ice cliff location file, will compute one if not provided (default: %(default)s)')
    parser.add_argument('-smr_cutoff',default=135.0,type=float,
        help='SMR cutoff to delineate melting ice cliffs and ponds (default: %(default)s)')
    parser.add_argument('-writeout',action='store_true',help='Writeout computed maps if invoked')
    parser.add_argument('-saveplot', action='store_true',  help='Writeout compute figures if invoked')
    parser.add_argument('-outdir', default=None, type=str, help='path to output directory where to store results\ndefaults to {DEM_timestamps}_lag_smb_results')
    parser.add_argument('-conserve_mass',action='store_true',help='do not use if full glacier is covered')
    return parser

def main():
    parser = getparser()
    args = parser.parse_args()
    
 
    print('\n%s' % datetime.now())
    print('%s UTC\n' % datetime.utcnow())
    if args.H_ver == 'farinotti':
        H_fn = constants.fetch_farinotti_thickness(args.glac_identifier)
        out_identifier = f"{args.glac_identifier}_farinotti"
    else:
        H_fn = constants.fetch_milan_thickness(args.glac_identifier)
        out_identifier = f"{args.glac_identifier}_millan"
    debris_thick_fn = constants.fetch_rounce_debris_thickness(args.glac_identifier)
    debris_melt_enhancement_fn = constants.fetch_rounce_debris_melt_enhancement(args.glac_identifier)
    glac_shp = gpd.read_file(constants.fetch_glac_shp(constants.rgi_dicts[args.glac_identifier])).to_crs('EPSG:32645')
    divQ2, euldhdt, lag_dhdt, downslope_dhdt, smb_dhdt,stats_df = glac_dyn.lag_smb_workflow(args.dem1_fn,args.dem2_fn,args.vx_fn,
                                                             args.vy_fn,H_fn,debris_thick_fn,debris_melt_enhancement_fn,glac_shp,out_identifier, args.lengthscale_factor,
                                                             args.num_thickness_divison,args.smr_cutoff,args.timescale,args.icecliff_gpkg,
                                                             args.writeout,args.saveplot,args.outdir,conserve_mass=args.conserve_mass,
                                                             smb_uncertainty_fn=args.smb_uncertainty_fn)
     

    print('\n%s' % datetime.now())
    print('%s UTC\n' % datetime.utcnow())

    print ("Script is complete!")
    

if __name__ == "__main__":
    main()


