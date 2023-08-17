Please follow the steps below to reproduce the results outlined in our study.
- If you have access to the WorldView source imagery, DEMs should be produced using settings described in the accompanying manuscript, which are inspired from [Bhushan and Shean, 2021](https://zenodo.org/record/4554647).
- -Coregister the DEMs using this two step procedure:
- 	Mask out glacier surfaces from the reference DEM: `dem_mask.py refdem.tif --glaciers`
- 	Co-register: `pc_align --max-displacement 30 --highest-accuracy --point-to-plane --save-transformed-source-points refdem_mask.tif source_dem.tif>
- 	Poin2dem
-  Generate hillshades using the gdaldem utility, with the following command: `gdaldem hillshade -combined -compute_edges <input_dem.tif> <output_hs.tif>`
-  Generate glacier velocity maps using the command: `disp_mgm_corr.py <hs1.tif> <hs2.tif> -dt yr -skip_rate 1`
-  If there are datagaps, fill them using the gaussian filling approach, as performed in the velocity_cleanup notebooks.
-  Analyse the products in QGIS, find out blunders and make polygons around them, save it to disk. Use the polygons to remove those erroneous patches using the command:
-  `parallel "clip_raster_shp.py {} polygon.geojson -invert" ::: <gap_filled_vx.tif> <gap_filled_vy.tif>`. The polygons we used to remove blunders in our manuscript are provided at:{TODO].
-  Compute SMB using the command: `compute_glacier_smb.py -dem1_fn <dem1.tif> -dem2_fn <dem2.tif> -vx_fn <vx_filled_filltered.tif> -vy_fn <vy_filled_filtered.tif> -glac_identifier <changri_nup> -timescale year -num_thickness_divison 20 -writeout -saveplot`
-  In the output folder, you will find a shapefile containing the potential location of ablation hotspots, open these up in QGIS with context orthoimagery, and add/remove polygons as you see fit.
-  Once happy that the shapefile is good to go, rerun the earlier command with an additional flag `-icecliff_gpkg final_icecliff_locations.gpkg`.
-  Analyse the resulting maps and draw figures as described in the acoompanying notebooks.  
-   
   
