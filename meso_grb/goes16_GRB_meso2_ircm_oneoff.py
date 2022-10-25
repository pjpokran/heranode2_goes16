#!/home/poker/miniconda3/envs/goes16_201710/bin/python

import netCDF4
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os,errno
#import os.rename
#import os.remove
import shutil
import sys
from PIL import Image

aoslogo = Image.open('/home/poker/uw-aoslogo.png')
aoslogoheight = aoslogo.size[1]
aoslogowidth = aoslogo.size[0]

# We need a float array between 0-1, rather than
# a uint8 array between 0-255
aoslogo = np.array(aoslogo).astype(np.float) / 255


dt = sys.argv[1]

f = netCDF4.Dataset(dt)

#print(f)

#print("f.variables[Rad] ",f.variables["Rad"])

#print("valid_pixel_count ",f.variables['valid_pixel_count'][0])
#print("undersaturated_pixel_count ",f.variables['undersaturated_pixel_count'][0])
#print("saturated_pixel_count ",f.variables['saturated_pixel_count'][0])
#print("min_radiance_value_of_valid_pixels ",f.variables['min_radiance_value_of_valid_pixels'][0])
#print("max_radiance_value_of_valid_pixels ",f.variables['max_radiance_value_of_valid_pixels'][0])
#print("mean_radiance_value_of_valid_pixels ",f.variables['mean_radiance_value_of_valid_pixels'][0])
#print("std_dev_radiance_value_of_valid_pixels ",f.variables['std_dev_radiance_value_of_valid_pixels'][0])
#print("nominal_satellite_height ",f.variables['nominal_satellite_height'][0])
#print("geospatial_lat_lon_extent ",f.variables['geospatial_lat_lon_extent'][0])
#print("goes_imager_projection ",f.variables['goes_imager_projection'][0])
#print("x_image ",f.variables['x_image'][0])
#print("x_image_bounds ",f.variables['x_image_bounds'][0])

# Unique x value for determining when scene changes
sunique = str(f.variables['x_image_bounds'][0]) + '_' + str(f.variables['y_image_bounds'][0])

#print("y_image ",f.variables['y_image'][0])
#print("y_image_bounds ",f.variables['y_image_bounds'][0])
#print("band_id ",f.variables['band_id'][0])
#print("band_wavelength ",f.variables['band_wavelength'][0])
#print("esun ",f.variables['esun'][0])
#print("kappa0 ",f.variables['kappa0'][0])
#print("earth_sun_distance_anomaly_in_AU ",f.variables['earth_sun_distance_anomaly_in_AU'][0])
#print(" ")



## Convert radiance to Brightness Temp 
fk1 = f.variables['planck_fk1'][0]
fk2 = f.variables['planck_fk2'][0]
bc1 = f.variables['planck_bc1'][0]
bc2 = f.variables['planck_bc2'][0]
#print("fk1 = ",fk1)
#print("fk2 = ",fk2)
#print("bc1 = ",bc1)
#print("bc2 = ",bc2)
#print(" ")
#print("scale factor is ",f.variables['Rad'].scale_factor)
#print("add_offset   is ",f.variables['Rad'].add_offset)
scalefactor = f.variables['Rad'].scale_factor
add_offset = f.variables['Rad'].add_offset
#data_var = f.variables['Rad'][0000:12000,0000:12000]
#data_var = f.variables['Rad'][:] *scalefactor + add_offset
data_var = f.variables['Rad'][:]
# Convert radiance to reflectance factor for Vis/NearIR (1-6)
# data_var = data_var * f.variables['kappa0'][0]
# reset values <0 to 0
# data_var[data_var < 0.] = 0.


# Convert Radiance to Brightness Temp w/planck function for IR (7-16)
#  T = [ fk2 / (alog((fk1 / Lλ) + 1)) - bc1 ] / bc2
data_var = (fk2 / ( np.log((fk1 / data_var) + 1 )) - bc1) / bc2

#print("max of data_var ",np.max(data_var))
#print("min of data_var ",np.min(data_var))

a = data_var
# a = 0.

xa = f.variables['x'][:]
ya = f.variables['y'][:]


#print("satellite height", f.variables['nominal_satellite_height'][0])
xa = xa*35785831
ya = ya*35785831


## Apply hybrid sqrt/linear scaling (0 < a < 0.91, b = sqrt(a); 0.91 <= 
##              0.91 <= a <= 1.16 linear e.g. b = sqrt(0.91) + (1. - (1.16 - a)/(1.16-0.91))*(1.-sqt(0.91))
##              a > 1.16, b=1
##                    = np.sqrt(a where a < 0.91)
##f.variables['planck_bc1'][0]
#print("y ",f.dimensions["y"].size)
#print("x ",f.dimensions["x"].size)
image_rows=f.dimensions["y"].size
image_columns=f.dimensions["x"].size


#condition1=np.less(a,0.91)
#condition2=np.greater_equal(a,0.91)
#
#a =(condition1*np.sqrt(data_var)) + (condition2*(0.9539 + (1.-(1.16-data_var)/0.25)*0.0461))
#print("max of a",np.max(a))
#print("min of a ",np.min(a))

#a = a/1.16

#a[a>1.]=1.

#print("max of a",np.max(a))
#print("min of a ",np.min(a))

#asub = a[::2,::2]
#print("asub ",asub)
#xasub = xa[::2]
#yasub = ya[::2]
#
#def rebin(a,shape):
#    sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]
#    return a.reshape(sh).mean(-1).mean(1)
#
#height, width = a.shape
#print(height,width)
#N1=2
#N2=2
#asub2 = np.average(np.split(np.average(np.split(a, width // N2, axis=1), axis=-1), height // N1, axis=1), axis=-1)
##xasub2 = np.average(np.split(xa,height//N2,axis=1))
##yasub2 = np.average(np.split(ya,width//N1,axis=1))
##xasub2 = 



#quit()


#print(np.average(a))
#if np.average(a) > 0.75:
#    quit()

#if np.average(a) < 0.025:
#    quit()

import cartopy.crs as ccrs

# Create a Globe specifying a spherical earth with the correct radius
#globe = ccrs.Globe(semimajor_axis=proj_var.semi_major,
#                   semiminor_axis=proj_var.semi_minor)
globe = ccrs.Globe(semimajor_axis=6378137.,semiminor_axis=6356752.)

#proj = ccrs.LambertConformal(central_longitude=proj_var.longitude_of_central_meridian,
#                             central_latitude=proj_var.latitude_of_projection_origin,
#                             standard_parallels=[proj_var.standard_parallel],
#                             globe=globe)

proj = ccrs.Geostationary(central_longitude=-75.0, 
                          satellite_height=35785831, 
                          globe=globe,sweep_axis='x')

# Define proj2 to be the lambert conformal projection that NOAAPORT CONUS was in at first
proj2 = ccrs.LambertConformal(central_longitude=-95.0,
                             central_latitude=25.0,
                             standard_parallels=(25,25),
                             globe=globe)


##########################################################
## HACK for elipse and mask being drawn inside of the actual edge of the earth
#override_ellipse(proj)
##########################################################

#wi_image_crop_top=0
#wi_image_crop_bottom=-4200
#wi_image_crop_left=4060
#wi_image_crop_right=-3680
#
#wi_image_size_y=(image_rows+wi_image_crop_bottom-wi_image_crop_top)
#wi_image_size_x=(image_columns+wi_image_crop_right-wi_image_crop_left)
#
#print("wi image size")
#print(wi_image_size_x, wi_image_size_y)
#
##wi_image_size_x=float(wi_image_size_x)/120.
##wi_image_size_y=float(wi_image_size_y)/120.
#wi_image_size_x=float(wi_image_size_x)/160.
#wi_image_size_y=float(wi_image_size_y)/160.
#
#mw_image_crop_top=0
#mw_image_crop_bottom=-3816
#mw_image_crop_left=2550
#mw_image_crop_right=-3001
#
#mw_image_size_y=(image_rows+mw_image_crop_bottom-mw_image_crop_top)
#mw_image_size_x=(image_columns+mw_image_crop_right-mw_image_crop_left)
#
#print("mw image size")
#print(mw_image_size_x, mw_image_size_y)
#mw_image_size_x=float(mw_image_size_x)/212.
#mw_image_size_y=float(mw_image_size_y)/212.
#
#conus_image_crop_top=0
#conus_image_crop_bottom=-505
#conus_image_crop_left=0
#conus_image_crop_right=-6
#
#conus_image_size_y=(int(image_rows/2)+conus_image_crop_bottom-conus_image_crop_top)
#conus_image_size_x=(int(image_columns/2)+conus_image_crop_right-conus_image_crop_left)
#
#print("conus image size")
#print(conus_image_size_x, conus_image_size_y)
#
#conus_image_size_x=float(conus_image_size_x)/250.
#conus_image_size_y=float(conus_image_size_y)/250.
#
## Northeast sector
#ne_image_crop_top=0
#ne_image_crop_bottom=-2609
#ne_image_crop_left=4300
#ne_image_crop_right=-2
#
#ne_image_size_y=(image_rows+ne_image_crop_bottom-ne_image_crop_top)
#ne_image_size_x=(image_columns+ne_image_crop_right-ne_image_crop_left)
#print("ne image size")
#print(ne_image_size_x, ne_image_size_y)
#
#ne_image_size_x=float(ne_image_size_x)/280.
#ne_image_size_y=float(ne_image_size_y)/280.
#
## Fullres Southern WI sector
#
##swi_image_crop_top=800
##swi_image_crop_bottom=-4700
##swi_image_crop_left=5060
##swi_image_crop_right=-4080
#
#swi_image_crop_top=450
#swi_image_crop_bottom=-4802
#swi_image_crop_left=4560
#swi_image_crop_right=-3938
#
#swi_image_size_y=(image_rows+swi_image_crop_bottom-swi_image_crop_top)
#swi_image_size_x=(image_columns+swi_image_crop_right-swi_image_crop_left)
#print("swi image size")
#print(swi_image_size_x, swi_image_size_y)
#
#swi_image_size_x=float(swi_image_size_x)/66.
#swi_image_size_y=float(swi_image_size_y)/66.
#
## Fullres Colorado sector
#
#co_image_crop_top=800
#co_image_crop_bottom=-4103
#co_image_crop_left=1800
#co_image_crop_right=-6290
#
#co_image_size_y=(image_rows+co_image_crop_bottom-co_image_crop_top)
#co_image_size_x=(image_columns+co_image_crop_right-co_image_crop_left)
#print("co image size")
#print(co_image_size_x, co_image_size_y)
#
#co_image_size_x=float(co_image_size_x)/60.
#co_image_size_y=float(co_image_size_y)/60.
#
## Fullres Florida sector
#
#fl_image_crop_top=2550
#fl_image_crop_bottom=-1920
#fl_image_crop_left=5500
#fl_image_crop_right=-2670
#
#fl_image_size_y=(image_rows+fl_image_crop_bottom-fl_image_crop_top)
#fl_image_size_x=(image_columns+fl_image_crop_right-fl_image_crop_left)
#print("fl image size")
#print(fl_image_size_x, fl_image_size_y)
#
#fl_image_size_x=float(fl_image_size_x)/78.
#fl_image_size_y=float(fl_image_size_y)/78.
#
## Gulf of Mexico sector
#
#gulf_image_crop_top=1950
#gulf_image_crop_bottom=-3
#gulf_image_crop_left=3500
#gulf_image_crop_right=-1
#
#gulf_image_size_y=(image_rows+gulf_image_crop_bottom-gulf_image_crop_top)
#gulf_image_size_x=(image_columns+gulf_image_crop_right-gulf_image_crop_left)
#print("gulf image size")
#print(gulf_image_size_x, gulf_image_size_y)
#
#gulf_image_size_x=float(gulf_image_size_x)/240.
#gulf_image_size_y=float(gulf_image_size_y)/240.
#
## Great Lakes sector
#greatlakes_image_crop_top=0
#greatlakes_image_crop_bottom=-4411
#greatlakes_image_crop_left=4400
#greatlakes_image_crop_right=-1603
#
#greatlakes_image_size_y=(image_rows+greatlakes_image_crop_bottom-greatlakes_image_crop_top)
#greatlakes_image_size_x=(image_columns+greatlakes_image_crop_right-greatlakes_image_crop_left)
#print("ne image size")
#print(greatlakes_image_size_x, greatlakes_image_size_y)
#
#greatlakes_image_size_x=float(greatlakes_image_size_x)/140.
#greatlakes_image_size_y=float(greatlakes_image_size_y)/140.
#
## Alex special sector
##alex_image_crop_top=3300
##alex_image_crop_bottom=-4800
##alex_image_crop_left=4760
##alex_image_crop_right=-6180
##
##alex_image_size_y=(image_rows+alex_image_crop_bottom-alex_image_crop_top)
##alex_image_size_x=(image_columns+alex_image_crop_right-alex_image_crop_left)
##
##alex_image_size_x=float(alex_image_size_x)/65.
##alex_image_size_y=float(alex_image_size_y)/65.
#
#
# Create a new figure with size 10" by 10"
## WI crop
## fig = plt.figure(figsize=(wi_image_size_x,wi_image_size_y),dpi=80.)
#fig = plt.figure(figsize=(wi_image_size_x,wi_image_size_y),dpi=78.)
## Midwest crop
#fig2 = plt.figure(figsize=(mw_image_size_x,mw_image_size_y),dpi=78.)
## CONUS crop
## fig3 = plt.figure(figsize=(conus_image_size_x,conus_image_size_y),dpi=160.)
##fig3 = plt.figure(figsize=(conus_image_size_x,conus_image_size_y),dpi=78.)
#fig3 = plt.figure(figsize=(21.50,14.00),dpi=78.)
## Northeast crop
#fig4 = plt.figure(figsize=(ne_image_size_x,ne_image_size_y),dpi=78.)
## Wisconsin fullres crop
#fig5 = plt.figure(figsize=(swi_image_size_x,swi_image_size_y),dpi=78.)
## Colorado fullres crop
#fig6 = plt.figure(figsize=(co_image_size_x,co_image_size_y),dpi=78.)
## Florida fullres crop
#fig7 = plt.figure(figsize=(fl_image_size_x,fl_image_size_y),dpi=78.)
## Gulf of Mexico region
#fig8 = plt.figure(figsize=(gulf_image_size_x,gulf_image_size_y),dpi=78.)
# Full res
#fig9 = plt.figure(figsize=(image_columns/78.,image_rows/78.))
fig9 = plt.figure(figsize=(14.,14.))
# Alex stormchase crop
#fig10 = plt.figure(figsize=(alex_image_size_x,alex_image_size_y),dpi=83.)
# Great Lakes
#fig11 = plt.figure(figsize=(greatlakes_image_size_x,greatlakes_image_size_y),dpi=78.)
# Wisconsin fullres crop

# Put a single axes on this figure; set the projection for the axes to be our
# Lambert conformal projection
#ax = fig.add_subplot(1, 1, 1, projection=proj2)
#ax.set_extent((-96.9,-83.7,40.8,48.4))
#ax.gridlines()
#ax2 = fig2.add_subplot(1, 1, 1, projection=proj2)
#ax2.set_extent((-105.8,-81.2,35.0,48.7))
#ax2.gridlines()
##ax3 = fig3.add_subplot(1, 1, 1, projection=proj2, axisbg='None')
##ax3 = fig3.add_subplot(1, 1, 1, projection=proj2)
#ax3 = fig3.add_subplot(1, 1, 1, projection=proj)
##ax3.set_extent((-120.8,-67.0,19.0,50.0))
##ax3.patch.set_facecolor('black')
#ax3.background_patch.set_fill(False)
#ax3.outline_patch.set_edgecolor('none')
#ax3.gridlines()
#ax4 = fig4.add_subplot(1, 1, 1, projection=proj2)
#ax4.set_extent((-92.1,-59.5,32.4,50.3))
#ax4.background_patch.set_fill(False)
#ax4.gridlines()
##ax5 = fig5.add_subplot(1, 1, 1, projection=proj)
#ax5 = fig5.add_subplot(1, 1, 1, projection=proj2)
#ax5.set_extent((-93.0,-86.05,41.0,45.0))
#ax5.gridlines()
#ax6 = fig6.add_subplot(1, 1, 1, projection=proj2)
#ax6.set_extent((-109.3,-101.5,36.6,41.6))
#ax6.gridlines()
#ax7 = fig7.add_subplot(1, 1, 1, projection=proj)
#ax8 = fig8.add_subplot(1, 1, 1, projection=proj)
ax9 = fig9.add_subplot(1, 1, 1, projection=proj)
##ax10 = fig10.add_subplot(1, 1, 1, projection=proj)
#ax11 = fig11.add_subplot(1, 1, 1, projection=proj2)
#ax11.set_extent((-94.5,-73.8,40.0,49.2))
#ax11.background_patch.set_fill(False)
#ax11.gridlines()

# McIDAS IR Color Table

cdict2 = {'red': ((0.0, 0.0, 0.0),
                 (.001, 1.00, 1.00),
                 (.107, 1.00, 1.00),
                 (.113, 0.498, 0.498),
                 (.173, 1.00, 1.00),
                 (.179, 0.902, 0.902),
                 (.227, 0.102, 0.102),
                 (.233, 0.00, 0.00),
                 (.287, 0.902, 0.902),
                 (.293, 1.00, 1.00),
                 (.346, 1.00, 1.00),
                 (.352, 1.00, 1.00),
                 (.406, 0.101, 0.101),
                 (.412, 0.00, 0.00),
                 (.481, 0.00, 0.00),
                 (.484, 0.00, 0.00),
                 (.543, 0.00, 0.00),
                 (.546, 0.773, 0.773),
                 (.994, 0.012, 0.012),
                 (.997, 0.004, 0.004),
                 (1.0, 0.0, 0.0)),
         'green': ((0.0, 0.0, 0.0),
                 (.001, 1.00, 1.00),
                 (.107, 1.00, 1.00),
                 (.113, 0.00, 0.00),
                 (.173, 0.498, 0.498),
                 (.179, 0.902, 0.902),
                 (.227, 0.102, 0.102),
                 (.233, 0.00, 0.00),
                 (.287, 0.00, 0.00),
                 (.293, 0.00, 0.00),
                 (.346, 0.902, 0.902),
                 (.352, 1.00, 1.00),
                 (.406, 1.00, 1.00),
                 (.412, 1.00, 1.00),
                 (.481, 0.00, 0.00),
                 (.484, 0.00, 0.00),
                 (.543, 1.00, 1.00),
                 (.546, 0.773, 0.773),
                 (.994, 0.012, 0.012),
                 (.997, 0.004, 0.004),
                   (1.0, 0.0, 0.0)),
         'blue': ((0.0, 0.00, 0.00),
                 (.001, 1.00, 1.00),
                 (.107, 0.00, 0.00),
                 (.113, 0.498, 0.498),
                 (.173, 0.786, 0.786),
                 (.179, 0.902, 0.902),
                 (.227, 0.102, 0.102),
                 (.233, 0.00, 0.00),
                 (.287, 0.00, 0.00),
                 (.293, 0.00, 0.00),
                 (.346, 0.00, 0.00),
                 (.352, 0.00, 0.00),
                 (.406, 0.00, 0.00),
                 (.412, 0.00, 0.00),
                 (.481, 0.451, 0.451),
                 (.484, 0.451, 0.451),
                 (.543, 1.00, 1.00),
                 (.546, 0.773, 0.773),
                 (.994, 0.012, 0.012),
                 (.997, 0.004, 0.004),
                  (1.0, 0.0, 0.0))}

import matplotlib as mpl
my_cmap = mpl.colors.LinearSegmentedColormap('my_colormap',cdict2,2048)


# Plot the data
# set the colormap to extend over a range of values from 162 to 330 using the
# my_cmap that we defined.

#im = ax.imshow(a[wi_image_crop_top:wi_image_crop_bottom,wi_image_crop_left:wi_image_crop_right], extent=(xa[wi_image_crop_left],xa[wi_image_crop_right],ya[wi_image_crop_bottom],ya[wi_image_crop_top]), origin='upper',cmap='Greys_r', vmin=0., vmax=1., transform=proj)
#im = ax2.imshow(a[mw_image_crop_top:mw_image_crop_bottom,mw_image_crop_left:mw_image_crop_right], extent=(xa[mw_image_crop_left],xa[mw_image_crop_right],ya[mw_image_crop_bottom],ya[mw_image_crop_top]), origin='upper',cmap='Greys_r', vmin=0., vmax=1., transform=proj)
##im = ax3.imshow(asub[conus_image_crop_top:conus_image_crop_bottom,conus_image_crop_left:conus_image_crop_right], extent=(xasub[conus_image_crop_left],xasub[conus_image_crop_right],yasub[conus_image_crop_bottom],yasub[conus_image_crop_top]), origin='upper',cmap='Greys_r', vmin=0., vmax=1., transform=proj)
##im = ax3.imshow(a[conus_image_crop_top:conus_image_crop_bottom,conus_image_crop_left:conus_image_crop_right], extent=(xa[conus_image_crop_left],xa[conus_image_crop_right],ya[conus_image_crop_bottom],ya[conus_image_crop_top]), origin='upper',cmap='Greys_r', vmin=0., vmax=1., transform=proj)
#im = ax3.imshow(a[conus_image_crop_top:conus_image_crop_bottom,conus_image_crop_left:conus_image_crop_right], extent=(xa[conus_image_crop_left],xa[conus_image_crop_right],ya[conus_image_crop_bottom],ya[conus_image_crop_top]), origin='upper',cmap='Greys_r', vmin=0., vmax=1.)
#im = ax4.imshow(a[ne_image_crop_top:ne_image_crop_bottom,ne_image_crop_left:ne_image_crop_right], extent=(xa[ne_image_crop_left],xa[ne_image_crop_right],ya[ne_image_crop_bottom],ya[ne_image_crop_top]), origin='upper',cmap='Greys_r', vmin=0., vmax=1., transform=proj)
#im = ax5.imshow(a[swi_image_crop_top:swi_image_crop_bottom,swi_image_crop_left:swi_image_crop_right], extent=(xa[swi_image_crop_left],xa[swi_image_crop_right],ya[swi_image_crop_bottom],ya[swi_image_crop_top]), origin='upper',cmap='Greys_r', vmin=0., vmax=1., transform=proj)
#im = ax6.imshow(a[co_image_crop_top:co_image_crop_bottom,co_image_crop_left:co_image_crop_right], extent=(xa[co_image_crop_left],xa[co_image_crop_right],ya[co_image_crop_bottom],ya[co_image_crop_top]), origin='upper',cmap='Greys_r', vmin=0., vmax=1., transform=proj)
#im = ax7.imshow(a[fl_image_crop_top:fl_image_crop_bottom,fl_image_crop_left:fl_image_crop_right], extent=(xa[fl_image_crop_left],xa[fl_image_crop_right],ya[fl_image_crop_bottom],ya[fl_image_crop_top]), origin='upper',cmap='Greys_r', vmin=0., vmax=1.)
#im = ax8.imshow(a[gulf_image_crop_top:gulf_image_crop_bottom,gulf_image_crop_left:gulf_image_crop_right], extent=(xa[gulf_image_crop_left],xa[gulf_image_crop_right],ya[gulf_image_crop_bottom],ya[gulf_image_crop_top]), origin='upper',cmap='Greys_r', vmin=0., vmax=1.)
im = ax9.imshow(a[:], extent=(xa[1],xa[-1],ya[-1],ya[1]), origin='upper', cmap=my_cmap, vmin=162., vmax=330.)
##im = ax10.imshow(a[alex_image_crop_top:alex_image_crop_bottom,alex_image_crop_left:alex_image_crop_right], extent=(xa[alex_image_crop_left],xa[alex_image_crop_right],ya[alex_image_crop_bottom],ya[alex_image_crop_top]), origin='upper',cmap='Greys_r', vmin=0., vmax=1.)

#im = ax11.imshow(a[greatlakes_image_crop_top:greatlakes_image_crop_bottom,greatlakes_image_crop_left:greatlakes_image_crop_right], extent=(xa[greatlakes_image_crop_left],xa[greatlakes_image_crop_right],ya[greatlakes_image_crop_bottom],ya[greatlakes_image_crop_top]), origin='upper',cmap='Greys_r', vmin=0., vmax=1., transform=proj)

import cartopy.feature as cfeat
from cartopy.io.shapereader import Reader
fname = '/home/poker/resources/counties.shp'
fname = '/home/poker/resources/cb_2016_us_county_5m.shp'
counties = Reader(fname)
#ax5.add_geometries(counties.geometries(), ccrs.PlateCarree(), edgecolor='darkgreen', facecolor='None')
#ax6.add_geometries(counties.geometries(), ccrs.PlateCarree(), edgecolor='darkgreen', facecolor='None')
#ax7.add_geometries(counties.geometries(), ccrs.PlateCarree(), edgecolor='darkgreen', facecolor='None')




# Set up a feature for the state/province lines. Tell cartopy not to fill in the polygons

state_boundaries = cfeat.NaturalEarthFeature(category='cultural',
                                             name='admin_1_states_provinces_lakes',
                                             scale='50m', facecolor='none', edgecolor='red')

state_boundaries2 = cfeat.NaturalEarthFeature(category='cultural',
                                             name='admin_1_states_provinces_lakes',
                                             scale='10m', facecolor='none', edgecolor='red')

# Add the feature with dotted lines, denoted by ':'
#ax.add_feature(state_boundaries, linewidth=2, linestyle=':', edgecolor='magenta')
#ax2.add_feature(state_boundaries, linewidth=2, linestyle=':', edgecolor='magenta')
#ax3.add_feature(state_boundaries, linewidth=2, linestyle=':', edgecolor='magenta')
#ax4.add_feature(state_boundaries, linewidth=2, linestyle=':', edgecolor='magenta')
#ax5.add_feature(state_boundaries2, linewidth=2, linestyle=':', edgecolor='magenta')
#ax6.add_feature(state_boundaries2, linewidth=2, linestyle=':', edgecolor='magenta')
#ax7.add_feature(state_boundaries2, linewidth=2, linestyle=':', edgecolor='magenta')
#ax8.add_feature(state_boundaries, linewidth=2, linestyle=':', edgecolor='magenta')
ax9.add_feature(state_boundaries, linestyle=':',linewidth=2, edgecolor='magenta')
###ax10.add_feature(state_boundaries, linestyle=':', linestyle=':', edgecolor='magenta')
#ax11.add_feature(state_boundaries, linewidth=2, linestyle=':', edgecolor='magenta')


#ax.coastlines(resolution='50m', color='green')
#ax2.coastlines(resolution='50m', color='green')
#ax3.coastlines(resolution='50m', color='green')
#ax4.coastlines(resolution='50m', color='green')
#ax5.coastlines(resolution='50m', color='green')
#ax6.coastlines(resolution='50m', color='green')
#ax7.coastlines(resolution='50m', color='green')
#ax8.coastlines(resolution='50m', color='green')
ax9.coastlines(resolution='10m', color='green')
##ax10.coastlines(resolution='50m', color='green')
#ax11.coastlines(resolution='50m', color='green')

# Add country borders with a thick line.
#ax.add_feature(cfeat.BORDERS, linewidth='1', edgecolor='green')
#ax2.add_feature(cfeat.BORDERS, linewidth='1', edgecolor='green')
#ax3.add_feature(cfeat.BORDERS, linewidth='1', edgecolor='green')
#ax4.add_feature(cfeat.BORDERS, linewidth='1', edgecolor='green')
#ax5.add_feature(cfeat.BORDERS, linewidth='1', edgecolor='green')
#ax6.add_feature(cfeat.BORDERS, linewidth='1', edgecolor='green')
#ax7.add_feature(cfeat.BORDERS, linewidth='1', edgecolor='green')
#ax8.add_feature(cfeat.BORDERS, linewidth='1', edgecolor='green')
ax9.add_feature(cfeat.BORDERS, linewidth='1', edgecolor='green')
###ax10.add_feature(cfeat.BORDERS, linewidth='1', edgecolor='green')
#ax11.add_feature(cfeat.BORDERS, linewidth='1', edgecolor='green')


from matplotlib import patheffects
outline_effect = [patheffects.withStroke(linewidth=2, foreground='black')]

#ax5.plot(-89.4012, 43.0731, 'bo', markersize=3, transform=ccrs.Geodetic())
#stn5 = ax5.text(-89.50, 43.02, 'MSN', transform=ccrs.Geodetic(), color='darkorange')
#stn5.set_path_effects(outline_effect)
#ax5.plot(-87.9065, 43.0389, 'bo', markersize=3, transform=ccrs.Geodetic())
#stn5 = ax5.text(-88.00, 42.98, 'MKE', transform=ccrs.Geodetic(), color='darkorange')
#stn5.set_path_effects(outline_effect)
#ax5.plot(-91.2396, 43.8014, 'bo', markersize=3, transform=ccrs.Geodetic())
#stn5 = ax5.text(-91.33, 43.75, 'LSE', transform=ccrs.Geodetic(), color='darkorange')
#stn5.set_path_effects(outline_effect)
#ax5.plot(-88.0198, 44.5192, 'bo', markersize=3, transform=ccrs.Geodetic())
#stn5 = ax5.text(-88.11, 44.46, 'GRB', transform=ccrs.Geodetic(), color='darkorange')
#stn5.set_path_effects(outline_effect)
#ax5.plot(-87.9073, 41.9742, 'bo', markersize=3, transform=ccrs.Geodetic())
#stn5 = ax5.text(-88.00, 41.82, 'ORD', transform=ccrs.Geodetic(), color='darkorange')
#stn5.set_path_effects(outline_effect)
#ax5.plot(-90.5083, 41.4496, 'bo', markersize=3, transform=ccrs.Geodetic())
#stn5 = ax5.text(-90.60, 41.30, 'MLI', transform=ccrs.Geodetic(), color='darkorange')
#stn5.set_path_effects(outline_effect)
##
#ax6.plot(-104.9903, 39.7392, 'bo', markersize=3, transform=ccrs.Geodetic())
#stn6 = ax6.text(-105.09, 39.68, 'DEN', transform=ccrs.Geodetic(), color='darkorange')
#stn6.set_path_effects(outline_effect)
#ax6.plot(-105.2705, 40.0150, 'bo', markersize=3, transform=ccrs.Geodetic())
#stn6 = ax6.text(-105.37, 39.96, 'BOU', transform=ccrs.Geodetic(), color='darkorange')
#stn6.set_path_effects(outline_effect)
#ax6.plot(-105.0844, 40.5853, 'bo', markersize=3, transform=ccrs.Geodetic())
#stn6 = ax6.text(-105.18, 40.53, 'FNL', transform=ccrs.Geodetic(), color='darkorange')
#stn6.set_path_effects(outline_effect)
#ax6.plot(-108.5506, 39.0639, 'bo', markersize=3, transform=ccrs.Geodetic())
#stn6 = ax6.text(-108.65, 39.01, 'GJT', transform=ccrs.Geodetic(), color='darkorange')
#stn6.set_path_effects(outline_effect)
#ax6.plot(-104.8214, 38.8339, 'bo', markersize=3, transform=ccrs.Geodetic())
#stn6 = ax6.text(-104.92, 38.78, 'COS', transform=ccrs.Geodetic(), color='darkorange')
#stn6.set_path_effects(outline_effect)
#ax6.plot(-104.6091, 38.2544, 'bo', markersize=3, transform=ccrs.Geodetic())
#stn6 = ax6.text(-104.70, 38.20, 'PUB', transform=ccrs.Geodetic(), color='darkorange')
#stn6.set_path_effects(outline_effect)
#ax6.plot(-106.9253, 38.5458, 'bo', markersize=3, transform=ccrs.Geodetic())
#stn6 = ax6.text(-107.02, 38.40, 'GUC', transform=ccrs.Geodetic(), color='darkorange')
#stn6.set_path_effects(outline_effect)
##
##
#fig.figimage(aoslogo,   0, fig.bbox.ymax - aoslogoheight + 8, zorder=10)
#fig2.figimage(aoslogo,  0, fig2.bbox.ymax - aoslogoheight + 20  , zorder=10)
#fig3.figimage(aoslogo,  0, fig3.bbox.ymax - aoslogoheight + 24, zorder=10)
#fig4.figimage(aoslogo,  0, fig4.bbox.ymax - aoslogoheight + 24 , zorder=10)
#fig5.figimage(aoslogo,  0, fig5.bbox.ymax - aoslogoheight + 24  , zorder=10)
#fig6.figimage(aoslogo,  0, fig6.bbox.ymax - aoslogoheight + 36  , zorder=10)
#fig7.figimage(aoslogo,  10, fig7.bbox.ymax - aoslogoheight   , zorder=10)
#fig8.figimage(aoslogo,  10, fig8.bbox.ymax - aoslogoheight   , zorder=10)
fig9.figimage(aoslogo,  0, 0, zorder=10)
###fig10.figimage(aoslogo,  10, int(fig10.bbox.ymax*.96386) - aoslogoheight - 18  , zorder=10)
#fig11.figimage(aoslogo,  0, fig11.bbox.ymax - aoslogoheight + 24 , zorder=10)

# color bar
cbaxes9 = fig9.add_axes([0.255,0.12,0.635,0.02])
cbar9 = fig9.colorbar(im, cax=cbaxes9, orientation='horizontal')
font_size = 14
#cbar9.set_label('Brightness Temperature (K)',size=18)
cbar9.ax.tick_params(labelsize=font_size)
cbar9.ax.xaxis.set_ticks_position('top')
cbar9.ax.xaxis.set_label_position('top')

import datetime

#time_var = f.date_created
time_var = f.time_coverage_start

#jyr = time_var[0:4]
#jday = time_var[4:7]
#print(jday)
iyear = time_var[0:4]
print("iyear ",iyear)
imonth = time_var[5:7]
print("imonth ",imonth)
import calendar
cmonth = calendar.month_abbr[int(imonth)]
print("cmonth ",cmonth)
iday = time_var[8:10]
print("iday ",iday)
itime = time_var[11:19]
itimehr = time_var[11:13]
itimemn = time_var[14:16]
itimesc = time_var[17:19]

#date = datetime.datetime(int(jyr), 1, 1) + datetime.timedelta(int(jday)-1)

ctime_string = iyear +' '+cmonth+' '+iday+'  '+itime+' GMT'
ctime_file_string = iyear + imonth + iday + itimehr + itimemn + itimesc + "_" + sunique
list_string = sunique + '.jpg'

#time_string = 'GOES-16 Band 14 (LW IR) valid %s'%time_var
#time_string = 'GOES-16 Band 14 (LW IR) valid %s %s %s %s'%iyear %cmonth %iday %itime
#time_string = 'GOES-16 Band 2 Red Visible\n %s '%ctime_string
time_string = 'GOES-16 Clean IR LW window (ABI ch 13)\n %s '%ctime_string
print(time_string)


#text = ax.text(0.50, 0.97, time_string,
#    horizontalalignment='center', transform = ax.transAxes,
#    color='yellow', fontsize='large', weight='bold')
#
#text.set_path_effects(outline_effect)
#
#text2 = ax2.text(0.50, 0.97, time_string,
#    horizontalalignment='center', transform = ax2.transAxes,
#    color='yellow', fontsize='large', weight='bold')
#
#text2.set_path_effects(outline_effect)
#
#text3 = ax3.text(0.50, 0.95, time_string,
#    horizontalalignment='center', transform = ax3.transAxes,
#    color='darkorange', fontsize='large', weight='bold')
#
#text3.set_path_effects(outline_effect)
#
#text4 = ax4.text(0.50, 0.95, time_string,
#    horizontalalignment='center', transform = ax4.transAxes,
#    color='darkorange', fontsize='large', weight='bold')
#
#text4.set_path_effects(outline_effect)
#
#text5 = ax5.text(0.52, 0.95, time_string,
#    horizontalalignment='center', transform = ax5.transAxes,
#    color='darkorange', fontsize='large', weight='bold', zorder=12)
#
#text5.set_path_effects(outline_effect)
#
#text6 = ax6.text(0.50, 0.95, time_string,
#    horizontalalignment='center', transform = ax6.transAxes,
#    color='darkorange', fontsize='large', weight='bold')
#
#text6.set_path_effects(outline_effect)
#
#text7 = ax7.text(0.50, 0.95, time_string,
#    horizontalalignment='center', transform = ax7.transAxes,
#    color='darkorange', fontsize='large', weight='bold')
#
#text7.set_path_effects(outline_effect)
#
#text8 = ax8.text(0.50, 0.95, time_string,
#    horizontalalignment='center', transform = ax8.transAxes,
#    color='darkorange', fontsize='large', weight='bold')
#
#text8.set_path_effects(outline_effect)
##
text9 = ax9.text(0.50, 0.95, time_string,
    horizontalalignment='center', transform = ax9.transAxes,
    color='yellow', fontsize='large', weight='bold')

text9.set_path_effects(outline_effect)
#
##text10 = ax10.text(0.50, 0.95, time_string,
##    horizontalalignment='center', transform = ax10.transAxes,
##    color='darkorange', fontsize='large', weight='bold')
##
##text10.set_path_effects(outline_effect)

#text11 = ax11.text(0.50, 0.95, time_string,
#    horizontalalignment='center', transform = ax11.transAxes,
#    color='darkorange', fontsize='large', weight='bold')
#
#text11.set_path_effects(outline_effect)

#filename1="/whirlwind/goes16/grb/vis/wi/"+ctime_file_string+"_wi.jpg"
#filename2="/whirlwind/goes16/grb/vis/mw/"+ctime_file_string+"_mw.jpg"
#filename3="/whirlwind/goes16/grb/vis/conus/"+ctime_file_string+"_conus.jpg"
#filename4="/whirlwind/goes16/grb/vis/ne/"+ctime_file_string+"_ne.jpg"
#filename5="/whirlwind/goes16/grb/vis/swi/"+ctime_file_string+"_swi.jpg"
#filename6="/whirlwind/goes16/grb/vis/co/"+ctime_file_string+"_co.jpg"
#filename7="/whirlwind/goes16/grb/vis/fl/"+ctime_file_string+"_fl.jpg"
#filename8="/whirlwind/goes16/grb/vis/gulf/"+ctime_file_string+"_gulf.jpg"
#filename9="/whirlwind/goes16/grb/vis/gulf/"+ctime_file_string+"_gulf.jpg"
filename9="/whirlwind/goes16/grb/meso_ircm/"+ctime_file_string+".jpg"
#filename11="/whirlwind/goes16/grb/vis/greatlakes/"+ctime_file_string+"_greatlakes.jpg"
#filename9="/whirlwind/goes16/grb/vis/full/"+ctime_file_string+"_full.jpg"
#filename10="/whirlwind/goes16/grb/vis/alex/"+ctime_file_string+"_alex.jpg"
#filename1="/whirlwind/goes16/test/"+ctime_file_string+"_wi.jpg"
#filename2="/whirlwind/goes16/test/"+ctime_file_string+"_mw.jpg"
#filename3="/whirlwind/goes16/test/"+ctime_file_string+"_conus.jpg"
#filename4="/whirlwind/goes16/test/"+ctime_file_string+"_ne.jpg"
#filename5="/whirlwind/goes16/test/"+ctime_file_string+"_swi.jpg"
#filename6="/whirlwind/goes16/test/"+ctime_file_string+"_co.jpg"
#filename7="/whirlwind/goes16/test/"+ctime_file_string+"_fl.jpg"
#filename8="/whirlwind/goes16/test/"+ctime_file_string+"_gulf.jpg"
#filename9="/whirlwind/goes16/test/"+ctime_file_string+"_full.jpg"
#filename10="/whirlwind/goes16/test/"+ctime_file_string+"_alex.jpg"


#fig.savefig(filename1, bbox_inches='tight',pad_inches=0)
#fig2.savefig(filename2, bbox_inches='tight',pad_inches=0)
#fig3.savefig(filename3, bbox_inches='tight', transparent=True, facecolor='black', pad_inches=0)
#fig4.savefig(filename4, bbox_inches='tight', transparent=True, facecolor='black', pad_inches=0)
#fig5.savefig(filename5, bbox_inches='tight',pad_inches=0)
#fig6.savefig(filename6, bbox_inches='tight',pad_inches=0)
#fig7.savefig(filename7, bbox_inches='tight',pad_inches=0)
#fig8.savefig(filename8, bbox_inches='tight',pad_inches=0)
fig9.savefig(filename9, bbox_inches='tight',pad_inches=0)
#fig10.savefig(filename10, bbox_inches='tight',pad_inches=0)
#fig11.savefig(filename11, bbox_inches='tight', transparent=True, facecolor='black', pad_inches=0)

quit()

#import os.rename    # os.rename(src,dest)
#import os.remove    # os.remove path
#import shutil.copy  # shutil.copy(src, dest)


def silentremove(filename):
    try: 
        os.remove(filename)
    except OSError:
        pass
    
def silentrename(filename1, filename2):
    try: 
        os.rename(filename1, filename2)
    except OSError:
        pass


shutil.copy(filename9, "/whirlwind/goes16/grb/meso_ircm/latest_meso_2.jpg")

import glob
file_list = glob.glob('/whirlwind/goes16/grb/meso_ircm/*' + list_string)
file_list.sort()
#print("file list is ",file_list)

thefile = open('/whirlwind/goes16/grb/meso_ircm/meso_2_30_temp.list', 'w')
thelist = file_list[-30:]
#print ("thelist is ",thelist)

for item in thelist:
    head, tail = os.path.split(item)
    thefile.write(tail + '\n')
thefile.close
os.rename('/whirlwind/goes16/grb/meso_ircm/meso_2_30_temp.list','/whirlwind/goes16/grb/meso_ircm/meso_2_30.list')

thefile = open('/whirlwind/goes16/grb/meso_ircm/meso_2_60_temp.list', 'w')
thelist = file_list[-60:]

for item in thelist:
    head, tail = os.path.split(item)
    thefile.write(tail + '\n')
thefile.close
os.rename('/whirlwind/goes16/grb/meso_ircm/meso_2_60_temp.list','/whirlwind/goes16/grb/meso_ircm/meso_2_60.list')

quit()


silentremove("/whirlwind/goes16/grb/vis/mw/latest_mw_72.jpg")
silentrename("/whirlwind/goes16/grb/vis/mw/latest_mw_71.jpg", "/whirlwind/goes16/grb/vis/mw/latest_mw_72.jpg")
silentrename("/whirlwind/goes16/grb/vis/mw/latest_mw_70.jpg", "/whirlwind/goes16/grb/vis/mw/latest_mw_71.jpg")
silentrename("/whirlwind/goes16/grb/vis/mw/latest_mw_69.jpg", "/whirlwind/goes16/grb/vis/mw/latest_mw_70.jpg")
silentrename("/whirlwind/goes16/grb/vis/mw/latest_mw_68.jpg", "/whirlwind/goes16/grb/vis/mw/latest_mw_69.jpg")
silentrename("/whirlwind/goes16/grb/vis/mw/latest_mw_67.jpg", "/whirlwind/goes16/grb/vis/mw/latest_mw_68.jpg")
silentrename("/whirlwind/goes16/grb/vis/mw/latest_mw_66.jpg", "/whirlwind/goes16/grb/vis/mw/latest_mw_67.jpg")
silentrename("/whirlwind/goes16/grb/vis/mw/latest_mw_65.jpg", "/whirlwind/goes16/grb/vis/mw/latest_mw_66.jpg")
silentrename("/whirlwind/goes16/grb/vis/mw/latest_mw_64.jpg", "/whirlwind/goes16/grb/vis/mw/latest_mw_65.jpg")
silentrename("/whirlwind/goes16/grb/vis/mw/latest_mw_63.jpg", "/whirlwind/goes16/grb/vis/mw/latest_mw_64.jpg")
silentrename("/whirlwind/goes16/grb/vis/mw/latest_mw_62.jpg", "/whirlwind/goes16/grb/vis/mw/latest_mw_63.jpg")
silentrename("/whirlwind/goes16/grb/vis/mw/latest_mw_61.jpg", "/whirlwind/goes16/grb/vis/mw/latest_mw_62.jpg")
silentrename("/whirlwind/goes16/grb/vis/mw/latest_mw_60.jpg", "/whirlwind/goes16/grb/vis/mw/latest_mw_61.jpg")
silentrename("/whirlwind/goes16/grb/vis/mw/latest_mw_59.jpg", "/whirlwind/goes16/grb/vis/mw/latest_mw_60.jpg")
silentrename("/whirlwind/goes16/grb/vis/mw/latest_mw_58.jpg", "/whirlwind/goes16/grb/vis/mw/latest_mw_59.jpg")
silentrename("/whirlwind/goes16/grb/vis/mw/latest_mw_57.jpg", "/whirlwind/goes16/grb/vis/mw/latest_mw_58.jpg")
silentrename("/whirlwind/goes16/grb/vis/mw/latest_mw_56.jpg", "/whirlwind/goes16/grb/vis/mw/latest_mw_57.jpg")
silentrename("/whirlwind/goes16/grb/vis/mw/latest_mw_55.jpg", "/whirlwind/goes16/grb/vis/mw/latest_mw_56.jpg")
silentrename("/whirlwind/goes16/grb/vis/mw/latest_mw_54.jpg", "/whirlwind/goes16/grb/vis/mw/latest_mw_55.jpg")
silentrename("/whirlwind/goes16/grb/vis/mw/latest_mw_53.jpg", "/whirlwind/goes16/grb/vis/mw/latest_mw_54.jpg")
silentrename("/whirlwind/goes16/grb/vis/mw/latest_mw_52.jpg", "/whirlwind/goes16/grb/vis/mw/latest_mw_53.jpg")
silentrename("/whirlwind/goes16/grb/vis/mw/latest_mw_51.jpg", "/whirlwind/goes16/grb/vis/mw/latest_mw_52.jpg")
silentrename("/whirlwind/goes16/grb/vis/mw/latest_mw_50.jpg", "/whirlwind/goes16/grb/vis/mw/latest_mw_51.jpg")
silentrename("/whirlwind/goes16/grb/vis/mw/latest_mw_49.jpg", "/whirlwind/goes16/grb/vis/mw/latest_mw_50.jpg")
silentrename("/whirlwind/goes16/grb/vis/mw/latest_mw_48.jpg", "/whirlwind/goes16/grb/vis/mw/latest_mw_49.jpg")
silentrename("/whirlwind/goes16/grb/vis/mw/latest_mw_47.jpg", "/whirlwind/goes16/grb/vis/mw/latest_mw_48.jpg")
silentrename("/whirlwind/goes16/grb/vis/mw/latest_mw_46.jpg", "/whirlwind/goes16/grb/vis/mw/latest_mw_47.jpg")
silentrename("/whirlwind/goes16/grb/vis/mw/latest_mw_45.jpg", "/whirlwind/goes16/grb/vis/mw/latest_mw_46.jpg")
silentrename("/whirlwind/goes16/grb/vis/mw/latest_mw_44.jpg", "/whirlwind/goes16/grb/vis/mw/latest_mw_45.jpg")
silentrename("/whirlwind/goes16/grb/vis/mw/latest_mw_43.jpg", "/whirlwind/goes16/grb/vis/mw/latest_mw_44.jpg")
silentrename("/whirlwind/goes16/grb/vis/mw/latest_mw_42.jpg", "/whirlwind/goes16/grb/vis/mw/latest_mw_43.jpg")
silentrename("/whirlwind/goes16/grb/vis/mw/latest_mw_41.jpg", "/whirlwind/goes16/grb/vis/mw/latest_mw_42.jpg")
silentrename("/whirlwind/goes16/grb/vis/mw/latest_mw_40.jpg", "/whirlwind/goes16/grb/vis/mw/latest_mw_41.jpg")
silentrename("/whirlwind/goes16/grb/vis/mw/latest_mw_39.jpg", "/whirlwind/goes16/grb/vis/mw/latest_mw_40.jpg")
silentrename("/whirlwind/goes16/grb/vis/mw/latest_mw_38.jpg", "/whirlwind/goes16/grb/vis/mw/latest_mw_39.jpg")
silentrename("/whirlwind/goes16/grb/vis/mw/latest_mw_37.jpg", "/whirlwind/goes16/grb/vis/mw/latest_mw_38.jpg")
silentrename("/whirlwind/goes16/grb/vis/mw/latest_mw_36.jpg", "/whirlwind/goes16/grb/vis/mw/latest_mw_37.jpg")
silentrename("/whirlwind/goes16/grb/vis/mw/latest_mw_35.jpg", "/whirlwind/goes16/grb/vis/mw/latest_mw_36.jpg")
silentrename("/whirlwind/goes16/grb/vis/mw/latest_mw_34.jpg", "/whirlwind/goes16/grb/vis/mw/latest_mw_35.jpg")
silentrename("/whirlwind/goes16/grb/vis/mw/latest_mw_33.jpg", "/whirlwind/goes16/grb/vis/mw/latest_mw_34.jpg")
silentrename("/whirlwind/goes16/grb/vis/mw/latest_mw_32.jpg", "/whirlwind/goes16/grb/vis/mw/latest_mw_33.jpg")
silentrename("/whirlwind/goes16/grb/vis/mw/latest_mw_31.jpg", "/whirlwind/goes16/grb/vis/mw/latest_mw_32.jpg")
silentrename("/whirlwind/goes16/grb/vis/mw/latest_mw_30.jpg", "/whirlwind/goes16/grb/vis/mw/latest_mw_31.jpg")
silentrename("/whirlwind/goes16/grb/vis/mw/latest_mw_29.jpg", "/whirlwind/goes16/grb/vis/mw/latest_mw_30.jpg")
silentrename("/whirlwind/goes16/grb/vis/mw/latest_mw_28.jpg", "/whirlwind/goes16/grb/vis/mw/latest_mw_29.jpg")
silentrename("/whirlwind/goes16/grb/vis/mw/latest_mw_27.jpg", "/whirlwind/goes16/grb/vis/mw/latest_mw_28.jpg")
silentrename("/whirlwind/goes16/grb/vis/mw/latest_mw_26.jpg", "/whirlwind/goes16/grb/vis/mw/latest_mw_27.jpg")
silentrename("/whirlwind/goes16/grb/vis/mw/latest_mw_25.jpg", "/whirlwind/goes16/grb/vis/mw/latest_mw_26.jpg")
silentrename("/whirlwind/goes16/grb/vis/mw/latest_mw_24.jpg", "/whirlwind/goes16/grb/vis/mw/latest_mw_25.jpg")
silentrename("/whirlwind/goes16/grb/vis/mw/latest_mw_23.jpg", "/whirlwind/goes16/grb/vis/mw/latest_mw_24.jpg")
silentrename("/whirlwind/goes16/grb/vis/mw/latest_mw_22.jpg", "/whirlwind/goes16/grb/vis/mw/latest_mw_23.jpg")
silentrename("/whirlwind/goes16/grb/vis/mw/latest_mw_21.jpg", "/whirlwind/goes16/grb/vis/mw/latest_mw_22.jpg")
silentrename("/whirlwind/goes16/grb/vis/mw/latest_mw_20.jpg", "/whirlwind/goes16/grb/vis/mw/latest_mw_21.jpg")
silentrename("/whirlwind/goes16/grb/vis/mw/latest_mw_19.jpg", "/whirlwind/goes16/grb/vis/mw/latest_mw_20.jpg")
silentrename("/whirlwind/goes16/grb/vis/mw/latest_mw_18.jpg", "/whirlwind/goes16/grb/vis/mw/latest_mw_19.jpg")
silentrename("/whirlwind/goes16/grb/vis/mw/latest_mw_17.jpg", "/whirlwind/goes16/grb/vis/mw/latest_mw_18.jpg")
silentrename("/whirlwind/goes16/grb/vis/mw/latest_mw_16.jpg", "/whirlwind/goes16/grb/vis/mw/latest_mw_17.jpg")
silentrename("/whirlwind/goes16/grb/vis/mw/latest_mw_15.jpg", "/whirlwind/goes16/grb/vis/mw/latest_mw_16.jpg")
silentrename("/whirlwind/goes16/grb/vis/mw/latest_mw_14.jpg", "/whirlwind/goes16/grb/vis/mw/latest_mw_15.jpg")
silentrename("/whirlwind/goes16/grb/vis/mw/latest_mw_13.jpg", "/whirlwind/goes16/grb/vis/mw/latest_mw_14.jpg")
silentrename("/whirlwind/goes16/grb/vis/mw/latest_mw_12.jpg", "/whirlwind/goes16/grb/vis/mw/latest_mw_13.jpg")
silentrename("/whirlwind/goes16/grb/vis/mw/latest_mw_11.jpg", "/whirlwind/goes16/grb/vis/mw/latest_mw_12.jpg")
silentrename("/whirlwind/goes16/grb/vis/mw/latest_mw_10.jpg", "/whirlwind/goes16/grb/vis/mw/latest_mw_11.jpg")
silentrename("/whirlwind/goes16/grb/vis/mw/latest_mw_9.jpg", "/whirlwind/goes16/grb/vis/mw/latest_mw_10.jpg")
silentrename("/whirlwind/goes16/grb/vis/mw/latest_mw_8.jpg", "/whirlwind/goes16/grb/vis/mw/latest_mw_9.jpg")
silentrename("/whirlwind/goes16/grb/vis/mw/latest_mw_7.jpg", "/whirlwind/goes16/grb/vis/mw/latest_mw_8.jpg")
silentrename("/whirlwind/goes16/grb/vis/mw/latest_mw_6.jpg", "/whirlwind/goes16/grb/vis/mw/latest_mw_7.jpg")
silentrename("/whirlwind/goes16/grb/vis/mw/latest_mw_5.jpg", "/whirlwind/goes16/grb/vis/mw/latest_mw_6.jpg")
silentrename("/whirlwind/goes16/grb/vis/mw/latest_mw_4.jpg", "/whirlwind/goes16/grb/vis/mw/latest_mw_5.jpg")
silentrename("/whirlwind/goes16/grb/vis/mw/latest_mw_3.jpg", "/whirlwind/goes16/grb/vis/mw/latest_mw_4.jpg")
silentrename("/whirlwind/goes16/grb/vis/mw/latest_mw_2.jpg", "/whirlwind/goes16/grb/vis/mw/latest_mw_3.jpg")
silentrename("/whirlwind/goes16/grb/vis/mw/latest_mw_1.jpg", "/whirlwind/goes16/grb/vis/mw/latest_mw_2.jpg")

shutil.copy(filename2, "/whirlwind/goes16/grb/vis/mw/latest_mw_1.jpg")


silentremove("/whirlwind/goes16/grb/vis/conus/latest_conus_72.jpg")
silentrename("/whirlwind/goes16/grb/vis/conus/latest_conus_71.jpg", "/whirlwind/goes16/grb/vis/conus/latest_conus_72.jpg")
silentrename("/whirlwind/goes16/grb/vis/conus/latest_conus_70.jpg", "/whirlwind/goes16/grb/vis/conus/latest_conus_71.jpg")
silentrename("/whirlwind/goes16/grb/vis/conus/latest_conus_69.jpg", "/whirlwind/goes16/grb/vis/conus/latest_conus_70.jpg")
silentrename("/whirlwind/goes16/grb/vis/conus/latest_conus_68.jpg", "/whirlwind/goes16/grb/vis/conus/latest_conus_69.jpg")
silentrename("/whirlwind/goes16/grb/vis/conus/latest_conus_67.jpg", "/whirlwind/goes16/grb/vis/conus/latest_conus_68.jpg")
silentrename("/whirlwind/goes16/grb/vis/conus/latest_conus_66.jpg", "/whirlwind/goes16/grb/vis/conus/latest_conus_67.jpg")
silentrename("/whirlwind/goes16/grb/vis/conus/latest_conus_65.jpg", "/whirlwind/goes16/grb/vis/conus/latest_conus_66.jpg")
silentrename("/whirlwind/goes16/grb/vis/conus/latest_conus_64.jpg", "/whirlwind/goes16/grb/vis/conus/latest_conus_65.jpg")
silentrename("/whirlwind/goes16/grb/vis/conus/latest_conus_63.jpg", "/whirlwind/goes16/grb/vis/conus/latest_conus_64.jpg")
silentrename("/whirlwind/goes16/grb/vis/conus/latest_conus_62.jpg", "/whirlwind/goes16/grb/vis/conus/latest_conus_63.jpg")
silentrename("/whirlwind/goes16/grb/vis/conus/latest_conus_61.jpg", "/whirlwind/goes16/grb/vis/conus/latest_conus_62.jpg")
silentrename("/whirlwind/goes16/grb/vis/conus/latest_conus_60.jpg", "/whirlwind/goes16/grb/vis/conus/latest_conus_61.jpg")
silentrename("/whirlwind/goes16/grb/vis/conus/latest_conus_59.jpg", "/whirlwind/goes16/grb/vis/conus/latest_conus_60.jpg")
silentrename("/whirlwind/goes16/grb/vis/conus/latest_conus_58.jpg", "/whirlwind/goes16/grb/vis/conus/latest_conus_59.jpg")
silentrename("/whirlwind/goes16/grb/vis/conus/latest_conus_57.jpg", "/whirlwind/goes16/grb/vis/conus/latest_conus_58.jpg")
silentrename("/whirlwind/goes16/grb/vis/conus/latest_conus_56.jpg", "/whirlwind/goes16/grb/vis/conus/latest_conus_57.jpg")
silentrename("/whirlwind/goes16/grb/vis/conus/latest_conus_55.jpg", "/whirlwind/goes16/grb/vis/conus/latest_conus_56.jpg")
silentrename("/whirlwind/goes16/grb/vis/conus/latest_conus_54.jpg", "/whirlwind/goes16/grb/vis/conus/latest_conus_55.jpg")
silentrename("/whirlwind/goes16/grb/vis/conus/latest_conus_53.jpg", "/whirlwind/goes16/grb/vis/conus/latest_conus_54.jpg")
silentrename("/whirlwind/goes16/grb/vis/conus/latest_conus_52.jpg", "/whirlwind/goes16/grb/vis/conus/latest_conus_53.jpg")
silentrename("/whirlwind/goes16/grb/vis/conus/latest_conus_51.jpg", "/whirlwind/goes16/grb/vis/conus/latest_conus_52.jpg")
silentrename("/whirlwind/goes16/grb/vis/conus/latest_conus_50.jpg", "/whirlwind/goes16/grb/vis/conus/latest_conus_51.jpg")
silentrename("/whirlwind/goes16/grb/vis/conus/latest_conus_49.jpg", "/whirlwind/goes16/grb/vis/conus/latest_conus_50.jpg")
silentrename("/whirlwind/goes16/grb/vis/conus/latest_conus_48.jpg", "/whirlwind/goes16/grb/vis/conus/latest_conus_49.jpg")
silentrename("/whirlwind/goes16/grb/vis/conus/latest_conus_47.jpg", "/whirlwind/goes16/grb/vis/conus/latest_conus_48.jpg")
silentrename("/whirlwind/goes16/grb/vis/conus/latest_conus_46.jpg", "/whirlwind/goes16/grb/vis/conus/latest_conus_47.jpg")
silentrename("/whirlwind/goes16/grb/vis/conus/latest_conus_45.jpg", "/whirlwind/goes16/grb/vis/conus/latest_conus_46.jpg")
silentrename("/whirlwind/goes16/grb/vis/conus/latest_conus_44.jpg", "/whirlwind/goes16/grb/vis/conus/latest_conus_45.jpg")
silentrename("/whirlwind/goes16/grb/vis/conus/latest_conus_43.jpg", "/whirlwind/goes16/grb/vis/conus/latest_conus_44.jpg")
silentrename("/whirlwind/goes16/grb/vis/conus/latest_conus_42.jpg", "/whirlwind/goes16/grb/vis/conus/latest_conus_43.jpg")
silentrename("/whirlwind/goes16/grb/vis/conus/latest_conus_41.jpg", "/whirlwind/goes16/grb/vis/conus/latest_conus_42.jpg")
silentrename("/whirlwind/goes16/grb/vis/conus/latest_conus_40.jpg", "/whirlwind/goes16/grb/vis/conus/latest_conus_41.jpg")
silentrename("/whirlwind/goes16/grb/vis/conus/latest_conus_39.jpg", "/whirlwind/goes16/grb/vis/conus/latest_conus_40.jpg")
silentrename("/whirlwind/goes16/grb/vis/conus/latest_conus_38.jpg", "/whirlwind/goes16/grb/vis/conus/latest_conus_39.jpg")
silentrename("/whirlwind/goes16/grb/vis/conus/latest_conus_37.jpg", "/whirlwind/goes16/grb/vis/conus/latest_conus_38.jpg")
silentrename("/whirlwind/goes16/grb/vis/conus/latest_conus_36.jpg", "/whirlwind/goes16/grb/vis/conus/latest_conus_37.jpg")
silentrename("/whirlwind/goes16/grb/vis/conus/latest_conus_35.jpg", "/whirlwind/goes16/grb/vis/conus/latest_conus_36.jpg")
silentrename("/whirlwind/goes16/grb/vis/conus/latest_conus_34.jpg", "/whirlwind/goes16/grb/vis/conus/latest_conus_35.jpg")
silentrename("/whirlwind/goes16/grb/vis/conus/latest_conus_33.jpg", "/whirlwind/goes16/grb/vis/conus/latest_conus_34.jpg")
silentrename("/whirlwind/goes16/grb/vis/conus/latest_conus_32.jpg", "/whirlwind/goes16/grb/vis/conus/latest_conus_33.jpg")
silentrename("/whirlwind/goes16/grb/vis/conus/latest_conus_31.jpg", "/whirlwind/goes16/grb/vis/conus/latest_conus_32.jpg")
silentrename("/whirlwind/goes16/grb/vis/conus/latest_conus_30.jpg", "/whirlwind/goes16/grb/vis/conus/latest_conus_31.jpg")
silentrename("/whirlwind/goes16/grb/vis/conus/latest_conus_29.jpg", "/whirlwind/goes16/grb/vis/conus/latest_conus_30.jpg")
silentrename("/whirlwind/goes16/grb/vis/conus/latest_conus_28.jpg", "/whirlwind/goes16/grb/vis/conus/latest_conus_29.jpg")
silentrename("/whirlwind/goes16/grb/vis/conus/latest_conus_27.jpg", "/whirlwind/goes16/grb/vis/conus/latest_conus_28.jpg")
silentrename("/whirlwind/goes16/grb/vis/conus/latest_conus_26.jpg", "/whirlwind/goes16/grb/vis/conus/latest_conus_27.jpg")
silentrename("/whirlwind/goes16/grb/vis/conus/latest_conus_25.jpg", "/whirlwind/goes16/grb/vis/conus/latest_conus_26.jpg")
silentrename("/whirlwind/goes16/grb/vis/conus/latest_conus_24.jpg", "/whirlwind/goes16/grb/vis/conus/latest_conus_25.jpg")
silentrename("/whirlwind/goes16/grb/vis/conus/latest_conus_23.jpg", "/whirlwind/goes16/grb/vis/conus/latest_conus_24.jpg")
silentrename("/whirlwind/goes16/grb/vis/conus/latest_conus_22.jpg", "/whirlwind/goes16/grb/vis/conus/latest_conus_23.jpg")
silentrename("/whirlwind/goes16/grb/vis/conus/latest_conus_21.jpg", "/whirlwind/goes16/grb/vis/conus/latest_conus_22.jpg")
silentrename("/whirlwind/goes16/grb/vis/conus/latest_conus_20.jpg", "/whirlwind/goes16/grb/vis/conus/latest_conus_21.jpg")
silentrename("/whirlwind/goes16/grb/vis/conus/latest_conus_19.jpg", "/whirlwind/goes16/grb/vis/conus/latest_conus_20.jpg")
silentrename("/whirlwind/goes16/grb/vis/conus/latest_conus_18.jpg", "/whirlwind/goes16/grb/vis/conus/latest_conus_19.jpg")
silentrename("/whirlwind/goes16/grb/vis/conus/latest_conus_17.jpg", "/whirlwind/goes16/grb/vis/conus/latest_conus_18.jpg")
silentrename("/whirlwind/goes16/grb/vis/conus/latest_conus_16.jpg", "/whirlwind/goes16/grb/vis/conus/latest_conus_17.jpg")
silentrename("/whirlwind/goes16/grb/vis/conus/latest_conus_15.jpg", "/whirlwind/goes16/grb/vis/conus/latest_conus_16.jpg")
silentrename("/whirlwind/goes16/grb/vis/conus/latest_conus_14.jpg", "/whirlwind/goes16/grb/vis/conus/latest_conus_15.jpg")
silentrename("/whirlwind/goes16/grb/vis/conus/latest_conus_13.jpg", "/whirlwind/goes16/grb/vis/conus/latest_conus_14.jpg")
silentrename("/whirlwind/goes16/grb/vis/conus/latest_conus_12.jpg", "/whirlwind/goes16/grb/vis/conus/latest_conus_13.jpg")
silentrename("/whirlwind/goes16/grb/vis/conus/latest_conus_11.jpg", "/whirlwind/goes16/grb/vis/conus/latest_conus_12.jpg")
silentrename("/whirlwind/goes16/grb/vis/conus/latest_conus_10.jpg", "/whirlwind/goes16/grb/vis/conus/latest_conus_11.jpg")
silentrename("/whirlwind/goes16/grb/vis/conus/latest_conus_9.jpg", "/whirlwind/goes16/grb/vis/conus/latest_conus_10.jpg")
silentrename("/whirlwind/goes16/grb/vis/conus/latest_conus_8.jpg", "/whirlwind/goes16/grb/vis/conus/latest_conus_9.jpg")
silentrename("/whirlwind/goes16/grb/vis/conus/latest_conus_7.jpg", "/whirlwind/goes16/grb/vis/conus/latest_conus_8.jpg")
silentrename("/whirlwind/goes16/grb/vis/conus/latest_conus_6.jpg", "/whirlwind/goes16/grb/vis/conus/latest_conus_7.jpg")
silentrename("/whirlwind/goes16/grb/vis/conus/latest_conus_5.jpg", "/whirlwind/goes16/grb/vis/conus/latest_conus_6.jpg")
silentrename("/whirlwind/goes16/grb/vis/conus/latest_conus_4.jpg", "/whirlwind/goes16/grb/vis/conus/latest_conus_5.jpg")
silentrename("/whirlwind/goes16/grb/vis/conus/latest_conus_3.jpg", "/whirlwind/goes16/grb/vis/conus/latest_conus_4.jpg")
silentrename("/whirlwind/goes16/grb/vis/conus/latest_conus_2.jpg", "/whirlwind/goes16/grb/vis/conus/latest_conus_3.jpg")
silentrename("/whirlwind/goes16/grb/vis/conus/latest_conus_1.jpg", "/whirlwind/goes16/grb/vis/conus/latest_conus_2.jpg")

shutil.copy(filename3, "/whirlwind/goes16/grb/vis/conus/latest_conus_1.jpg")



# Northeast
silentremove("/whirlwind/goes16/grb/vis/ne/latest_ne_72.jpg")
silentrename("/whirlwind/goes16/grb/vis/ne/latest_ne_71.jpg", "/whirlwind/goes16/grb/vis/ne/latest_ne_72.jpg")
silentrename("/whirlwind/goes16/grb/vis/ne/latest_ne_70.jpg", "/whirlwind/goes16/grb/vis/ne/latest_ne_71.jpg")
silentrename("/whirlwind/goes16/grb/vis/ne/latest_ne_69.jpg", "/whirlwind/goes16/grb/vis/ne/latest_ne_70.jpg")
silentrename("/whirlwind/goes16/grb/vis/ne/latest_ne_68.jpg", "/whirlwind/goes16/grb/vis/ne/latest_ne_69.jpg")
silentrename("/whirlwind/goes16/grb/vis/ne/latest_ne_67.jpg", "/whirlwind/goes16/grb/vis/ne/latest_ne_68.jpg")
silentrename("/whirlwind/goes16/grb/vis/ne/latest_ne_66.jpg", "/whirlwind/goes16/grb/vis/ne/latest_ne_67.jpg")
silentrename("/whirlwind/goes16/grb/vis/ne/latest_ne_65.jpg", "/whirlwind/goes16/grb/vis/ne/latest_ne_66.jpg")
silentrename("/whirlwind/goes16/grb/vis/ne/latest_ne_64.jpg", "/whirlwind/goes16/grb/vis/ne/latest_ne_65.jpg")
silentrename("/whirlwind/goes16/grb/vis/ne/latest_ne_63.jpg", "/whirlwind/goes16/grb/vis/ne/latest_ne_64.jpg")
silentrename("/whirlwind/goes16/grb/vis/ne/latest_ne_62.jpg", "/whirlwind/goes16/grb/vis/ne/latest_ne_63.jpg")
silentrename("/whirlwind/goes16/grb/vis/ne/latest_ne_61.jpg", "/whirlwind/goes16/grb/vis/ne/latest_ne_62.jpg")
silentrename("/whirlwind/goes16/grb/vis/ne/latest_ne_60.jpg", "/whirlwind/goes16/grb/vis/ne/latest_ne_61.jpg")
silentrename("/whirlwind/goes16/grb/vis/ne/latest_ne_59.jpg", "/whirlwind/goes16/grb/vis/ne/latest_ne_60.jpg")
silentrename("/whirlwind/goes16/grb/vis/ne/latest_ne_58.jpg", "/whirlwind/goes16/grb/vis/ne/latest_ne_59.jpg")
silentrename("/whirlwind/goes16/grb/vis/ne/latest_ne_57.jpg", "/whirlwind/goes16/grb/vis/ne/latest_ne_58.jpg")
silentrename("/whirlwind/goes16/grb/vis/ne/latest_ne_56.jpg", "/whirlwind/goes16/grb/vis/ne/latest_ne_57.jpg")
silentrename("/whirlwind/goes16/grb/vis/ne/latest_ne_55.jpg", "/whirlwind/goes16/grb/vis/ne/latest_ne_56.jpg")
silentrename("/whirlwind/goes16/grb/vis/ne/latest_ne_54.jpg", "/whirlwind/goes16/grb/vis/ne/latest_ne_55.jpg")
silentrename("/whirlwind/goes16/grb/vis/ne/latest_ne_53.jpg", "/whirlwind/goes16/grb/vis/ne/latest_ne_54.jpg")
silentrename("/whirlwind/goes16/grb/vis/ne/latest_ne_52.jpg", "/whirlwind/goes16/grb/vis/ne/latest_ne_53.jpg")
silentrename("/whirlwind/goes16/grb/vis/ne/latest_ne_51.jpg", "/whirlwind/goes16/grb/vis/ne/latest_ne_52.jpg")
silentrename("/whirlwind/goes16/grb/vis/ne/latest_ne_50.jpg", "/whirlwind/goes16/grb/vis/ne/latest_ne_51.jpg")
silentrename("/whirlwind/goes16/grb/vis/ne/latest_ne_49.jpg", "/whirlwind/goes16/grb/vis/ne/latest_ne_50.jpg")
silentrename("/whirlwind/goes16/grb/vis/ne/latest_ne_48.jpg", "/whirlwind/goes16/grb/vis/ne/latest_ne_49.jpg")
silentrename("/whirlwind/goes16/grb/vis/ne/latest_ne_47.jpg", "/whirlwind/goes16/grb/vis/ne/latest_ne_48.jpg")
silentrename("/whirlwind/goes16/grb/vis/ne/latest_ne_46.jpg", "/whirlwind/goes16/grb/vis/ne/latest_ne_47.jpg")
silentrename("/whirlwind/goes16/grb/vis/ne/latest_ne_45.jpg", "/whirlwind/goes16/grb/vis/ne/latest_ne_46.jpg")
silentrename("/whirlwind/goes16/grb/vis/ne/latest_ne_44.jpg", "/whirlwind/goes16/grb/vis/ne/latest_ne_45.jpg")
silentrename("/whirlwind/goes16/grb/vis/ne/latest_ne_43.jpg", "/whirlwind/goes16/grb/vis/ne/latest_ne_44.jpg")
silentrename("/whirlwind/goes16/grb/vis/ne/latest_ne_42.jpg", "/whirlwind/goes16/grb/vis/ne/latest_ne_43.jpg")
silentrename("/whirlwind/goes16/grb/vis/ne/latest_ne_41.jpg", "/whirlwind/goes16/grb/vis/ne/latest_ne_42.jpg")
silentrename("/whirlwind/goes16/grb/vis/ne/latest_ne_40.jpg", "/whirlwind/goes16/grb/vis/ne/latest_ne_41.jpg")
silentrename("/whirlwind/goes16/grb/vis/ne/latest_ne_39.jpg", "/whirlwind/goes16/grb/vis/ne/latest_ne_40.jpg")
silentrename("/whirlwind/goes16/grb/vis/ne/latest_ne_38.jpg", "/whirlwind/goes16/grb/vis/ne/latest_ne_39.jpg")
silentrename("/whirlwind/goes16/grb/vis/ne/latest_ne_37.jpg", "/whirlwind/goes16/grb/vis/ne/latest_ne_38.jpg")
silentrename("/whirlwind/goes16/grb/vis/ne/latest_ne_36.jpg", "/whirlwind/goes16/grb/vis/ne/latest_ne_37.jpg")
silentrename("/whirlwind/goes16/grb/vis/ne/latest_ne_35.jpg", "/whirlwind/goes16/grb/vis/ne/latest_ne_36.jpg")
silentrename("/whirlwind/goes16/grb/vis/ne/latest_ne_34.jpg", "/whirlwind/goes16/grb/vis/ne/latest_ne_35.jpg")
silentrename("/whirlwind/goes16/grb/vis/ne/latest_ne_33.jpg", "/whirlwind/goes16/grb/vis/ne/latest_ne_34.jpg")
silentrename("/whirlwind/goes16/grb/vis/ne/latest_ne_32.jpg", "/whirlwind/goes16/grb/vis/ne/latest_ne_33.jpg")
silentrename("/whirlwind/goes16/grb/vis/ne/latest_ne_31.jpg", "/whirlwind/goes16/grb/vis/ne/latest_ne_32.jpg")
silentrename("/whirlwind/goes16/grb/vis/ne/latest_ne_30.jpg", "/whirlwind/goes16/grb/vis/ne/latest_ne_31.jpg")
silentrename("/whirlwind/goes16/grb/vis/ne/latest_ne_29.jpg", "/whirlwind/goes16/grb/vis/ne/latest_ne_30.jpg")
silentrename("/whirlwind/goes16/grb/vis/ne/latest_ne_28.jpg", "/whirlwind/goes16/grb/vis/ne/latest_ne_29.jpg")
silentrename("/whirlwind/goes16/grb/vis/ne/latest_ne_27.jpg", "/whirlwind/goes16/grb/vis/ne/latest_ne_28.jpg")
silentrename("/whirlwind/goes16/grb/vis/ne/latest_ne_26.jpg", "/whirlwind/goes16/grb/vis/ne/latest_ne_27.jpg")
silentrename("/whirlwind/goes16/grb/vis/ne/latest_ne_25.jpg", "/whirlwind/goes16/grb/vis/ne/latest_ne_26.jpg")
silentrename("/whirlwind/goes16/grb/vis/ne/latest_ne_24.jpg", "/whirlwind/goes16/grb/vis/ne/latest_ne_25.jpg")
silentrename("/whirlwind/goes16/grb/vis/ne/latest_ne_23.jpg", "/whirlwind/goes16/grb/vis/ne/latest_ne_24.jpg")
silentrename("/whirlwind/goes16/grb/vis/ne/latest_ne_22.jpg", "/whirlwind/goes16/grb/vis/ne/latest_ne_23.jpg")
silentrename("/whirlwind/goes16/grb/vis/ne/latest_ne_21.jpg", "/whirlwind/goes16/grb/vis/ne/latest_ne_22.jpg")
silentrename("/whirlwind/goes16/grb/vis/ne/latest_ne_20.jpg", "/whirlwind/goes16/grb/vis/ne/latest_ne_21.jpg")
silentrename("/whirlwind/goes16/grb/vis/ne/latest_ne_19.jpg", "/whirlwind/goes16/grb/vis/ne/latest_ne_20.jpg")
silentrename("/whirlwind/goes16/grb/vis/ne/latest_ne_18.jpg", "/whirlwind/goes16/grb/vis/ne/latest_ne_19.jpg")
silentrename("/whirlwind/goes16/grb/vis/ne/latest_ne_17.jpg", "/whirlwind/goes16/grb/vis/ne/latest_ne_18.jpg")
silentrename("/whirlwind/goes16/grb/vis/ne/latest_ne_16.jpg", "/whirlwind/goes16/grb/vis/ne/latest_ne_17.jpg")
silentrename("/whirlwind/goes16/grb/vis/ne/latest_ne_15.jpg", "/whirlwind/goes16/grb/vis/ne/latest_ne_16.jpg")
silentrename("/whirlwind/goes16/grb/vis/ne/latest_ne_14.jpg", "/whirlwind/goes16/grb/vis/ne/latest_ne_15.jpg")
silentrename("/whirlwind/goes16/grb/vis/ne/latest_ne_13.jpg", "/whirlwind/goes16/grb/vis/ne/latest_ne_14.jpg")
silentrename("/whirlwind/goes16/grb/vis/ne/latest_ne_12.jpg", "/whirlwind/goes16/grb/vis/ne/latest_ne_13.jpg")
silentrename("/whirlwind/goes16/grb/vis/ne/latest_ne_11.jpg", "/whirlwind/goes16/grb/vis/ne/latest_ne_12.jpg")
silentrename("/whirlwind/goes16/grb/vis/ne/latest_ne_10.jpg", "/whirlwind/goes16/grb/vis/ne/latest_ne_11.jpg")
silentrename("/whirlwind/goes16/grb/vis/ne/latest_ne_9.jpg", "/whirlwind/goes16/grb/vis/ne/latest_ne_10.jpg")
silentrename("/whirlwind/goes16/grb/vis/ne/latest_ne_8.jpg", "/whirlwind/goes16/grb/vis/ne/latest_ne_9.jpg")
silentrename("/whirlwind/goes16/grb/vis/ne/latest_ne_7.jpg", "/whirlwind/goes16/grb/vis/ne/latest_ne_8.jpg")
silentrename("/whirlwind/goes16/grb/vis/ne/latest_ne_6.jpg", "/whirlwind/goes16/grb/vis/ne/latest_ne_7.jpg")
silentrename("/whirlwind/goes16/grb/vis/ne/latest_ne_5.jpg", "/whirlwind/goes16/grb/vis/ne/latest_ne_6.jpg")
silentrename("/whirlwind/goes16/grb/vis/ne/latest_ne_4.jpg", "/whirlwind/goes16/grb/vis/ne/latest_ne_5.jpg")
silentrename("/whirlwind/goes16/grb/vis/ne/latest_ne_3.jpg", "/whirlwind/goes16/grb/vis/ne/latest_ne_4.jpg")
silentrename("/whirlwind/goes16/grb/vis/ne/latest_ne_2.jpg", "/whirlwind/goes16/grb/vis/ne/latest_ne_3.jpg")
silentrename("/whirlwind/goes16/grb/vis/ne/latest_ne_1.jpg", "/whirlwind/goes16/grb/vis/ne/latest_ne_2.jpg")

shutil.copy(filename4, "/whirlwind/goes16/grb/vis/ne/latest_ne_1.jpg")

# Madison close-up
silentremove("/whirlwind/goes16/grb/vis/swi/latest_swi_72.jpg")
silentrename("/whirlwind/goes16/grb/vis/swi/latest_swi_71.jpg", "/whirlwind/goes16/grb/vis/swi/latest_swi_72.jpg")
silentrename("/whirlwind/goes16/grb/vis/swi/latest_swi_70.jpg", "/whirlwind/goes16/grb/vis/swi/latest_swi_71.jpg")
silentrename("/whirlwind/goes16/grb/vis/swi/latest_swi_69.jpg", "/whirlwind/goes16/grb/vis/swi/latest_swi_70.jpg")
silentrename("/whirlwind/goes16/grb/vis/swi/latest_swi_68.jpg", "/whirlwind/goes16/grb/vis/swi/latest_swi_69.jpg")
silentrename("/whirlwind/goes16/grb/vis/swi/latest_swi_67.jpg", "/whirlwind/goes16/grb/vis/swi/latest_swi_68.jpg")
silentrename("/whirlwind/goes16/grb/vis/swi/latest_swi_66.jpg", "/whirlwind/goes16/grb/vis/swi/latest_swi_67.jpg")
silentrename("/whirlwind/goes16/grb/vis/swi/latest_swi_65.jpg", "/whirlwind/goes16/grb/vis/swi/latest_swi_66.jpg")
silentrename("/whirlwind/goes16/grb/vis/swi/latest_swi_64.jpg", "/whirlwind/goes16/grb/vis/swi/latest_swi_65.jpg")
silentrename("/whirlwind/goes16/grb/vis/swi/latest_swi_63.jpg", "/whirlwind/goes16/grb/vis/swi/latest_swi_64.jpg")
silentrename("/whirlwind/goes16/grb/vis/swi/latest_swi_62.jpg", "/whirlwind/goes16/grb/vis/swi/latest_swi_63.jpg")
silentrename("/whirlwind/goes16/grb/vis/swi/latest_swi_61.jpg", "/whirlwind/goes16/grb/vis/swi/latest_swi_62.jpg")
silentrename("/whirlwind/goes16/grb/vis/swi/latest_swi_60.jpg", "/whirlwind/goes16/grb/vis/swi/latest_swi_61.jpg")
silentrename("/whirlwind/goes16/grb/vis/swi/latest_swi_59.jpg", "/whirlwind/goes16/grb/vis/swi/latest_swi_60.jpg")
silentrename("/whirlwind/goes16/grb/vis/swi/latest_swi_58.jpg", "/whirlwind/goes16/grb/vis/swi/latest_swi_59.jpg")
silentrename("/whirlwind/goes16/grb/vis/swi/latest_swi_57.jpg", "/whirlwind/goes16/grb/vis/swi/latest_swi_58.jpg")
silentrename("/whirlwind/goes16/grb/vis/swi/latest_swi_56.jpg", "/whirlwind/goes16/grb/vis/swi/latest_swi_57.jpg")
silentrename("/whirlwind/goes16/grb/vis/swi/latest_swi_55.jpg", "/whirlwind/goes16/grb/vis/swi/latest_swi_56.jpg")
silentrename("/whirlwind/goes16/grb/vis/swi/latest_swi_54.jpg", "/whirlwind/goes16/grb/vis/swi/latest_swi_55.jpg")
silentrename("/whirlwind/goes16/grb/vis/swi/latest_swi_53.jpg", "/whirlwind/goes16/grb/vis/swi/latest_swi_54.jpg")
silentrename("/whirlwind/goes16/grb/vis/swi/latest_swi_52.jpg", "/whirlwind/goes16/grb/vis/swi/latest_swi_53.jpg")
silentrename("/whirlwind/goes16/grb/vis/swi/latest_swi_51.jpg", "/whirlwind/goes16/grb/vis/swi/latest_swi_52.jpg")
silentrename("/whirlwind/goes16/grb/vis/swi/latest_swi_50.jpg", "/whirlwind/goes16/grb/vis/swi/latest_swi_51.jpg")
silentrename("/whirlwind/goes16/grb/vis/swi/latest_swi_49.jpg", "/whirlwind/goes16/grb/vis/swi/latest_swi_50.jpg")
silentrename("/whirlwind/goes16/grb/vis/swi/latest_swi_48.jpg", "/whirlwind/goes16/grb/vis/swi/latest_swi_49.jpg")
silentrename("/whirlwind/goes16/grb/vis/swi/latest_swi_47.jpg", "/whirlwind/goes16/grb/vis/swi/latest_swi_48.jpg")
silentrename("/whirlwind/goes16/grb/vis/swi/latest_swi_46.jpg", "/whirlwind/goes16/grb/vis/swi/latest_swi_47.jpg")
silentrename("/whirlwind/goes16/grb/vis/swi/latest_swi_45.jpg", "/whirlwind/goes16/grb/vis/swi/latest_swi_46.jpg")
silentrename("/whirlwind/goes16/grb/vis/swi/latest_swi_44.jpg", "/whirlwind/goes16/grb/vis/swi/latest_swi_45.jpg")
silentrename("/whirlwind/goes16/grb/vis/swi/latest_swi_43.jpg", "/whirlwind/goes16/grb/vis/swi/latest_swi_44.jpg")
silentrename("/whirlwind/goes16/grb/vis/swi/latest_swi_42.jpg", "/whirlwind/goes16/grb/vis/swi/latest_swi_43.jpg")
silentrename("/whirlwind/goes16/grb/vis/swi/latest_swi_41.jpg", "/whirlwind/goes16/grb/vis/swi/latest_swi_42.jpg")
silentrename("/whirlwind/goes16/grb/vis/swi/latest_swi_40.jpg", "/whirlwind/goes16/grb/vis/swi/latest_swi_41.jpg")
silentrename("/whirlwind/goes16/grb/vis/swi/latest_swi_39.jpg", "/whirlwind/goes16/grb/vis/swi/latest_swi_40.jpg")
silentrename("/whirlwind/goes16/grb/vis/swi/latest_swi_38.jpg", "/whirlwind/goes16/grb/vis/swi/latest_swi_39.jpg")
silentrename("/whirlwind/goes16/grb/vis/swi/latest_swi_37.jpg", "/whirlwind/goes16/grb/vis/swi/latest_swi_38.jpg")
silentrename("/whirlwind/goes16/grb/vis/swi/latest_swi_36.jpg", "/whirlwind/goes16/grb/vis/swi/latest_swi_37.jpg")
silentrename("/whirlwind/goes16/grb/vis/swi/latest_swi_35.jpg", "/whirlwind/goes16/grb/vis/swi/latest_swi_36.jpg")
silentrename("/whirlwind/goes16/grb/vis/swi/latest_swi_34.jpg", "/whirlwind/goes16/grb/vis/swi/latest_swi_35.jpg")
silentrename("/whirlwind/goes16/grb/vis/swi/latest_swi_33.jpg", "/whirlwind/goes16/grb/vis/swi/latest_swi_34.jpg")
silentrename("/whirlwind/goes16/grb/vis/swi/latest_swi_32.jpg", "/whirlwind/goes16/grb/vis/swi/latest_swi_33.jpg")
silentrename("/whirlwind/goes16/grb/vis/swi/latest_swi_31.jpg", "/whirlwind/goes16/grb/vis/swi/latest_swi_32.jpg")
silentrename("/whirlwind/goes16/grb/vis/swi/latest_swi_30.jpg", "/whirlwind/goes16/grb/vis/swi/latest_swi_31.jpg")
silentrename("/whirlwind/goes16/grb/vis/swi/latest_swi_29.jpg", "/whirlwind/goes16/grb/vis/swi/latest_swi_30.jpg")
silentrename("/whirlwind/goes16/grb/vis/swi/latest_swi_28.jpg", "/whirlwind/goes16/grb/vis/swi/latest_swi_29.jpg")
silentrename("/whirlwind/goes16/grb/vis/swi/latest_swi_27.jpg", "/whirlwind/goes16/grb/vis/swi/latest_swi_28.jpg")
silentrename("/whirlwind/goes16/grb/vis/swi/latest_swi_26.jpg", "/whirlwind/goes16/grb/vis/swi/latest_swi_27.jpg")
silentrename("/whirlwind/goes16/grb/vis/swi/latest_swi_25.jpg", "/whirlwind/goes16/grb/vis/swi/latest_swi_26.jpg")
silentrename("/whirlwind/goes16/grb/vis/swi/latest_swi_24.jpg", "/whirlwind/goes16/grb/vis/swi/latest_swi_25.jpg")
silentrename("/whirlwind/goes16/grb/vis/swi/latest_swi_23.jpg", "/whirlwind/goes16/grb/vis/swi/latest_swi_24.jpg")
silentrename("/whirlwind/goes16/grb/vis/swi/latest_swi_22.jpg", "/whirlwind/goes16/grb/vis/swi/latest_swi_23.jpg")
silentrename("/whirlwind/goes16/grb/vis/swi/latest_swi_21.jpg", "/whirlwind/goes16/grb/vis/swi/latest_swi_22.jpg")
silentrename("/whirlwind/goes16/grb/vis/swi/latest_swi_20.jpg", "/whirlwind/goes16/grb/vis/swi/latest_swi_21.jpg")
silentrename("/whirlwind/goes16/grb/vis/swi/latest_swi_19.jpg", "/whirlwind/goes16/grb/vis/swi/latest_swi_20.jpg")
silentrename("/whirlwind/goes16/grb/vis/swi/latest_swi_18.jpg", "/whirlwind/goes16/grb/vis/swi/latest_swi_19.jpg")
silentrename("/whirlwind/goes16/grb/vis/swi/latest_swi_17.jpg", "/whirlwind/goes16/grb/vis/swi/latest_swi_18.jpg")
silentrename("/whirlwind/goes16/grb/vis/swi/latest_swi_16.jpg", "/whirlwind/goes16/grb/vis/swi/latest_swi_17.jpg")
silentrename("/whirlwind/goes16/grb/vis/swi/latest_swi_15.jpg", "/whirlwind/goes16/grb/vis/swi/latest_swi_16.jpg")
silentrename("/whirlwind/goes16/grb/vis/swi/latest_swi_14.jpg", "/whirlwind/goes16/grb/vis/swi/latest_swi_15.jpg")
silentrename("/whirlwind/goes16/grb/vis/swi/latest_swi_13.jpg", "/whirlwind/goes16/grb/vis/swi/latest_swi_14.jpg")
silentrename("/whirlwind/goes16/grb/vis/swi/latest_swi_12.jpg", "/whirlwind/goes16/grb/vis/swi/latest_swi_13.jpg")
silentrename("/whirlwind/goes16/grb/vis/swi/latest_swi_11.jpg", "/whirlwind/goes16/grb/vis/swi/latest_swi_12.jpg")
silentrename("/whirlwind/goes16/grb/vis/swi/latest_swi_10.jpg", "/whirlwind/goes16/grb/vis/swi/latest_swi_11.jpg")
silentrename("/whirlwind/goes16/grb/vis/swi/latest_swi_9.jpg", "/whirlwind/goes16/grb/vis/swi/latest_swi_10.jpg")
silentrename("/whirlwind/goes16/grb/vis/swi/latest_swi_8.jpg", "/whirlwind/goes16/grb/vis/swi/latest_swi_9.jpg")
silentrename("/whirlwind/goes16/grb/vis/swi/latest_swi_7.jpg", "/whirlwind/goes16/grb/vis/swi/latest_swi_8.jpg")
silentrename("/whirlwind/goes16/grb/vis/swi/latest_swi_6.jpg", "/whirlwind/goes16/grb/vis/swi/latest_swi_7.jpg")
silentrename("/whirlwind/goes16/grb/vis/swi/latest_swi_5.jpg", "/whirlwind/goes16/grb/vis/swi/latest_swi_6.jpg")
silentrename("/whirlwind/goes16/grb/vis/swi/latest_swi_4.jpg", "/whirlwind/goes16/grb/vis/swi/latest_swi_5.jpg")
silentrename("/whirlwind/goes16/grb/vis/swi/latest_swi_3.jpg", "/whirlwind/goes16/grb/vis/swi/latest_swi_4.jpg")
silentrename("/whirlwind/goes16/grb/vis/swi/latest_swi_2.jpg", "/whirlwind/goes16/grb/vis/swi/latest_swi_3.jpg")
silentrename("/whirlwind/goes16/grb/vis/swi/latest_swi_1.jpg", "/whirlwind/goes16/grb/vis/swi/latest_swi_2.jpg")

shutil.copy(filename5, "/whirlwind/goes16/grb/vis/swi/latest_swi_1.jpg")

# Colorado
silentremove("/whirlwind/goes16/grb/vis/co/latest_co_72.jpg")
silentrename("/whirlwind/goes16/grb/vis/co/latest_co_71.jpg", "/whirlwind/goes16/grb/vis/co/latest_co_72.jpg")
silentrename("/whirlwind/goes16/grb/vis/co/latest_co_70.jpg", "/whirlwind/goes16/grb/vis/co/latest_co_71.jpg")
silentrename("/whirlwind/goes16/grb/vis/co/latest_co_69.jpg", "/whirlwind/goes16/grb/vis/co/latest_co_70.jpg")
silentrename("/whirlwind/goes16/grb/vis/co/latest_co_68.jpg", "/whirlwind/goes16/grb/vis/co/latest_co_69.jpg")
silentrename("/whirlwind/goes16/grb/vis/co/latest_co_67.jpg", "/whirlwind/goes16/grb/vis/co/latest_co_68.jpg")
silentrename("/whirlwind/goes16/grb/vis/co/latest_co_66.jpg", "/whirlwind/goes16/grb/vis/co/latest_co_67.jpg")
silentrename("/whirlwind/goes16/grb/vis/co/latest_co_65.jpg", "/whirlwind/goes16/grb/vis/co/latest_co_66.jpg")
silentrename("/whirlwind/goes16/grb/vis/co/latest_co_64.jpg", "/whirlwind/goes16/grb/vis/co/latest_co_65.jpg")
silentrename("/whirlwind/goes16/grb/vis/co/latest_co_63.jpg", "/whirlwind/goes16/grb/vis/co/latest_co_64.jpg")
silentrename("/whirlwind/goes16/grb/vis/co/latest_co_62.jpg", "/whirlwind/goes16/grb/vis/co/latest_co_63.jpg")
silentrename("/whirlwind/goes16/grb/vis/co/latest_co_61.jpg", "/whirlwind/goes16/grb/vis/co/latest_co_62.jpg")
silentrename("/whirlwind/goes16/grb/vis/co/latest_co_60.jpg", "/whirlwind/goes16/grb/vis/co/latest_co_61.jpg")
silentrename("/whirlwind/goes16/grb/vis/co/latest_co_59.jpg", "/whirlwind/goes16/grb/vis/co/latest_co_60.jpg")
silentrename("/whirlwind/goes16/grb/vis/co/latest_co_58.jpg", "/whirlwind/goes16/grb/vis/co/latest_co_59.jpg")
silentrename("/whirlwind/goes16/grb/vis/co/latest_co_57.jpg", "/whirlwind/goes16/grb/vis/co/latest_co_58.jpg")
silentrename("/whirlwind/goes16/grb/vis/co/latest_co_56.jpg", "/whirlwind/goes16/grb/vis/co/latest_co_57.jpg")
silentrename("/whirlwind/goes16/grb/vis/co/latest_co_55.jpg", "/whirlwind/goes16/grb/vis/co/latest_co_56.jpg")
silentrename("/whirlwind/goes16/grb/vis/co/latest_co_54.jpg", "/whirlwind/goes16/grb/vis/co/latest_co_55.jpg")
silentrename("/whirlwind/goes16/grb/vis/co/latest_co_53.jpg", "/whirlwind/goes16/grb/vis/co/latest_co_54.jpg")
silentrename("/whirlwind/goes16/grb/vis/co/latest_co_52.jpg", "/whirlwind/goes16/grb/vis/co/latest_co_53.jpg")
silentrename("/whirlwind/goes16/grb/vis/co/latest_co_51.jpg", "/whirlwind/goes16/grb/vis/co/latest_co_52.jpg")
silentrename("/whirlwind/goes16/grb/vis/co/latest_co_50.jpg", "/whirlwind/goes16/grb/vis/co/latest_co_51.jpg")
silentrename("/whirlwind/goes16/grb/vis/co/latest_co_49.jpg", "/whirlwind/goes16/grb/vis/co/latest_co_50.jpg")
silentrename("/whirlwind/goes16/grb/vis/co/latest_co_48.jpg", "/whirlwind/goes16/grb/vis/co/latest_co_49.jpg")
silentrename("/whirlwind/goes16/grb/vis/co/latest_co_47.jpg", "/whirlwind/goes16/grb/vis/co/latest_co_48.jpg")
silentrename("/whirlwind/goes16/grb/vis/co/latest_co_46.jpg", "/whirlwind/goes16/grb/vis/co/latest_co_47.jpg")
silentrename("/whirlwind/goes16/grb/vis/co/latest_co_45.jpg", "/whirlwind/goes16/grb/vis/co/latest_co_46.jpg")
silentrename("/whirlwind/goes16/grb/vis/co/latest_co_44.jpg", "/whirlwind/goes16/grb/vis/co/latest_co_45.jpg")
silentrename("/whirlwind/goes16/grb/vis/co/latest_co_43.jpg", "/whirlwind/goes16/grb/vis/co/latest_co_44.jpg")
silentrename("/whirlwind/goes16/grb/vis/co/latest_co_42.jpg", "/whirlwind/goes16/grb/vis/co/latest_co_43.jpg")
silentrename("/whirlwind/goes16/grb/vis/co/latest_co_41.jpg", "/whirlwind/goes16/grb/vis/co/latest_co_42.jpg")
silentrename("/whirlwind/goes16/grb/vis/co/latest_co_40.jpg", "/whirlwind/goes16/grb/vis/co/latest_co_41.jpg")
silentrename("/whirlwind/goes16/grb/vis/co/latest_co_39.jpg", "/whirlwind/goes16/grb/vis/co/latest_co_40.jpg")
silentrename("/whirlwind/goes16/grb/vis/co/latest_co_38.jpg", "/whirlwind/goes16/grb/vis/co/latest_co_39.jpg")
silentrename("/whirlwind/goes16/grb/vis/co/latest_co_37.jpg", "/whirlwind/goes16/grb/vis/co/latest_co_38.jpg")
silentrename("/whirlwind/goes16/grb/vis/co/latest_co_36.jpg", "/whirlwind/goes16/grb/vis/co/latest_co_37.jpg")
silentrename("/whirlwind/goes16/grb/vis/co/latest_co_35.jpg", "/whirlwind/goes16/grb/vis/co/latest_co_36.jpg")
silentrename("/whirlwind/goes16/grb/vis/co/latest_co_34.jpg", "/whirlwind/goes16/grb/vis/co/latest_co_35.jpg")
silentrename("/whirlwind/goes16/grb/vis/co/latest_co_33.jpg", "/whirlwind/goes16/grb/vis/co/latest_co_34.jpg")
silentrename("/whirlwind/goes16/grb/vis/co/latest_co_32.jpg", "/whirlwind/goes16/grb/vis/co/latest_co_33.jpg")
silentrename("/whirlwind/goes16/grb/vis/co/latest_co_31.jpg", "/whirlwind/goes16/grb/vis/co/latest_co_32.jpg")
silentrename("/whirlwind/goes16/grb/vis/co/latest_co_30.jpg", "/whirlwind/goes16/grb/vis/co/latest_co_31.jpg")
silentrename("/whirlwind/goes16/grb/vis/co/latest_co_29.jpg", "/whirlwind/goes16/grb/vis/co/latest_co_30.jpg")
silentrename("/whirlwind/goes16/grb/vis/co/latest_co_28.jpg", "/whirlwind/goes16/grb/vis/co/latest_co_29.jpg")
silentrename("/whirlwind/goes16/grb/vis/co/latest_co_27.jpg", "/whirlwind/goes16/grb/vis/co/latest_co_28.jpg")
silentrename("/whirlwind/goes16/grb/vis/co/latest_co_26.jpg", "/whirlwind/goes16/grb/vis/co/latest_co_27.jpg")
silentrename("/whirlwind/goes16/grb/vis/co/latest_co_25.jpg", "/whirlwind/goes16/grb/vis/co/latest_co_26.jpg")
silentrename("/whirlwind/goes16/grb/vis/co/latest_co_24.jpg", "/whirlwind/goes16/grb/vis/co/latest_co_25.jpg")
silentrename("/whirlwind/goes16/grb/vis/co/latest_co_23.jpg", "/whirlwind/goes16/grb/vis/co/latest_co_24.jpg")
silentrename("/whirlwind/goes16/grb/vis/co/latest_co_22.jpg", "/whirlwind/goes16/grb/vis/co/latest_co_23.jpg")
silentrename("/whirlwind/goes16/grb/vis/co/latest_co_21.jpg", "/whirlwind/goes16/grb/vis/co/latest_co_22.jpg")
silentrename("/whirlwind/goes16/grb/vis/co/latest_co_20.jpg", "/whirlwind/goes16/grb/vis/co/latest_co_21.jpg")
silentrename("/whirlwind/goes16/grb/vis/co/latest_co_19.jpg", "/whirlwind/goes16/grb/vis/co/latest_co_20.jpg")
silentrename("/whirlwind/goes16/grb/vis/co/latest_co_18.jpg", "/whirlwind/goes16/grb/vis/co/latest_co_19.jpg")
silentrename("/whirlwind/goes16/grb/vis/co/latest_co_17.jpg", "/whirlwind/goes16/grb/vis/co/latest_co_18.jpg")
silentrename("/whirlwind/goes16/grb/vis/co/latest_co_16.jpg", "/whirlwind/goes16/grb/vis/co/latest_co_17.jpg")
silentrename("/whirlwind/goes16/grb/vis/co/latest_co_15.jpg", "/whirlwind/goes16/grb/vis/co/latest_co_16.jpg")
silentrename("/whirlwind/goes16/grb/vis/co/latest_co_14.jpg", "/whirlwind/goes16/grb/vis/co/latest_co_15.jpg")
silentrename("/whirlwind/goes16/grb/vis/co/latest_co_13.jpg", "/whirlwind/goes16/grb/vis/co/latest_co_14.jpg")
silentrename("/whirlwind/goes16/grb/vis/co/latest_co_12.jpg", "/whirlwind/goes16/grb/vis/co/latest_co_13.jpg")
silentrename("/whirlwind/goes16/grb/vis/co/latest_co_11.jpg", "/whirlwind/goes16/grb/vis/co/latest_co_12.jpg")
silentrename("/whirlwind/goes16/grb/vis/co/latest_co_10.jpg", "/whirlwind/goes16/grb/vis/co/latest_co_11.jpg")
silentrename("/whirlwind/goes16/grb/vis/co/latest_co_9.jpg", "/whirlwind/goes16/grb/vis/co/latest_co_10.jpg")
silentrename("/whirlwind/goes16/grb/vis/co/latest_co_8.jpg", "/whirlwind/goes16/grb/vis/co/latest_co_9.jpg")
silentrename("/whirlwind/goes16/grb/vis/co/latest_co_7.jpg", "/whirlwind/goes16/grb/vis/co/latest_co_8.jpg")
silentrename("/whirlwind/goes16/grb/vis/co/latest_co_6.jpg", "/whirlwind/goes16/grb/vis/co/latest_co_7.jpg")
silentrename("/whirlwind/goes16/grb/vis/co/latest_co_5.jpg", "/whirlwind/goes16/grb/vis/co/latest_co_6.jpg")
silentrename("/whirlwind/goes16/grb/vis/co/latest_co_4.jpg", "/whirlwind/goes16/grb/vis/co/latest_co_5.jpg")
silentrename("/whirlwind/goes16/grb/vis/co/latest_co_3.jpg", "/whirlwind/goes16/grb/vis/co/latest_co_4.jpg")
silentrename("/whirlwind/goes16/grb/vis/co/latest_co_2.jpg", "/whirlwind/goes16/grb/vis/co/latest_co_3.jpg")
silentrename("/whirlwind/goes16/grb/vis/co/latest_co_1.jpg", "/whirlwind/goes16/grb/vis/co/latest_co_2.jpg")

shutil.copy(filename6, "/whirlwind/goes16/grb/vis/co/latest_co_1.jpg")

# Florida
silentremove("/whirlwind/goes16/grb/vis/fl/latest_fl_72.jpg")
silentrename("/whirlwind/goes16/grb/vis/fl/latest_fl_71.jpg", "/whirlwind/goes16/grb/vis/fl/latest_fl_72.jpg")
silentrename("/whirlwind/goes16/grb/vis/fl/latest_fl_70.jpg", "/whirlwind/goes16/grb/vis/fl/latest_fl_71.jpg")
silentrename("/whirlwind/goes16/grb/vis/fl/latest_fl_69.jpg", "/whirlwind/goes16/grb/vis/fl/latest_fl_70.jpg")
silentrename("/whirlwind/goes16/grb/vis/fl/latest_fl_68.jpg", "/whirlwind/goes16/grb/vis/fl/latest_fl_69.jpg")
silentrename("/whirlwind/goes16/grb/vis/fl/latest_fl_67.jpg", "/whirlwind/goes16/grb/vis/fl/latest_fl_68.jpg")
silentrename("/whirlwind/goes16/grb/vis/fl/latest_fl_66.jpg", "/whirlwind/goes16/grb/vis/fl/latest_fl_67.jpg")
silentrename("/whirlwind/goes16/grb/vis/fl/latest_fl_65.jpg", "/whirlwind/goes16/grb/vis/fl/latest_fl_66.jpg")
silentrename("/whirlwind/goes16/grb/vis/fl/latest_fl_64.jpg", "/whirlwind/goes16/grb/vis/fl/latest_fl_65.jpg")
silentrename("/whirlwind/goes16/grb/vis/fl/latest_fl_63.jpg", "/whirlwind/goes16/grb/vis/fl/latest_fl_64.jpg")
silentrename("/whirlwind/goes16/grb/vis/fl/latest_fl_62.jpg", "/whirlwind/goes16/grb/vis/fl/latest_fl_63.jpg")
silentrename("/whirlwind/goes16/grb/vis/fl/latest_fl_61.jpg", "/whirlwind/goes16/grb/vis/fl/latest_fl_62.jpg")
silentrename("/whirlwind/goes16/grb/vis/fl/latest_fl_60.jpg", "/whirlwind/goes16/grb/vis/fl/latest_fl_61.jpg")
silentrename("/whirlwind/goes16/grb/vis/fl/latest_fl_59.jpg", "/whirlwind/goes16/grb/vis/fl/latest_fl_60.jpg")
silentrename("/whirlwind/goes16/grb/vis/fl/latest_fl_58.jpg", "/whirlwind/goes16/grb/vis/fl/latest_fl_59.jpg")
silentrename("/whirlwind/goes16/grb/vis/fl/latest_fl_57.jpg", "/whirlwind/goes16/grb/vis/fl/latest_fl_58.jpg")
silentrename("/whirlwind/goes16/grb/vis/fl/latest_fl_56.jpg", "/whirlwind/goes16/grb/vis/fl/latest_fl_57.jpg")
silentrename("/whirlwind/goes16/grb/vis/fl/latest_fl_55.jpg", "/whirlwind/goes16/grb/vis/fl/latest_fl_56.jpg")
silentrename("/whirlwind/goes16/grb/vis/fl/latest_fl_54.jpg", "/whirlwind/goes16/grb/vis/fl/latest_fl_55.jpg")
silentrename("/whirlwind/goes16/grb/vis/fl/latest_fl_53.jpg", "/whirlwind/goes16/grb/vis/fl/latest_fl_54.jpg")
silentrename("/whirlwind/goes16/grb/vis/fl/latest_fl_52.jpg", "/whirlwind/goes16/grb/vis/fl/latest_fl_53.jpg")
silentrename("/whirlwind/goes16/grb/vis/fl/latest_fl_51.jpg", "/whirlwind/goes16/grb/vis/fl/latest_fl_52.jpg")
silentrename("/whirlwind/goes16/grb/vis/fl/latest_fl_50.jpg", "/whirlwind/goes16/grb/vis/fl/latest_fl_51.jpg")
silentrename("/whirlwind/goes16/grb/vis/fl/latest_fl_49.jpg", "/whirlwind/goes16/grb/vis/fl/latest_fl_50.jpg")
silentrename("/whirlwind/goes16/grb/vis/fl/latest_fl_48.jpg", "/whirlwind/goes16/grb/vis/fl/latest_fl_49.jpg")
silentrename("/whirlwind/goes16/grb/vis/fl/latest_fl_47.jpg", "/whirlwind/goes16/grb/vis/fl/latest_fl_48.jpg")
silentrename("/whirlwind/goes16/grb/vis/fl/latest_fl_46.jpg", "/whirlwind/goes16/grb/vis/fl/latest_fl_47.jpg")
silentrename("/whirlwind/goes16/grb/vis/fl/latest_fl_45.jpg", "/whirlwind/goes16/grb/vis/fl/latest_fl_46.jpg")
silentrename("/whirlwind/goes16/grb/vis/fl/latest_fl_44.jpg", "/whirlwind/goes16/grb/vis/fl/latest_fl_45.jpg")
silentrename("/whirlwind/goes16/grb/vis/fl/latest_fl_43.jpg", "/whirlwind/goes16/grb/vis/fl/latest_fl_44.jpg")
silentrename("/whirlwind/goes16/grb/vis/fl/latest_fl_42.jpg", "/whirlwind/goes16/grb/vis/fl/latest_fl_43.jpg")
silentrename("/whirlwind/goes16/grb/vis/fl/latest_fl_41.jpg", "/whirlwind/goes16/grb/vis/fl/latest_fl_42.jpg")
silentrename("/whirlwind/goes16/grb/vis/fl/latest_fl_40.jpg", "/whirlwind/goes16/grb/vis/fl/latest_fl_41.jpg")
silentrename("/whirlwind/goes16/grb/vis/fl/latest_fl_39.jpg", "/whirlwind/goes16/grb/vis/fl/latest_fl_40.jpg")
silentrename("/whirlwind/goes16/grb/vis/fl/latest_fl_38.jpg", "/whirlwind/goes16/grb/vis/fl/latest_fl_39.jpg")
silentrename("/whirlwind/goes16/grb/vis/fl/latest_fl_37.jpg", "/whirlwind/goes16/grb/vis/fl/latest_fl_38.jpg")
silentrename("/whirlwind/goes16/grb/vis/fl/latest_fl_36.jpg", "/whirlwind/goes16/grb/vis/fl/latest_fl_37.jpg")
silentrename("/whirlwind/goes16/grb/vis/fl/latest_fl_35.jpg", "/whirlwind/goes16/grb/vis/fl/latest_fl_36.jpg")
silentrename("/whirlwind/goes16/grb/vis/fl/latest_fl_34.jpg", "/whirlwind/goes16/grb/vis/fl/latest_fl_35.jpg")
silentrename("/whirlwind/goes16/grb/vis/fl/latest_fl_33.jpg", "/whirlwind/goes16/grb/vis/fl/latest_fl_34.jpg")
silentrename("/whirlwind/goes16/grb/vis/fl/latest_fl_32.jpg", "/whirlwind/goes16/grb/vis/fl/latest_fl_33.jpg")
silentrename("/whirlwind/goes16/grb/vis/fl/latest_fl_31.jpg", "/whirlwind/goes16/grb/vis/fl/latest_fl_32.jpg")
silentrename("/whirlwind/goes16/grb/vis/fl/latest_fl_30.jpg", "/whirlwind/goes16/grb/vis/fl/latest_fl_31.jpg")
silentrename("/whirlwind/goes16/grb/vis/fl/latest_fl_29.jpg", "/whirlwind/goes16/grb/vis/fl/latest_fl_30.jpg")
silentrename("/whirlwind/goes16/grb/vis/fl/latest_fl_28.jpg", "/whirlwind/goes16/grb/vis/fl/latest_fl_29.jpg")
silentrename("/whirlwind/goes16/grb/vis/fl/latest_fl_27.jpg", "/whirlwind/goes16/grb/vis/fl/latest_fl_28.jpg")
silentrename("/whirlwind/goes16/grb/vis/fl/latest_fl_26.jpg", "/whirlwind/goes16/grb/vis/fl/latest_fl_27.jpg")
silentrename("/whirlwind/goes16/grb/vis/fl/latest_fl_25.jpg", "/whirlwind/goes16/grb/vis/fl/latest_fl_26.jpg")
silentrename("/whirlwind/goes16/grb/vis/fl/latest_fl_24.jpg", "/whirlwind/goes16/grb/vis/fl/latest_fl_25.jpg")
silentrename("/whirlwind/goes16/grb/vis/fl/latest_fl_23.jpg", "/whirlwind/goes16/grb/vis/fl/latest_fl_24.jpg")
silentrename("/whirlwind/goes16/grb/vis/fl/latest_fl_22.jpg", "/whirlwind/goes16/grb/vis/fl/latest_fl_23.jpg")
silentrename("/whirlwind/goes16/grb/vis/fl/latest_fl_21.jpg", "/whirlwind/goes16/grb/vis/fl/latest_fl_22.jpg")
silentrename("/whirlwind/goes16/grb/vis/fl/latest_fl_20.jpg", "/whirlwind/goes16/grb/vis/fl/latest_fl_21.jpg")
silentrename("/whirlwind/goes16/grb/vis/fl/latest_fl_19.jpg", "/whirlwind/goes16/grb/vis/fl/latest_fl_20.jpg")
silentrename("/whirlwind/goes16/grb/vis/fl/latest_fl_18.jpg", "/whirlwind/goes16/grb/vis/fl/latest_fl_19.jpg")
silentrename("/whirlwind/goes16/grb/vis/fl/latest_fl_17.jpg", "/whirlwind/goes16/grb/vis/fl/latest_fl_18.jpg")
silentrename("/whirlwind/goes16/grb/vis/fl/latest_fl_16.jpg", "/whirlwind/goes16/grb/vis/fl/latest_fl_17.jpg")
silentrename("/whirlwind/goes16/grb/vis/fl/latest_fl_15.jpg", "/whirlwind/goes16/grb/vis/fl/latest_fl_16.jpg")
silentrename("/whirlwind/goes16/grb/vis/fl/latest_fl_14.jpg", "/whirlwind/goes16/grb/vis/fl/latest_fl_15.jpg")
silentrename("/whirlwind/goes16/grb/vis/fl/latest_fl_13.jpg", "/whirlwind/goes16/grb/vis/fl/latest_fl_14.jpg")
silentrename("/whirlwind/goes16/grb/vis/fl/latest_fl_12.jpg", "/whirlwind/goes16/grb/vis/fl/latest_fl_13.jpg")
silentrename("/whirlwind/goes16/grb/vis/fl/latest_fl_11.jpg", "/whirlwind/goes16/grb/vis/fl/latest_fl_12.jpg")
silentrename("/whirlwind/goes16/grb/vis/fl/latest_fl_10.jpg", "/whirlwind/goes16/grb/vis/fl/latest_fl_11.jpg")
silentrename("/whirlwind/goes16/grb/vis/fl/latest_fl_9.jpg", "/whirlwind/goes16/grb/vis/fl/latest_fl_10.jpg")
silentrename("/whirlwind/goes16/grb/vis/fl/latest_fl_8.jpg", "/whirlwind/goes16/grb/vis/fl/latest_fl_9.jpg")
silentrename("/whirlwind/goes16/grb/vis/fl/latest_fl_7.jpg", "/whirlwind/goes16/grb/vis/fl/latest_fl_8.jpg")
silentrename("/whirlwind/goes16/grb/vis/fl/latest_fl_6.jpg", "/whirlwind/goes16/grb/vis/fl/latest_fl_7.jpg")
silentrename("/whirlwind/goes16/grb/vis/fl/latest_fl_5.jpg", "/whirlwind/goes16/grb/vis/fl/latest_fl_6.jpg")
silentrename("/whirlwind/goes16/grb/vis/fl/latest_fl_4.jpg", "/whirlwind/goes16/grb/vis/fl/latest_fl_5.jpg")
silentrename("/whirlwind/goes16/grb/vis/fl/latest_fl_3.jpg", "/whirlwind/goes16/grb/vis/fl/latest_fl_4.jpg")
silentrename("/whirlwind/goes16/grb/vis/fl/latest_fl_2.jpg", "/whirlwind/goes16/grb/vis/fl/latest_fl_3.jpg")
silentrename("/whirlwind/goes16/grb/vis/fl/latest_fl_1.jpg", "/whirlwind/goes16/grb/vis/fl/latest_fl_2.jpg")

shutil.copy(filename7, "/whirlwind/goes16/grb/vis/fl/latest_fl_1.jpg")

# Gulf of Mexico
silentremove("/whirlwind/goes16/grb/vis/gulf/latest_gulf_72.jpg")
silentrename("/whirlwind/goes16/grb/vis/gulf/latest_gulf_71.jpg", "/whirlwind/goes16/grb/vis/gulf/latest_gulf_72.jpg")
silentrename("/whirlwind/goes16/grb/vis/gulf/latest_gulf_70.jpg", "/whirlwind/goes16/grb/vis/gulf/latest_gulf_71.jpg")
silentrename("/whirlwind/goes16/grb/vis/gulf/latest_gulf_69.jpg", "/whirlwind/goes16/grb/vis/gulf/latest_gulf_70.jpg")
silentrename("/whirlwind/goes16/grb/vis/gulf/latest_gulf_68.jpg", "/whirlwind/goes16/grb/vis/gulf/latest_gulf_69.jpg")
silentrename("/whirlwind/goes16/grb/vis/gulf/latest_gulf_67.jpg", "/whirlwind/goes16/grb/vis/gulf/latest_gulf_68.jpg")
silentrename("/whirlwind/goes16/grb/vis/gulf/latest_gulf_66.jpg", "/whirlwind/goes16/grb/vis/gulf/latest_gulf_67.jpg")
silentrename("/whirlwind/goes16/grb/vis/gulf/latest_gulf_65.jpg", "/whirlwind/goes16/grb/vis/gulf/latest_gulf_66.jpg")
silentrename("/whirlwind/goes16/grb/vis/gulf/latest_gulf_64.jpg", "/whirlwind/goes16/grb/vis/gulf/latest_gulf_65.jpg")
silentrename("/whirlwind/goes16/grb/vis/gulf/latest_gulf_63.jpg", "/whirlwind/goes16/grb/vis/gulf/latest_gulf_64.jpg")
silentrename("/whirlwind/goes16/grb/vis/gulf/latest_gulf_62.jpg", "/whirlwind/goes16/grb/vis/gulf/latest_gulf_63.jpg")
silentrename("/whirlwind/goes16/grb/vis/gulf/latest_gulf_61.jpg", "/whirlwind/goes16/grb/vis/gulf/latest_gulf_62.jpg")
silentrename("/whirlwind/goes16/grb/vis/gulf/latest_gulf_60.jpg", "/whirlwind/goes16/grb/vis/gulf/latest_gulf_61.jpg")
silentrename("/whirlwind/goes16/grb/vis/gulf/latest_gulf_59.jpg", "/whirlwind/goes16/grb/vis/gulf/latest_gulf_60.jpg")
silentrename("/whirlwind/goes16/grb/vis/gulf/latest_gulf_58.jpg", "/whirlwind/goes16/grb/vis/gulf/latest_gulf_59.jpg")
silentrename("/whirlwind/goes16/grb/vis/gulf/latest_gulf_57.jpg", "/whirlwind/goes16/grb/vis/gulf/latest_gulf_58.jpg")
silentrename("/whirlwind/goes16/grb/vis/gulf/latest_gulf_56.jpg", "/whirlwind/goes16/grb/vis/gulf/latest_gulf_57.jpg")
silentrename("/whirlwind/goes16/grb/vis/gulf/latest_gulf_55.jpg", "/whirlwind/goes16/grb/vis/gulf/latest_gulf_56.jpg")
silentrename("/whirlwind/goes16/grb/vis/gulf/latest_gulf_54.jpg", "/whirlwind/goes16/grb/vis/gulf/latest_gulf_55.jpg")
silentrename("/whirlwind/goes16/grb/vis/gulf/latest_gulf_53.jpg", "/whirlwind/goes16/grb/vis/gulf/latest_gulf_54.jpg")
silentrename("/whirlwind/goes16/grb/vis/gulf/latest_gulf_52.jpg", "/whirlwind/goes16/grb/vis/gulf/latest_gulf_53.jpg")
silentrename("/whirlwind/goes16/grb/vis/gulf/latest_gulf_51.jpg", "/whirlwind/goes16/grb/vis/gulf/latest_gulf_52.jpg")
silentrename("/whirlwind/goes16/grb/vis/gulf/latest_gulf_50.jpg", "/whirlwind/goes16/grb/vis/gulf/latest_gulf_51.jpg")
silentrename("/whirlwind/goes16/grb/vis/gulf/latest_gulf_49.jpg", "/whirlwind/goes16/grb/vis/gulf/latest_gulf_50.jpg")
silentrename("/whirlwind/goes16/grb/vis/gulf/latest_gulf_48.jpg", "/whirlwind/goes16/grb/vis/gulf/latest_gulf_49.jpg")
silentrename("/whirlwind/goes16/grb/vis/gulf/latest_gulf_47.jpg", "/whirlwind/goes16/grb/vis/gulf/latest_gulf_48.jpg")
silentrename("/whirlwind/goes16/grb/vis/gulf/latest_gulf_46.jpg", "/whirlwind/goes16/grb/vis/gulf/latest_gulf_47.jpg")
silentrename("/whirlwind/goes16/grb/vis/gulf/latest_gulf_45.jpg", "/whirlwind/goes16/grb/vis/gulf/latest_gulf_46.jpg")
silentrename("/whirlwind/goes16/grb/vis/gulf/latest_gulf_44.jpg", "/whirlwind/goes16/grb/vis/gulf/latest_gulf_45.jpg")
silentrename("/whirlwind/goes16/grb/vis/gulf/latest_gulf_43.jpg", "/whirlwind/goes16/grb/vis/gulf/latest_gulf_44.jpg")
silentrename("/whirlwind/goes16/grb/vis/gulf/latest_gulf_42.jpg", "/whirlwind/goes16/grb/vis/gulf/latest_gulf_43.jpg")
silentrename("/whirlwind/goes16/grb/vis/gulf/latest_gulf_41.jpg", "/whirlwind/goes16/grb/vis/gulf/latest_gulf_42.jpg")
silentrename("/whirlwind/goes16/grb/vis/gulf/latest_gulf_40.jpg", "/whirlwind/goes16/grb/vis/gulf/latest_gulf_41.jpg")
silentrename("/whirlwind/goes16/grb/vis/gulf/latest_gulf_39.jpg", "/whirlwind/goes16/grb/vis/gulf/latest_gulf_40.jpg")
silentrename("/whirlwind/goes16/grb/vis/gulf/latest_gulf_38.jpg", "/whirlwind/goes16/grb/vis/gulf/latest_gulf_39.jpg")
silentrename("/whirlwind/goes16/grb/vis/gulf/latest_gulf_37.jpg", "/whirlwind/goes16/grb/vis/gulf/latest_gulf_38.jpg")
silentrename("/whirlwind/goes16/grb/vis/gulf/latest_gulf_36.jpg", "/whirlwind/goes16/grb/vis/gulf/latest_gulf_37.jpg")
silentrename("/whirlwind/goes16/grb/vis/gulf/latest_gulf_35.jpg", "/whirlwind/goes16/grb/vis/gulf/latest_gulf_36.jpg")
silentrename("/whirlwind/goes16/grb/vis/gulf/latest_gulf_34.jpg", "/whirlwind/goes16/grb/vis/gulf/latest_gulf_35.jpg")
silentrename("/whirlwind/goes16/grb/vis/gulf/latest_gulf_33.jpg", "/whirlwind/goes16/grb/vis/gulf/latest_gulf_34.jpg")
silentrename("/whirlwind/goes16/grb/vis/gulf/latest_gulf_32.jpg", "/whirlwind/goes16/grb/vis/gulf/latest_gulf_33.jpg")
silentrename("/whirlwind/goes16/grb/vis/gulf/latest_gulf_31.jpg", "/whirlwind/goes16/grb/vis/gulf/latest_gulf_32.jpg")
silentrename("/whirlwind/goes16/grb/vis/gulf/latest_gulf_30.jpg", "/whirlwind/goes16/grb/vis/gulf/latest_gulf_31.jpg")
silentrename("/whirlwind/goes16/grb/vis/gulf/latest_gulf_29.jpg", "/whirlwind/goes16/grb/vis/gulf/latest_gulf_30.jpg")
silentrename("/whirlwind/goes16/grb/vis/gulf/latest_gulf_28.jpg", "/whirlwind/goes16/grb/vis/gulf/latest_gulf_29.jpg")
silentrename("/whirlwind/goes16/grb/vis/gulf/latest_gulf_27.jpg", "/whirlwind/goes16/grb/vis/gulf/latest_gulf_28.jpg")
silentrename("/whirlwind/goes16/grb/vis/gulf/latest_gulf_26.jpg", "/whirlwind/goes16/grb/vis/gulf/latest_gulf_27.jpg")
silentrename("/whirlwind/goes16/grb/vis/gulf/latest_gulf_25.jpg", "/whirlwind/goes16/grb/vis/gulf/latest_gulf_26.jpg")
silentrename("/whirlwind/goes16/grb/vis/gulf/latest_gulf_24.jpg", "/whirlwind/goes16/grb/vis/gulf/latest_gulf_25.jpg")
silentrename("/whirlwind/goes16/grb/vis/gulf/latest_gulf_23.jpg", "/whirlwind/goes16/grb/vis/gulf/latest_gulf_24.jpg")
silentrename("/whirlwind/goes16/grb/vis/gulf/latest_gulf_22.jpg", "/whirlwind/goes16/grb/vis/gulf/latest_gulf_23.jpg")
silentrename("/whirlwind/goes16/grb/vis/gulf/latest_gulf_21.jpg", "/whirlwind/goes16/grb/vis/gulf/latest_gulf_22.jpg")
silentrename("/whirlwind/goes16/grb/vis/gulf/latest_gulf_20.jpg", "/whirlwind/goes16/grb/vis/gulf/latest_gulf_21.jpg")
silentrename("/whirlwind/goes16/grb/vis/gulf/latest_gulf_19.jpg", "/whirlwind/goes16/grb/vis/gulf/latest_gulf_20.jpg")
silentrename("/whirlwind/goes16/grb/vis/gulf/latest_gulf_18.jpg", "/whirlwind/goes16/grb/vis/gulf/latest_gulf_19.jpg")
silentrename("/whirlwind/goes16/grb/vis/gulf/latest_gulf_17.jpg", "/whirlwind/goes16/grb/vis/gulf/latest_gulf_18.jpg")
silentrename("/whirlwind/goes16/grb/vis/gulf/latest_gulf_16.jpg", "/whirlwind/goes16/grb/vis/gulf/latest_gulf_17.jpg")
silentrename("/whirlwind/goes16/grb/vis/gulf/latest_gulf_15.jpg", "/whirlwind/goes16/grb/vis/gulf/latest_gulf_16.jpg")
silentrename("/whirlwind/goes16/grb/vis/gulf/latest_gulf_14.jpg", "/whirlwind/goes16/grb/vis/gulf/latest_gulf_15.jpg")
silentrename("/whirlwind/goes16/grb/vis/gulf/latest_gulf_13.jpg", "/whirlwind/goes16/grb/vis/gulf/latest_gulf_14.jpg")
silentrename("/whirlwind/goes16/grb/vis/gulf/latest_gulf_12.jpg", "/whirlwind/goes16/grb/vis/gulf/latest_gulf_13.jpg")
silentrename("/whirlwind/goes16/grb/vis/gulf/latest_gulf_11.jpg", "/whirlwind/goes16/grb/vis/gulf/latest_gulf_12.jpg")
silentrename("/whirlwind/goes16/grb/vis/gulf/latest_gulf_10.jpg", "/whirlwind/goes16/grb/vis/gulf/latest_gulf_11.jpg")
silentrename("/whirlwind/goes16/grb/vis/gulf/latest_gulf_9.jpg", "/whirlwind/goes16/grb/vis/gulf/latest_gulf_10.jpg")
silentrename("/whirlwind/goes16/grb/vis/gulf/latest_gulf_8.jpg", "/whirlwind/goes16/grb/vis/gulf/latest_gulf_9.jpg")
silentrename("/whirlwind/goes16/grb/vis/gulf/latest_gulf_7.jpg", "/whirlwind/goes16/grb/vis/gulf/latest_gulf_8.jpg")
silentrename("/whirlwind/goes16/grb/vis/gulf/latest_gulf_6.jpg", "/whirlwind/goes16/grb/vis/gulf/latest_gulf_7.jpg")
silentrename("/whirlwind/goes16/grb/vis/gulf/latest_gulf_5.jpg", "/whirlwind/goes16/grb/vis/gulf/latest_gulf_6.jpg")
silentrename("/whirlwind/goes16/grb/vis/gulf/latest_gulf_4.jpg", "/whirlwind/goes16/grb/vis/gulf/latest_gulf_5.jpg")
silentrename("/whirlwind/goes16/grb/vis/gulf/latest_gulf_3.jpg", "/whirlwind/goes16/grb/vis/gulf/latest_gulf_4.jpg")
silentrename("/whirlwind/goes16/grb/vis/gulf/latest_gulf_2.jpg", "/whirlwind/goes16/grb/vis/gulf/latest_gulf_3.jpg")
silentrename("/whirlwind/goes16/grb/vis/gulf/latest_gulf_1.jpg", "/whirlwind/goes16/grb/vis/gulf/latest_gulf_2.jpg")

shutil.copy(filename8, "/whirlwind/goes16/grb/vis/gulf/latest_gulf_1.jpg")
# Northeast

silentremove("/whirlwind/goes16/grb/vis/greatlakes/latest_greatlakes_72.jpg")
silentrename("/whirlwind/goes16/grb/vis/greatlakes/latest_greatlakes_71.jpg", "/whirlwind/goes16/grb/vis/greatlakes/latest_greatlakes_72.jpg")
silentrename("/whirlwind/goes16/grb/vis/greatlakes/latest_greatlakes_70.jpg", "/whirlwind/goes16/grb/vis/greatlakes/latest_greatlakes_71.jpg")
silentrename("/whirlwind/goes16/grb/vis/greatlakes/latest_greatlakes_69.jpg", "/whirlwind/goes16/grb/vis/greatlakes/latest_greatlakes_70.jpg")
silentrename("/whirlwind/goes16/grb/vis/greatlakes/latest_greatlakes_68.jpg", "/whirlwind/goes16/grb/vis/greatlakes/latest_greatlakes_69.jpg")
silentrename("/whirlwind/goes16/grb/vis/greatlakes/latest_greatlakes_67.jpg", "/whirlwind/goes16/grb/vis/greatlakes/latest_greatlakes_68.jpg")
silentrename("/whirlwind/goes16/grb/vis/greatlakes/latest_greatlakes_66.jpg", "/whirlwind/goes16/grb/vis/greatlakes/latest_greatlakes_67.jpg")
silentrename("/whirlwind/goes16/grb/vis/greatlakes/latest_greatlakes_65.jpg", "/whirlwind/goes16/grb/vis/greatlakes/latest_greatlakes_66.jpg")
silentrename("/whirlwind/goes16/grb/vis/greatlakes/latest_greatlakes_64.jpg", "/whirlwind/goes16/grb/vis/greatlakes/latest_greatlakes_65.jpg")
silentrename("/whirlwind/goes16/grb/vis/greatlakes/latest_greatlakes_63.jpg", "/whirlwind/goes16/grb/vis/greatlakes/latest_greatlakes_64.jpg")
silentrename("/whirlwind/goes16/grb/vis/greatlakes/latest_greatlakes_62.jpg", "/whirlwind/goes16/grb/vis/greatlakes/latest_greatlakes_63.jpg")
silentrename("/whirlwind/goes16/grb/vis/greatlakes/latest_greatlakes_61.jpg", "/whirlwind/goes16/grb/vis/greatlakes/latest_greatlakes_62.jpg")
silentrename("/whirlwind/goes16/grb/vis/greatlakes/latest_greatlakes_60.jpg", "/whirlwind/goes16/grb/vis/greatlakes/latest_greatlakes_61.jpg")
silentrename("/whirlwind/goes16/grb/vis/greatlakes/latest_greatlakes_59.jpg", "/whirlwind/goes16/grb/vis/greatlakes/latest_greatlakes_60.jpg")
silentrename("/whirlwind/goes16/grb/vis/greatlakes/latest_greatlakes_58.jpg", "/whirlwind/goes16/grb/vis/greatlakes/latest_greatlakes_59.jpg")
silentrename("/whirlwind/goes16/grb/vis/greatlakes/latest_greatlakes_57.jpg", "/whirlwind/goes16/grb/vis/greatlakes/latest_greatlakes_58.jpg")
silentrename("/whirlwind/goes16/grb/vis/greatlakes/latest_greatlakes_56.jpg", "/whirlwind/goes16/grb/vis/greatlakes/latest_greatlakes_57.jpg")
silentrename("/whirlwind/goes16/grb/vis/greatlakes/latest_greatlakes_55.jpg", "/whirlwind/goes16/grb/vis/greatlakes/latest_greatlakes_56.jpg")
silentrename("/whirlwind/goes16/grb/vis/greatlakes/latest_greatlakes_54.jpg", "/whirlwind/goes16/grb/vis/greatlakes/latest_greatlakes_55.jpg")
silentrename("/whirlwind/goes16/grb/vis/greatlakes/latest_greatlakes_53.jpg", "/whirlwind/goes16/grb/vis/greatlakes/latest_greatlakes_54.jpg")
silentrename("/whirlwind/goes16/grb/vis/greatlakes/latest_greatlakes_52.jpg", "/whirlwind/goes16/grb/vis/greatlakes/latest_greatlakes_53.jpg")
silentrename("/whirlwind/goes16/grb/vis/greatlakes/latest_greatlakes_51.jpg", "/whirlwind/goes16/grb/vis/greatlakes/latest_greatlakes_52.jpg")
silentrename("/whirlwind/goes16/grb/vis/greatlakes/latest_greatlakes_50.jpg", "/whirlwind/goes16/grb/vis/greatlakes/latest_greatlakes_51.jpg")
silentrename("/whirlwind/goes16/grb/vis/greatlakes/latest_greatlakes_49.jpg", "/whirlwind/goes16/grb/vis/greatlakes/latest_greatlakes_50.jpg")
silentrename("/whirlwind/goes16/grb/vis/greatlakes/latest_greatlakes_48.jpg", "/whirlwind/goes16/grb/vis/greatlakes/latest_greatlakes_49.jpg")
silentrename("/whirlwind/goes16/grb/vis/greatlakes/latest_greatlakes_47.jpg", "/whirlwind/goes16/grb/vis/greatlakes/latest_greatlakes_48.jpg")
silentrename("/whirlwind/goes16/grb/vis/greatlakes/latest_greatlakes_46.jpg", "/whirlwind/goes16/grb/vis/greatlakes/latest_greatlakes_47.jpg")
silentrename("/whirlwind/goes16/grb/vis/greatlakes/latest_greatlakes_45.jpg", "/whirlwind/goes16/grb/vis/greatlakes/latest_greatlakes_46.jpg")
silentrename("/whirlwind/goes16/grb/vis/greatlakes/latest_greatlakes_44.jpg", "/whirlwind/goes16/grb/vis/greatlakes/latest_greatlakes_45.jpg")
silentrename("/whirlwind/goes16/grb/vis/greatlakes/latest_greatlakes_43.jpg", "/whirlwind/goes16/grb/vis/greatlakes/latest_greatlakes_44.jpg")
silentrename("/whirlwind/goes16/grb/vis/greatlakes/latest_greatlakes_42.jpg", "/whirlwind/goes16/grb/vis/greatlakes/latest_greatlakes_43.jpg")
silentrename("/whirlwind/goes16/grb/vis/greatlakes/latest_greatlakes_41.jpg", "/whirlwind/goes16/grb/vis/greatlakes/latest_greatlakes_42.jpg")
silentrename("/whirlwind/goes16/grb/vis/greatlakes/latest_greatlakes_40.jpg", "/whirlwind/goes16/grb/vis/greatlakes/latest_greatlakes_41.jpg")
silentrename("/whirlwind/goes16/grb/vis/greatlakes/latest_greatlakes_39.jpg", "/whirlwind/goes16/grb/vis/greatlakes/latest_greatlakes_40.jpg")
silentrename("/whirlwind/goes16/grb/vis/greatlakes/latest_greatlakes_38.jpg", "/whirlwind/goes16/grb/vis/greatlakes/latest_greatlakes_39.jpg")
silentrename("/whirlwind/goes16/grb/vis/greatlakes/latest_greatlakes_37.jpg", "/whirlwind/goes16/grb/vis/greatlakes/latest_greatlakes_38.jpg")
silentrename("/whirlwind/goes16/grb/vis/greatlakes/latest_greatlakes_36.jpg", "/whirlwind/goes16/grb/vis/greatlakes/latest_greatlakes_37.jpg")
silentrename("/whirlwind/goes16/grb/vis/greatlakes/latest_greatlakes_35.jpg", "/whirlwind/goes16/grb/vis/greatlakes/latest_greatlakes_36.jpg")
silentrename("/whirlwind/goes16/grb/vis/greatlakes/latest_greatlakes_34.jpg", "/whirlwind/goes16/grb/vis/greatlakes/latest_greatlakes_35.jpg")
silentrename("/whirlwind/goes16/grb/vis/greatlakes/latest_greatlakes_33.jpg", "/whirlwind/goes16/grb/vis/greatlakes/latest_greatlakes_34.jpg")
silentrename("/whirlwind/goes16/grb/vis/greatlakes/latest_greatlakes_32.jpg", "/whirlwind/goes16/grb/vis/greatlakes/latest_greatlakes_33.jpg")
silentrename("/whirlwind/goes16/grb/vis/greatlakes/latest_greatlakes_31.jpg", "/whirlwind/goes16/grb/vis/greatlakes/latest_greatlakes_32.jpg")
silentrename("/whirlwind/goes16/grb/vis/greatlakes/latest_greatlakes_30.jpg", "/whirlwind/goes16/grb/vis/greatlakes/latest_greatlakes_31.jpg")
silentrename("/whirlwind/goes16/grb/vis/greatlakes/latest_greatlakes_29.jpg", "/whirlwind/goes16/grb/vis/greatlakes/latest_greatlakes_30.jpg")
silentrename("/whirlwind/goes16/grb/vis/greatlakes/latest_greatlakes_28.jpg", "/whirlwind/goes16/grb/vis/greatlakes/latest_greatlakes_29.jpg")
silentrename("/whirlwind/goes16/grb/vis/greatlakes/latest_greatlakes_27.jpg", "/whirlwind/goes16/grb/vis/greatlakes/latest_greatlakes_28.jpg")
silentrename("/whirlwind/goes16/grb/vis/greatlakes/latest_greatlakes_26.jpg", "/whirlwind/goes16/grb/vis/greatlakes/latest_greatlakes_27.jpg")
silentrename("/whirlwind/goes16/grb/vis/greatlakes/latest_greatlakes_25.jpg", "/whirlwind/goes16/grb/vis/greatlakes/latest_greatlakes_26.jpg")
silentrename("/whirlwind/goes16/grb/vis/greatlakes/latest_greatlakes_24.jpg", "/whirlwind/goes16/grb/vis/greatlakes/latest_greatlakes_25.jpg")
silentrename("/whirlwind/goes16/grb/vis/greatlakes/latest_greatlakes_23.jpg", "/whirlwind/goes16/grb/vis/greatlakes/latest_greatlakes_24.jpg")
silentrename("/whirlwind/goes16/grb/vis/greatlakes/latest_greatlakes_22.jpg", "/whirlwind/goes16/grb/vis/greatlakes/latest_greatlakes_23.jpg")
silentrename("/whirlwind/goes16/grb/vis/greatlakes/latest_greatlakes_21.jpg", "/whirlwind/goes16/grb/vis/greatlakes/latest_greatlakes_22.jpg")
silentrename("/whirlwind/goes16/grb/vis/greatlakes/latest_greatlakes_20.jpg", "/whirlwind/goes16/grb/vis/greatlakes/latest_greatlakes_21.jpg")
silentrename("/whirlwind/goes16/grb/vis/greatlakes/latest_greatlakes_19.jpg", "/whirlwind/goes16/grb/vis/greatlakes/latest_greatlakes_20.jpg")
silentrename("/whirlwind/goes16/grb/vis/greatlakes/latest_greatlakes_18.jpg", "/whirlwind/goes16/grb/vis/greatlakes/latest_greatlakes_19.jpg")
silentrename("/whirlwind/goes16/grb/vis/greatlakes/latest_greatlakes_17.jpg", "/whirlwind/goes16/grb/vis/greatlakes/latest_greatlakes_18.jpg")
silentrename("/whirlwind/goes16/grb/vis/greatlakes/latest_greatlakes_16.jpg", "/whirlwind/goes16/grb/vis/greatlakes/latest_greatlakes_17.jpg")
silentrename("/whirlwind/goes16/grb/vis/greatlakes/latest_greatlakes_15.jpg", "/whirlwind/goes16/grb/vis/greatlakes/latest_greatlakes_16.jpg")
silentrename("/whirlwind/goes16/grb/vis/greatlakes/latest_greatlakes_14.jpg", "/whirlwind/goes16/grb/vis/greatlakes/latest_greatlakes_15.jpg")
silentrename("/whirlwind/goes16/grb/vis/greatlakes/latest_greatlakes_13.jpg", "/whirlwind/goes16/grb/vis/greatlakes/latest_greatlakes_14.jpg")
silentrename("/whirlwind/goes16/grb/vis/greatlakes/latest_greatlakes_12.jpg", "/whirlwind/goes16/grb/vis/greatlakes/latest_greatlakes_13.jpg")
silentrename("/whirlwind/goes16/grb/vis/greatlakes/latest_greatlakes_11.jpg", "/whirlwind/goes16/grb/vis/greatlakes/latest_greatlakes_12.jpg")
silentrename("/whirlwind/goes16/grb/vis/greatlakes/latest_greatlakes_10.jpg", "/whirlwind/goes16/grb/vis/greatlakes/latest_greatlakes_11.jpg")
silentrename("/whirlwind/goes16/grb/vis/greatlakes/latest_greatlakes_9.jpg", "/whirlwind/goes16/grb/vis/greatlakes/latest_greatlakes_10.jpg")
silentrename("/whirlwind/goes16/grb/vis/greatlakes/latest_greatlakes_8.jpg", "/whirlwind/goes16/grb/vis/greatlakes/latest_greatlakes_9.jpg")
silentrename("/whirlwind/goes16/grb/vis/greatlakes/latest_greatlakes_7.jpg", "/whirlwind/goes16/grb/vis/greatlakes/latest_greatlakes_8.jpg")
silentrename("/whirlwind/goes16/grb/vis/greatlakes/latest_greatlakes_6.jpg", "/whirlwind/goes16/grb/vis/greatlakes/latest_greatlakes_7.jpg")
silentrename("/whirlwind/goes16/grb/vis/greatlakes/latest_greatlakes_5.jpg", "/whirlwind/goes16/grb/vis/greatlakes/latest_greatlakes_6.jpg")
silentrename("/whirlwind/goes16/grb/vis/greatlakes/latest_greatlakes_4.jpg", "/whirlwind/goes16/grb/vis/greatlakes/latest_greatlakes_5.jpg")
silentrename("/whirlwind/goes16/grb/vis/greatlakes/latest_greatlakes_3.jpg", "/whirlwind/goes16/grb/vis/greatlakes/latest_greatlakes_4.jpg")
silentrename("/whirlwind/goes16/grb/vis/greatlakes/latest_greatlakes_2.jpg", "/whirlwind/goes16/grb/vis/greatlakes/latest_greatlakes_3.jpg")
silentrename("/whirlwind/goes16/grb/vis/greatlakes/latest_greatlakes_1.jpg", "/whirlwind/goes16/grb/vis/greatlakes/latest_greatlakes_2.jpg")

shutil.copy(filename11, "/whirlwind/goes16/grb/vis/greatlakes/latest_greatlakes_1.jpg")

quit()

# Alex Stormchase
os.remove("/whirlwind/goes16/vis/alex/latest_alex_72.jpg")
os.rename("/whirlwind/goes16/vis/alex/latest_alex_71.jpg", "/whirlwind/goes16/vis/alex/latest_alex_72.jpg")
os.rename("/whirlwind/goes16/vis/alex/latest_alex_70.jpg", "/whirlwind/goes16/vis/alex/latest_alex_71.jpg")
os.rename("/whirlwind/goes16/vis/alex/latest_alex_69.jpg", "/whirlwind/goes16/vis/alex/latest_alex_70.jpg")
os.rename("/whirlwind/goes16/vis/alex/latest_alex_68.jpg", "/whirlwind/goes16/vis/alex/latest_alex_69.jpg")
os.rename("/whirlwind/goes16/vis/alex/latest_alex_67.jpg", "/whirlwind/goes16/vis/alex/latest_alex_68.jpg")
os.rename("/whirlwind/goes16/vis/alex/latest_alex_66.jpg", "/whirlwind/goes16/vis/alex/latest_alex_67.jpg")
os.rename("/whirlwind/goes16/vis/alex/latest_alex_65.jpg", "/whirlwind/goes16/vis/alex/latest_alex_66.jpg")
os.rename("/whirlwind/goes16/vis/alex/latest_alex_64.jpg", "/whirlwind/goes16/vis/alex/latest_alex_65.jpg")
os.rename("/whirlwind/goes16/vis/alex/latest_alex_63.jpg", "/whirlwind/goes16/vis/alex/latest_alex_64.jpg")
os.rename("/whirlwind/goes16/vis/alex/latest_alex_62.jpg", "/whirlwind/goes16/vis/alex/latest_alex_63.jpg")
os.rename("/whirlwind/goes16/vis/alex/latest_alex_61.jpg", "/whirlwind/goes16/vis/alex/latest_alex_62.jpg")
os.rename("/whirlwind/goes16/vis/alex/latest_alex_60.jpg", "/whirlwind/goes16/vis/alex/latest_alex_61.jpg")
os.rename("/whirlwind/goes16/vis/alex/latest_alex_59.jpg", "/whirlwind/goes16/vis/alex/latest_alex_60.jpg")
os.rename("/whirlwind/goes16/vis/alex/latest_alex_58.jpg", "/whirlwind/goes16/vis/alex/latest_alex_59.jpg")
os.rename("/whirlwind/goes16/vis/alex/latest_alex_57.jpg", "/whirlwind/goes16/vis/alex/latest_alex_58.jpg")
os.rename("/whirlwind/goes16/vis/alex/latest_alex_56.jpg", "/whirlwind/goes16/vis/alex/latest_alex_57.jpg")
os.rename("/whirlwind/goes16/vis/alex/latest_alex_55.jpg", "/whirlwind/goes16/vis/alex/latest_alex_56.jpg")
os.rename("/whirlwind/goes16/vis/alex/latest_alex_54.jpg", "/whirlwind/goes16/vis/alex/latest_alex_55.jpg")
os.rename("/whirlwind/goes16/vis/alex/latest_alex_53.jpg", "/whirlwind/goes16/vis/alex/latest_alex_54.jpg")
os.rename("/whirlwind/goes16/vis/alex/latest_alex_52.jpg", "/whirlwind/goes16/vis/alex/latest_alex_53.jpg")
os.rename("/whirlwind/goes16/vis/alex/latest_alex_51.jpg", "/whirlwind/goes16/vis/alex/latest_alex_52.jpg")
os.rename("/whirlwind/goes16/vis/alex/latest_alex_50.jpg", "/whirlwind/goes16/vis/alex/latest_alex_51.jpg")
os.rename("/whirlwind/goes16/vis/alex/latest_alex_49.jpg", "/whirlwind/goes16/vis/alex/latest_alex_50.jpg")
os.rename("/whirlwind/goes16/vis/alex/latest_alex_48.jpg", "/whirlwind/goes16/vis/alex/latest_alex_49.jpg")
os.rename("/whirlwind/goes16/vis/alex/latest_alex_47.jpg", "/whirlwind/goes16/vis/alex/latest_alex_48.jpg")
os.rename("/whirlwind/goes16/vis/alex/latest_alex_46.jpg", "/whirlwind/goes16/vis/alex/latest_alex_47.jpg")
os.rename("/whirlwind/goes16/vis/alex/latest_alex_45.jpg", "/whirlwind/goes16/vis/alex/latest_alex_46.jpg")
os.rename("/whirlwind/goes16/vis/alex/latest_alex_44.jpg", "/whirlwind/goes16/vis/alex/latest_alex_45.jpg")
os.rename("/whirlwind/goes16/vis/alex/latest_alex_43.jpg", "/whirlwind/goes16/vis/alex/latest_alex_44.jpg")
os.rename("/whirlwind/goes16/vis/alex/latest_alex_42.jpg", "/whirlwind/goes16/vis/alex/latest_alex_43.jpg")
os.rename("/whirlwind/goes16/vis/alex/latest_alex_41.jpg", "/whirlwind/goes16/vis/alex/latest_alex_42.jpg")
os.rename("/whirlwind/goes16/vis/alex/latest_alex_40.jpg", "/whirlwind/goes16/vis/alex/latest_alex_41.jpg")
os.rename("/whirlwind/goes16/vis/alex/latest_alex_39.jpg", "/whirlwind/goes16/vis/alex/latest_alex_40.jpg")
os.rename("/whirlwind/goes16/vis/alex/latest_alex_38.jpg", "/whirlwind/goes16/vis/alex/latest_alex_39.jpg")
os.rename("/whirlwind/goes16/vis/alex/latest_alex_37.jpg", "/whirlwind/goes16/vis/alex/latest_alex_38.jpg")
os.rename("/whirlwind/goes16/vis/alex/latest_alex_36.jpg", "/whirlwind/goes16/vis/alex/latest_alex_37.jpg")
os.rename("/whirlwind/goes16/vis/alex/latest_alex_35.jpg", "/whirlwind/goes16/vis/alex/latest_alex_36.jpg")
os.rename("/whirlwind/goes16/vis/alex/latest_alex_34.jpg", "/whirlwind/goes16/vis/alex/latest_alex_35.jpg")
os.rename("/whirlwind/goes16/vis/alex/latest_alex_33.jpg", "/whirlwind/goes16/vis/alex/latest_alex_34.jpg")
os.rename("/whirlwind/goes16/vis/alex/latest_alex_32.jpg", "/whirlwind/goes16/vis/alex/latest_alex_33.jpg")
os.rename("/whirlwind/goes16/vis/alex/latest_alex_31.jpg", "/whirlwind/goes16/vis/alex/latest_alex_32.jpg")
os.rename("/whirlwind/goes16/vis/alex/latest_alex_30.jpg", "/whirlwind/goes16/vis/alex/latest_alex_31.jpg")
os.rename("/whirlwind/goes16/vis/alex/latest_alex_29.jpg", "/whirlwind/goes16/vis/alex/latest_alex_30.jpg")
os.rename("/whirlwind/goes16/vis/alex/latest_alex_28.jpg", "/whirlwind/goes16/vis/alex/latest_alex_29.jpg")
os.rename("/whirlwind/goes16/vis/alex/latest_alex_27.jpg", "/whirlwind/goes16/vis/alex/latest_alex_28.jpg")
os.rename("/whirlwind/goes16/vis/alex/latest_alex_26.jpg", "/whirlwind/goes16/vis/alex/latest_alex_27.jpg")
os.rename("/whirlwind/goes16/vis/alex/latest_alex_25.jpg", "/whirlwind/goes16/vis/alex/latest_alex_26.jpg")
os.rename("/whirlwind/goes16/vis/alex/latest_alex_24.jpg", "/whirlwind/goes16/vis/alex/latest_alex_25.jpg")
os.rename("/whirlwind/goes16/vis/alex/latest_alex_23.jpg", "/whirlwind/goes16/vis/alex/latest_alex_24.jpg")
os.rename("/whirlwind/goes16/vis/alex/latest_alex_22.jpg", "/whirlwind/goes16/vis/alex/latest_alex_23.jpg")
os.rename("/whirlwind/goes16/vis/alex/latest_alex_21.jpg", "/whirlwind/goes16/vis/alex/latest_alex_22.jpg")
os.rename("/whirlwind/goes16/vis/alex/latest_alex_20.jpg", "/whirlwind/goes16/vis/alex/latest_alex_21.jpg")
os.rename("/whirlwind/goes16/vis/alex/latest_alex_19.jpg", "/whirlwind/goes16/vis/alex/latest_alex_20.jpg")
os.rename("/whirlwind/goes16/vis/alex/latest_alex_18.jpg", "/whirlwind/goes16/vis/alex/latest_alex_19.jpg")
os.rename("/whirlwind/goes16/vis/alex/latest_alex_17.jpg", "/whirlwind/goes16/vis/alex/latest_alex_18.jpg")
os.rename("/whirlwind/goes16/vis/alex/latest_alex_16.jpg", "/whirlwind/goes16/vis/alex/latest_alex_17.jpg")
os.rename("/whirlwind/goes16/vis/alex/latest_alex_15.jpg", "/whirlwind/goes16/vis/alex/latest_alex_16.jpg")
os.rename("/whirlwind/goes16/vis/alex/latest_alex_14.jpg", "/whirlwind/goes16/vis/alex/latest_alex_15.jpg")
os.rename("/whirlwind/goes16/vis/alex/latest_alex_13.jpg", "/whirlwind/goes16/vis/alex/latest_alex_14.jpg")
os.rename("/whirlwind/goes16/vis/alex/latest_alex_12.jpg", "/whirlwind/goes16/vis/alex/latest_alex_13.jpg")
os.rename("/whirlwind/goes16/vis/alex/latest_alex_11.jpg", "/whirlwind/goes16/vis/alex/latest_alex_12.jpg")
os.rename("/whirlwind/goes16/vis/alex/latest_alex_10.jpg", "/whirlwind/goes16/vis/alex/latest_alex_11.jpg")
os.rename("/whirlwind/goes16/vis/alex/latest_alex_9.jpg", "/whirlwind/goes16/vis/alex/latest_alex_10.jpg")
os.rename("/whirlwind/goes16/vis/alex/latest_alex_8.jpg", "/whirlwind/goes16/vis/alex/latest_alex_9.jpg")
os.rename("/whirlwind/goes16/vis/alex/latest_alex_7.jpg", "/whirlwind/goes16/vis/alex/latest_alex_8.jpg")
os.rename("/whirlwind/goes16/vis/alex/latest_alex_6.jpg", "/whirlwind/goes16/vis/alex/latest_alex_7.jpg")
os.rename("/whirlwind/goes16/vis/alex/latest_alex_5.jpg", "/whirlwind/goes16/vis/alex/latest_alex_6.jpg")
os.rename("/whirlwind/goes16/vis/alex/latest_alex_4.jpg", "/whirlwind/goes16/vis/alex/latest_alex_5.jpg")
os.rename("/whirlwind/goes16/vis/alex/latest_alex_3.jpg", "/whirlwind/goes16/vis/alex/latest_alex_4.jpg")
os.rename("/whirlwind/goes16/vis/alex/latest_alex_2.jpg", "/whirlwind/goes16/vis/alex/latest_alex_3.jpg")
os.rename("/whirlwind/goes16/vis/alex/latest_alex_1.jpg", "/whirlwind/goes16/vis/alex/latest_alex_2.jpg")

shutil.copy(filename10, "/whirlwind/goes16/vis/alex/latest_alex_1.jpg")


shutil.copy(filename9, "/whirlwind/goes16/vis/full/latest_full_1.jpg")

#####


    
silentremove("/whirlwind/goes16/irc13m/namer/latest_namer_24.jpg")
silentrename("/whirlwind/goes16/irc13m/namer/latest_namer_23.jpg", "/whirlwind/goes16/irc13m/namer/latest_namer_24.jpg")
silentrename("/whirlwind/goes16/irc13m/namer/latest_namer_22.jpg", "/whirlwind/goes16/irc13m/namer/latest_namer_23.jpg")
silentrename("/whirlwind/goes16/irc13m/namer/latest_namer_21.jpg", "/whirlwind/goes16/irc13m/namer/latest_namer_22.jpg")
silentrename("/whirlwind/goes16/irc13m/namer/latest_namer_20.jpg", "/whirlwind/goes16/irc13m/namer/latest_namer_21.jpg")
silentrename("/whirlwind/goes16/irc13m/namer/latest_namer_19.jpg", "/whirlwind/goes16/irc13m/namer/latest_namer_20.jpg")
silentrename("/whirlwind/goes16/irc13m/namer/latest_namer_18.jpg", "/whirlwind/goes16/irc13m/namer/latest_namer_19.jpg")
silentrename("/whirlwind/goes16/irc13m/namer/latest_namer_17.jpg", "/whirlwind/goes16/irc13m/namer/latest_namer_18.jpg")
silentrename("/whirlwind/goes16/irc13m/namer/latest_namer_16.jpg", "/whirlwind/goes16/irc13m/namer/latest_namer_17.jpg")
silentrename("/whirlwind/goes16/irc13m/namer/latest_namer_15.jpg", "/whirlwind/goes16/irc13m/namer/latest_namer_16.jpg")
silentrename("/whirlwind/goes16/irc13m/namer/latest_namer_14.jpg", "/whirlwind/goes16/irc13m/namer/latest_namer_15.jpg")
silentrename("/whirlwind/goes16/irc13m/namer/latest_namer_13.jpg", "/whirlwind/goes16/irc13m/namer/latest_namer_14.jpg")
silentrename("/whirlwind/goes16/irc13m/namer/latest_namer_12.jpg", "/whirlwind/goes16/irc13m/namer/latest_namer_13.jpg")
silentrename("/whirlwind/goes16/irc13m/namer/latest_namer_11.jpg", "/whirlwind/goes16/irc13m/namer/latest_namer_12.jpg")
silentrename("/whirlwind/goes16/irc13m/namer/latest_namer_10.jpg", "/whirlwind/goes16/irc13m/namer/latest_namer_11.jpg")
silentrename("/whirlwind/goes16/irc13m/namer/latest_namer_9.jpg", "/whirlwind/goes16/irc13m/namer/latest_namer_10.jpg")
silentrename("/whirlwind/goes16/irc13m/namer/latest_namer_8.jpg", "/whirlwind/goes16/irc13m/namer/latest_namer_9.jpg")
silentrename("/whirlwind/goes16/irc13m/namer/latest_namer_7.jpg", "/whirlwind/goes16/irc13m/namer/latest_namer_8.jpg")
silentrename("/whirlwind/goes16/irc13m/namer/latest_namer_6.jpg", "/whirlwind/goes16/irc13m/namer/latest_namer_7.jpg")
silentrename("/whirlwind/goes16/irc13m/namer/latest_namer_5.jpg", "/whirlwind/goes16/irc13m/namer/latest_namer_6.jpg")
silentrename("/whirlwind/goes16/irc13m/namer/latest_namer_4.jpg", "/whirlwind/goes16/irc13m/namer/latest_namer_5.jpg")
silentrename("/whirlwind/goes16/irc13m/namer/latest_namer_3.jpg", "/whirlwind/goes16/irc13m/namer/latest_namer_4.jpg")
silentrename("/whirlwind/goes16/irc13m/namer/latest_namer_2.jpg", "/whirlwind/goes16/irc13m/namer/latest_namer_3.jpg")
silentrename("/whirlwind/goes16/irc13m/namer/latest_namer_1.jpg", "/whirlwind/goes16/irc13m/namer/latest_namer_2.jpg")
    
shutil.copy(filename1, "/whirlwind/goes16/irc13m/namer/latest_namer_1.jpg")


silentremove("/whirlwind/goes16/irc13m/fulldisk/latest_fulldisk_24.jpg")
silentrename("/whirlwind/goes16/irc13m/fulldisk/latest_fulldisk_23.jpg", "/whirlwind/goes16/irc13m/fulldisk/latest_fulldisk_24.jpg")
silentrename("/whirlwind/goes16/irc13m/fulldisk/latest_fulldisk_22.jpg", "/whirlwind/goes16/irc13m/fulldisk/latest_fulldisk_23.jpg")
silentrename("/whirlwind/goes16/irc13m/fulldisk/latest_fulldisk_21.jpg", "/whirlwind/goes16/irc13m/fulldisk/latest_fulldisk_22.jpg")
silentrename("/whirlwind/goes16/irc13m/fulldisk/latest_fulldisk_20.jpg", "/whirlwind/goes16/irc13m/fulldisk/latest_fulldisk_21.jpg")
silentrename("/whirlwind/goes16/irc13m/fulldisk/latest_fulldisk_19.jpg", "/whirlwind/goes16/irc13m/fulldisk/latest_fulldisk_20.jpg")
silentrename("/whirlwind/goes16/irc13m/fulldisk/latest_fulldisk_18.jpg", "/whirlwind/goes16/irc13m/fulldisk/latest_fulldisk_19.jpg")
silentrename("/whirlwind/goes16/irc13m/fulldisk/latest_fulldisk_17.jpg", "/whirlwind/goes16/irc13m/fulldisk/latest_fulldisk_18.jpg")
silentrename("/whirlwind/goes16/irc13m/fulldisk/latest_fulldisk_16.jpg", "/whirlwind/goes16/irc13m/fulldisk/latest_fulldisk_17.jpg")
silentrename("/whirlwind/goes16/irc13m/fulldisk/latest_fulldisk_15.jpg", "/whirlwind/goes16/irc13m/fulldisk/latest_fulldisk_16.jpg")
silentrename("/whirlwind/goes16/irc13m/fulldisk/latest_fulldisk_14.jpg", "/whirlwind/goes16/irc13m/fulldisk/latest_fulldisk_15.jpg")
silentrename("/whirlwind/goes16/irc13m/fulldisk/latest_fulldisk_13.jpg", "/whirlwind/goes16/irc13m/fulldisk/latest_fulldisk_14.jpg")
silentrename("/whirlwind/goes16/irc13m/fulldisk/latest_fulldisk_12.jpg", "/whirlwind/goes16/irc13m/fulldisk/latest_fulldisk_13.jpg")
silentrename("/whirlwind/goes16/irc13m/fulldisk/latest_fulldisk_11.jpg", "/whirlwind/goes16/irc13m/fulldisk/latest_fulldisk_12.jpg")
silentrename("/whirlwind/goes16/irc13m/fulldisk/latest_fulldisk_10.jpg", "/whirlwind/goes16/irc13m/fulldisk/latest_fulldisk_11.jpg")
silentrename("/whirlwind/goes16/irc13m/fulldisk/latest_fulldisk_9.jpg", "/whirlwind/goes16/irc13m/fulldisk/latest_fulldisk_10.jpg")
silentrename("/whirlwind/goes16/irc13m/fulldisk/latest_fulldisk_8.jpg", "/whirlwind/goes16/irc13m/fulldisk/latest_fulldisk_9.jpg")
silentrename("/whirlwind/goes16/irc13m/fulldisk/latest_fulldisk_7.jpg", "/whirlwind/goes16/irc13m/fulldisk/latest_fulldisk_8.jpg")
silentrename("/whirlwind/goes16/irc13m/fulldisk/latest_fulldisk_6.jpg", "/whirlwind/goes16/irc13m/fulldisk/latest_fulldisk_7.jpg")
silentrename("/whirlwind/goes16/irc13m/fulldisk/latest_fulldisk_5.jpg", "/whirlwind/goes16/irc13m/fulldisk/latest_fulldisk_6.jpg")
silentrename("/whirlwind/goes16/irc13m/fulldisk/latest_fulldisk_4.jpg", "/whirlwind/goes16/irc13m/fulldisk/latest_fulldisk_5.jpg")
silentrename("/whirlwind/goes16/irc13m/fulldisk/latest_fulldisk_3.jpg", "/whirlwind/goes16/irc13m/fulldisk/latest_fulldisk_4.jpg")
silentrename("/whirlwind/goes16/irc13m/fulldisk/latest_fulldisk_2.jpg", "/whirlwind/goes16/irc13m/fulldisk/latest_fulldisk_3.jpg")
silentrename("/whirlwind/goes16/irc13m/fulldisk/latest_fulldisk_1.jpg", "/whirlwind/goes16/irc13m/fulldisk/latest_fulldisk_2.jpg")
    
shutil.copy(filename2, "/whirlwind/goes16/irc13m/fulldisk/latest_fulldisk_1.jpg")
shutil.copy(filename9, "/whirlwind/goes16/irc13m/fulldisk_full/latest_fulldisk_full_1.jpg")

quit()

os.remove("/whirlwind/goes16/vis/conus/latest_conus_72.jpg")
os.rename("/whirlwind/goes16/vis/conus/latest_conus_71.jpg", "/whirlwind/goes16/vis/conus/latest_conus_72.jpg")
os.rename("/whirlwind/goes16/vis/conus/latest_conus_70.jpg", "/whirlwind/goes16/vis/conus/latest_conus_71.jpg")
os.rename("/whirlwind/goes16/vis/conus/latest_conus_69.jpg", "/whirlwind/goes16/vis/conus/latest_conus_70.jpg")
os.rename("/whirlwind/goes16/vis/conus/latest_conus_68.jpg", "/whirlwind/goes16/vis/conus/latest_conus_69.jpg")
os.rename("/whirlwind/goes16/vis/conus/latest_conus_67.jpg", "/whirlwind/goes16/vis/conus/latest_conus_68.jpg")
os.rename("/whirlwind/goes16/vis/conus/latest_conus_66.jpg", "/whirlwind/goes16/vis/conus/latest_conus_67.jpg")
os.rename("/whirlwind/goes16/vis/conus/latest_conus_65.jpg", "/whirlwind/goes16/vis/conus/latest_conus_66.jpg")
os.rename("/whirlwind/goes16/vis/conus/latest_conus_64.jpg", "/whirlwind/goes16/vis/conus/latest_conus_65.jpg")
os.rename("/whirlwind/goes16/vis/conus/latest_conus_63.jpg", "/whirlwind/goes16/vis/conus/latest_conus_64.jpg")
os.rename("/whirlwind/goes16/vis/conus/latest_conus_62.jpg", "/whirlwind/goes16/vis/conus/latest_conus_63.jpg")
os.rename("/whirlwind/goes16/vis/conus/latest_conus_61.jpg", "/whirlwind/goes16/vis/conus/latest_conus_62.jpg")
os.rename("/whirlwind/goes16/vis/conus/latest_conus_60.jpg", "/whirlwind/goes16/vis/conus/latest_conus_61.jpg")
os.rename("/whirlwind/goes16/vis/conus/latest_conus_59.jpg", "/whirlwind/goes16/vis/conus/latest_conus_60.jpg")
os.rename("/whirlwind/goes16/vis/conus/latest_conus_58.jpg", "/whirlwind/goes16/vis/conus/latest_conus_59.jpg")
os.rename("/whirlwind/goes16/vis/conus/latest_conus_57.jpg", "/whirlwind/goes16/vis/conus/latest_conus_58.jpg")
os.rename("/whirlwind/goes16/vis/conus/latest_conus_56.jpg", "/whirlwind/goes16/vis/conus/latest_conus_57.jpg")
os.rename("/whirlwind/goes16/vis/conus/latest_conus_55.jpg", "/whirlwind/goes16/vis/conus/latest_conus_56.jpg")
os.rename("/whirlwind/goes16/vis/conus/latest_conus_54.jpg", "/whirlwind/goes16/vis/conus/latest_conus_55.jpg")
os.rename("/whirlwind/goes16/vis/conus/latest_conus_53.jpg", "/whirlwind/goes16/vis/conus/latest_conus_54.jpg")
os.rename("/whirlwind/goes16/vis/conus/latest_conus_52.jpg", "/whirlwind/goes16/vis/conus/latest_conus_53.jpg")
os.rename("/whirlwind/goes16/vis/conus/latest_conus_51.jpg", "/whirlwind/goes16/vis/conus/latest_conus_52.jpg")
os.rename("/whirlwind/goes16/vis/conus/latest_conus_50.jpg", "/whirlwind/goes16/vis/conus/latest_conus_51.jpg")
os.rename("/whirlwind/goes16/vis/conus/latest_conus_49.jpg", "/whirlwind/goes16/vis/conus/latest_conus_50.jpg")
os.rename("/whirlwind/goes16/vis/conus/latest_conus_48.jpg", "/whirlwind/goes16/vis/conus/latest_conus_49.jpg")
os.rename("/whirlwind/goes16/vis/conus/latest_conus_47.jpg", "/whirlwind/goes16/vis/conus/latest_conus_48.jpg")
os.rename("/whirlwind/goes16/vis/conus/latest_conus_46.jpg", "/whirlwind/goes16/vis/conus/latest_conus_47.jpg")
os.rename("/whirlwind/goes16/vis/conus/latest_conus_45.jpg", "/whirlwind/goes16/vis/conus/latest_conus_46.jpg")
os.rename("/whirlwind/goes16/vis/conus/latest_conus_44.jpg", "/whirlwind/goes16/vis/conus/latest_conus_45.jpg")
os.rename("/whirlwind/goes16/vis/conus/latest_conus_43.jpg", "/whirlwind/goes16/vis/conus/latest_conus_44.jpg")
os.rename("/whirlwind/goes16/vis/conus/latest_conus_42.jpg", "/whirlwind/goes16/vis/conus/latest_conus_43.jpg")
os.rename("/whirlwind/goes16/vis/conus/latest_conus_41.jpg", "/whirlwind/goes16/vis/conus/latest_conus_42.jpg")
os.rename("/whirlwind/goes16/vis/conus/latest_conus_40.jpg", "/whirlwind/goes16/vis/conus/latest_conus_41.jpg")
os.rename("/whirlwind/goes16/vis/conus/latest_conus_39.jpg", "/whirlwind/goes16/vis/conus/latest_conus_40.jpg")
os.rename("/whirlwind/goes16/vis/conus/latest_conus_38.jpg", "/whirlwind/goes16/vis/conus/latest_conus_39.jpg")
os.rename("/whirlwind/goes16/vis/conus/latest_conus_37.jpg", "/whirlwind/goes16/vis/conus/latest_conus_38.jpg")
os.rename("/whirlwind/goes16/vis/conus/latest_conus_36.jpg", "/whirlwind/goes16/vis/conus/latest_conus_37.jpg")
os.rename("/whirlwind/goes16/vis/conus/latest_conus_35.jpg", "/whirlwind/goes16/vis/conus/latest_conus_36.jpg")
os.rename("/whirlwind/goes16/vis/conus/latest_conus_34.jpg", "/whirlwind/goes16/vis/conus/latest_conus_35.jpg")
os.rename("/whirlwind/goes16/vis/conus/latest_conus_33.jpg", "/whirlwind/goes16/vis/conus/latest_conus_34.jpg")
os.rename("/whirlwind/goes16/vis/conus/latest_conus_32.jpg", "/whirlwind/goes16/vis/conus/latest_conus_33.jpg")
os.rename("/whirlwind/goes16/vis/conus/latest_conus_31.jpg", "/whirlwind/goes16/vis/conus/latest_conus_32.jpg")
os.rename("/whirlwind/goes16/vis/conus/latest_conus_30.jpg", "/whirlwind/goes16/vis/conus/latest_conus_31.jpg")
os.rename("/whirlwind/goes16/vis/conus/latest_conus_29.jpg", "/whirlwind/goes16/vis/conus/latest_conus_30.jpg")
os.rename("/whirlwind/goes16/vis/conus/latest_conus_28.jpg", "/whirlwind/goes16/vis/conus/latest_conus_29.jpg")
os.rename("/whirlwind/goes16/vis/conus/latest_conus_27.jpg", "/whirlwind/goes16/vis/conus/latest_conus_28.jpg")
os.rename("/whirlwind/goes16/vis/conus/latest_conus_26.jpg", "/whirlwind/goes16/vis/conus/latest_conus_27.jpg")
os.rename("/whirlwind/goes16/vis/conus/latest_conus_25.jpg", "/whirlwind/goes16/vis/conus/latest_conus_26.jpg")
os.rename("/whirlwind/goes16/vis/conus/latest_conus_24.jpg", "/whirlwind/goes16/vis/conus/latest_conus_25.jpg")
os.rename("/whirlwind/goes16/vis/conus/latest_conus_23.jpg", "/whirlwind/goes16/vis/conus/latest_conus_24.jpg")
os.rename("/whirlwind/goes16/vis/conus/latest_conus_22.jpg", "/whirlwind/goes16/vis/conus/latest_conus_23.jpg")
os.rename("/whirlwind/goes16/vis/conus/latest_conus_21.jpg", "/whirlwind/goes16/vis/conus/latest_conus_22.jpg")
os.rename("/whirlwind/goes16/vis/conus/latest_conus_20.jpg", "/whirlwind/goes16/vis/conus/latest_conus_21.jpg")
os.rename("/whirlwind/goes16/vis/conus/latest_conus_19.jpg", "/whirlwind/goes16/vis/conus/latest_conus_20.jpg")
os.rename("/whirlwind/goes16/vis/conus/latest_conus_18.jpg", "/whirlwind/goes16/vis/conus/latest_conus_19.jpg")
os.rename("/whirlwind/goes16/vis/conus/latest_conus_17.jpg", "/whirlwind/goes16/vis/conus/latest_conus_18.jpg")
os.rename("/whirlwind/goes16/vis/conus/latest_conus_16.jpg", "/whirlwind/goes16/vis/conus/latest_conus_17.jpg")
os.rename("/whirlwind/goes16/vis/conus/latest_conus_15.jpg", "/whirlwind/goes16/vis/conus/latest_conus_16.jpg")
os.rename("/whirlwind/goes16/vis/conus/latest_conus_14.jpg", "/whirlwind/goes16/vis/conus/latest_conus_15.jpg")
os.rename("/whirlwind/goes16/vis/conus/latest_conus_13.jpg", "/whirlwind/goes16/vis/conus/latest_conus_14.jpg")
os.rename("/whirlwind/goes16/vis/conus/latest_conus_12.jpg", "/whirlwind/goes16/vis/conus/latest_conus_13.jpg")
os.rename("/whirlwind/goes16/vis/conus/latest_conus_11.jpg", "/whirlwind/goes16/vis/conus/latest_conus_12.jpg")
os.rename("/whirlwind/goes16/vis/conus/latest_conus_10.jpg", "/whirlwind/goes16/vis/conus/latest_conus_11.jpg")
os.rename("/whirlwind/goes16/vis/conus/latest_conus_9.jpg", "/whirlwind/goes16/vis/conus/latest_conus_10.jpg")
os.rename("/whirlwind/goes16/vis/conus/latest_conus_8.jpg", "/whirlwind/goes16/vis/conus/latest_conus_9.jpg")
os.rename("/whirlwind/goes16/vis/conus/latest_conus_7.jpg", "/whirlwind/goes16/vis/conus/latest_conus_8.jpg")
os.rename("/whirlwind/goes16/vis/conus/latest_conus_6.jpg", "/whirlwind/goes16/vis/conus/latest_conus_7.jpg")
os.rename("/whirlwind/goes16/vis/conus/latest_conus_5.jpg", "/whirlwind/goes16/vis/conus/latest_conus_6.jpg")
os.rename("/whirlwind/goes16/vis/conus/latest_conus_4.jpg", "/whirlwind/goes16/vis/conus/latest_conus_5.jpg")
os.rename("/whirlwind/goes16/vis/conus/latest_conus_3.jpg", "/whirlwind/goes16/vis/conus/latest_conus_4.jpg")
os.rename("/whirlwind/goes16/vis/conus/latest_conus_2.jpg", "/whirlwind/goes16/vis/conus/latest_conus_3.jpg")
os.rename("/whirlwind/goes16/vis/conus/latest_conus_1.jpg", "/whirlwind/goes16/vis/conus/latest_conus_2.jpg")

shutil.copy(filename3, "/whirlwind/goes16/vis/conus/latest_conus_1.jpg")

