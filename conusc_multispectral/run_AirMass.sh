#!/bin/bash

export PATH="/home/poker/miniconda3/envs/goes16_201710/bin/:$PATH"

export LD_LIBRARY_PATH=


cd /home/poker/goes16/conusc_multispectral

cp /home/ldm/data/grb/conus/12/latest.nc /dev/shm/latest_airmass_12.nc
cmp /home/ldm/data/grb/conus/12/latest.nc /dev/shm/latest_airmass_12.nc > /dev/null
CONDITION=$?
#echo $CONDITION

while :; do

  until [ $CONDITION -eq 1 ] ; do
#     echo same
     sleep 5
     cmp /home/ldm/data/grb/conus/12/latest.nc /dev/shm/latest_airmass_12.nc > /dev/null
     CONDITION=$?
  done
#  echo different
  cp /home/ldm/data/grb/conus/12/latest.nc /dev/shm/latest_airmass_12.nc
  sleep 45
  /home/poker/miniconda3/envs/goes16_201710/bin/python goes16_conus_GRB_AirMass.py /home/ldm/data/grb/conus/08/latest.nc  /home/ldm/data/grb/conus/10/latest.nc  /home/ldm/data/grb/conus/12/latest.nc /home/ldm/data/grb/conus/13/latest.nc
  cmp /home/ldm/data/grb/conus/12/latest.nc /dev/shm/latest_airmass_12.nc > /dev/null
  CONDITION=$?
#  echo repeat

done

