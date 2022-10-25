#!/bin/bash

export PATH="/home/poker/miniconda3/envs/goes16_201710/bin/:$PATH"

export LD_LIBRARY_PATH=


cd /home/poker/goes16/conusc_multispectral

cp /home/ldm/data/grb/conus/13/latest.nc /dev/shm/latest_daycloud_13.nc
cmp /home/ldm/data/grb/conus/13/latest.nc /dev/shm/latest_daycloud_13.nc > /dev/null
CONDITION=$?
#echo $CONDITION

while :; do

  until [ $CONDITION -eq 1 ] ; do
#     echo same
     sleep 5
     cmp /home/ldm/data/grb/conus/13/latest.nc /dev/shm/latest_daycloud_13.nc > /dev/null
     CONDITION=$?
  done
#  echo different
  sleep 25
  cp /home/ldm/data/grb/conus/13/latest.nc /dev/shm/latest_daycloud_13.nc
  /home/poker/miniconda3/envs/goes16_201710/bin/python goes16_conus_GRB_DayCloud.py /home/ldm/data/grb/conus/13/latest.nc /home/ldm/data/grb/conus/02/latest.nc /home/ldm/data/grb/conus/05/latest.nc
  cmp /home/ldm/data/grb/conus/13/latest.nc /dev/shm/latest_daycloud_13.nc > /dev/null
  CONDITION=$?
#  echo repeat

done

