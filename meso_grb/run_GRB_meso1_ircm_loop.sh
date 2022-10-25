#!/bin/bash

export PATH="/home/poker/miniconda3/envs/goes16_201710/bin/:$PATH"
#sleep 42
cd /home/poker/goes16/meso_grb

cp /home/ldm/data/grb/meso/13/latest1.nc /dev/shm/latest_meso1_13.nc
cmp /home/ldm/data/grb/meso/13/latest1.nc /dev/shm/latest_meso1_13.nc > /dev/null
CONDITION=$?
#echo $CONDITION

while :; do

  until [ $CONDITION -eq 1 ] ; do
#     echo same
     sleep 5
     cmp /home/ldm/data/grb/meso/13/latest1.nc /dev/shm/latest_meso1_13.nc > /dev/null
     CONDITION=$?
  done

#  echo different
  cp /home/ldm/data/grb/meso/13/latest1.nc /dev/shm/latest_meso1_13.nc
  /home/poker/miniconda3/envs/goes16_201710/bin/python goes16_GRB_meso1_ircm.py /dev/shm/latest_meso1_13.nc
  cmp /home/ldm/data/grb/meso/13/latest1.nc /dev/shm/latest_meso1_13.nc > /dev/null
  CONDITION=$?
#  echo repeat

done





/home/poker/miniconda3/envs/goes16_201710/bin/python goes16_GRB_meso1_ircm.py /home/ldm/data/grb/meso/13/latest1.nc


