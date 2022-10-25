#!/bin/bash

export PATH="/home/poker/miniconda3/envs/goes16_201710/bin/:$PATH"

cd /home/poker/goes16/meso_grb

cp /home/ldm/data/grb/meso/02/latest2.nc /dev/shm/latest_mesoswisc2_vis.nc
cp /home/ldm/data/grb/meso/02/latest1.nc /dev/shm/latest_mesoswisc1_vis.nc
cmp /home/ldm/data/grb/meso/02/latest2.nc /dev/shm/latest_mesoswisc2_vis.nc > /dev/null
CONDITION=$?
#echo $CONDITION

while :; do

  until [ $CONDITION -eq 1 ] ; do
#     echo same
     sleep 5
     cmp /home/ldm/data/grb/meso/02/latest2.nc /dev/shm/latest_mesoswisc2_vis.nc > /dev/null
     CONDITION=$?
  done

echo mesowisc2 is different - wait for meso1
  cp /home/ldm/data/grb/meso/02/latest2.nc /dev/shm/latest_mesoswisc2_vis.nc

export CONDITION=0
  until [ $CONDITION -eq 1 ] ; do
#     echo same
     sleep 5
     cmp /home/ldm/data/grb/meso/02/latest1.nc /dev/shm/latest_mesoswisc1_vis.nc > /dev/null
     CONDITION=$?
  done

echo mesowisc1 is different - plot now
  cp /home/ldm/data/grb/meso/02/latest1.nc /dev/shm/latest_mesoswisc1_vis.nc

  echo '### START python     at ' `date`
  /home/poker/miniconda3/envs/goes16_201710/bin/python goes16_GRB_meso_swisc_vis_sqrt.py /dev/shm/latest_mesoswisc1_vis.nc /dev/shm/latest_mesoswisc2_vis.nc
  echo '### START python #2  at ' `date`
  /home/poker/miniconda3/envs/goes16_201710/bin/python goes16_GRB_meso_swisc_vis.py /dev/shm/latest_mesoswisc1_vis.nc /dev/shm/latest_mesoswisc2_vis.nc
  echo '### DONE  python     at ' `date`
  cmp /home/ldm/data/grb/meso/02/latest2.nc /dev/shm/latest_mesoswisc2_vis.nc > /dev/null
  CONDITION=$?
#  echo repeat

done




