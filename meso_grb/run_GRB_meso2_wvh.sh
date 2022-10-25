#!/bin/bash

export PATH="/home/poker/miniconda3/envs/goes16_201710/bin/:$PATH"
#export time=`date -u "+%Y%m%d%H%M" -d "13 min ago"`
#export time=`ls -1 /weather/data/goes16/TIRU/14/*PAA.nc | awk '{$1 = substr($1,30,12)} 1' | sort -u | tail -2 | head -1`
#sleep 8
#export time=`ls -1 /home/ldm/data/grb/meso/08/OR_ABI-L1b-RadM2* | tail -1`

sleep 22

#echo $time

cd /home/poker/goes16/meso_grb

/home/poker/miniconda3/envs/goes16_201710/bin/python goes16_GRB_meso2_wvh.py /home/ldm/data/grb/meso/08/latest2.nc


