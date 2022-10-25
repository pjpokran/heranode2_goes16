#!/bin/bash

export PATH="/home/poker/miniconda3/envs/goes16_201710/bin/:$PATH"
#export time=`date -u "+%Y%m%d%H%M" -d "13 min ago"`
#export time=`ls -1 /weather/data/goes16/TIRU/14/*PAA.nc | awk '{$1 = substr($1,30,12)} 1' | sort -u | tail -2 | head -1`
#sleep 4
#export time=`ls -1 /home/ldm/data/grb/meso/09/OR_ABI-L1b-RadM1* | tail -1`

sleep 44

#echo $time

cd /home/poker/goes16/meso_grb

/home/poker/miniconda3/envs/goes16_201710/bin/python goes16_GRB_meso1_wvc.py /home/ldm/data/grb/meso/09/latest1.nc


