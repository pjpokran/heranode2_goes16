#!/bin/bash

export PATH="/home/poker/miniconda3/bin:$PATH"
#export time=`date -u "+%Y%m%d%H%M" -d "6 min ago"`
export time=`ls -1tr /home/ldm/data/grb/conus/02 | tail -1`

echo $time

cd /home/poker/goes16/conus_grb

/home/poker/miniconda3/envs/goes16_201710/bin/python goes16_GRB_conus_vis.py /home/ldm/data/grb/conus/02/$time


