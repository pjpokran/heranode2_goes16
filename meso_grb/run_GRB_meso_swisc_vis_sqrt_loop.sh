#!/bin/bash

export PATH="/home/poker/miniconda3/envs/goes16_201710/bin/:$PATH"

cd /home/poker/goes16/meso_grb

while :; do

  for file in /home/ldm/data/grb-east-mesotemp/02swi/*RadM2*
  do
#  if [ ! -z "$file" ]
  if [ $file == "/home/ldm/data/grb-east-mesotemp/02swi/*RadM2*" ]
  then
    echo NO FILES
  else
# OR_ABI-L1b-RadM2-M6C07_G16_s20210610323524_e20210610323593_c20210610324025.nc
#      echo process $file
      echo "file  is" $file
      part1=${file:0:54}
      part2=${file:56:22}
      f2=${part1}1-
      f2+=$part2
      file2=("${f2}"*)
      echo file2 is $file2
#      /home/poker/miniconda3/envs/goes16_201710/bin/python goes16_GRB_meso_swisc_ircm.py $file2 $file
#      /home/poker/miniconda3/envs/goes16_201710/bin/python goes16_GRB_meso_swisc_swir.py $file2 $file
  echo '### START python     at ' `date`
  /home/poker/miniconda3/envs/goes16_201710/bin/python goes16_GRB_meso_swisc_vis_sqrt.py $file2 $file
  echo '### START python #2  at ' `date`
  /home/poker/miniconda3/envs/goes16_201710/bin/python goes16_GRB_meso_swisc_vis.py $file2 $file
      /bin/rm $file2 $file

#      /home/poker/miniconda3/envs/goes16_201710/bin/python goes16_GRB_meso1_ircm.py $file
#      /bin/rm $file
  fi
  done

echo sleep 5
sleep 5

done

#cp /home/ldm/data/grb/meso/02/latest2.nc /dev/shm/latest_mesoswisc2_vis.nc
#cp /home/ldm/data/grb/meso/02/latest1.nc /dev/shm/latest_mesoswisc1_vis.nc
#cmp /home/ldm/data/grb/meso/02/latest2.nc /dev/shm/latest_mesoswisc2_vis.nc > /dev/null
#CONDITION=$?
##echo $CONDITION
#
#while :; do
#
#  until [ $CONDITION -eq 1 ] ; do
##     echo same
#     sleep 5
#     cmp /home/ldm/data/grb/meso/02/latest2.nc /dev/shm/latest_mesoswisc2_vis.nc > /dev/null
#     CONDITION=$?
#  done
#
##  echo different
#  cp /home/ldm/data/grb/meso/02/latest2.nc /dev/shm/latest_mesoswisc2_vis.nc
#  cp /home/ldm/data/grb/meso/02/latest1.nc /dev/shm/latest_mesoswisc1_vis.nc
#  echo '### START python     at ' `date`
#  /home/poker/miniconda3/envs/goes16_201710/bin/python goes16_GRB_meso_swisc_vis_sqrt.py /dev/shm/latest_mesoswisc1_vis.nc /dev/shm/latest_mesoswisc2_vis.nc
#  echo '### START python #2  at ' `date`
#  /home/poker/miniconda3/envs/goes16_201710/bin/python goes16_GRB_meso_swisc_vis.py /dev/shm/latest_mesoswisc1_vis.nc /dev/shm/latest_mesoswisc2_vis.nc
#  echo '### DONE  python     at ' `date`
#  cmp /home/ldm/data/grb/meso/02/latest2.nc /dev/shm/latest_mesoswisc2_vis.nc > /dev/null
#  CONDITION=$?
##  echo repeat
#
#done




