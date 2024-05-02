#!/bin/bash
Qmode="0.0 0.0 0.0"
newdir="../MgB2_Bands_Q-0-0-0"
mkdir $newdir
for ((i=1; i<=9;i++))
do
mkdir $newdir/frphon_mode$i
sed -i "/MODULATION/c\MODULATION = 2 2 2,  $Qmode  $i  0.5   0.0" ./modulation.conf
phonopy modulation.conf
mv MPOSCAR* $newdir/frphon_mode$i/.
cp modulation.conf $newdir/frphon_mode$i/.
done
