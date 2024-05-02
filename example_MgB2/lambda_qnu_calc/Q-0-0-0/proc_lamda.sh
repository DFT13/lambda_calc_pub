#!/bin/bash
width=99
DensofStates=5.6774
for ((i=1; i<=9;i++))
do

  python ../../../lamda_calc_band_v0-2.py \
  --eqfile ../scell_2x2x2/data-file-schema_mgb2_2x2x2prim.xml \
  --phonfile data-file-schema_mgb2_q-0-0-0_band${i}.xml \
  --NEf=${DensofStates} \
  --window=${width}

bndnum=$(printf "%02d" $i)
mv outlamda1.csv ${width}meV_outlamda1_band${bndnum}.csv
done

paste -d ',' outsmear.csv *outlamda1_band* >combined_outlamda1_${width}meV.csv
mkdir analysis_${width}meV
mv ${width}meV_outlamda* analysis_${width}meV/.
