#!/bin/bash

cd ..
cd PanoHead
python gen_samples_exp.py --outdir=/data/hwkwak/sketch2head/data/real_pic --trunc=0.7 --shapes=false --seeds=0-9999 --network models/easy-khair-180-gpc0.8-trans10-025000.pkl  --reload_modules True

cd ..
cd informative-drawings
python test.py --name opensketch_style --dataroot ../sketch2head/data/ --purpose sketch