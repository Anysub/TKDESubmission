#!/usr/bin/env bash
#python3 ./main.py  --input ..\..\data\cora\homograph.txt --output ..\..\data\cora\AP_128.emb --dimensions 128 --workers 56 --num-walks 1000 --window-size 10 --walk-length 100
#python3 ./main.py  --input ../../data/freebase/homograph.txt --output ../../data/freebase/MADW_8.emb --dimensions 8 --workers 56
# python3 ./main.py  --input ../../data/dblp2/homograph.txt --output ../../data/dblp2/apc_128.emb --dimensions 128 --workers 56 --walk-length 100 --num-walks 40 --window-size 5
python3 ./main.py  --input ../../data/yelp/homograph.txt --output ../../data/yelp/rubk_128.emb --dimensions 128 --workers 56 --walk-length 100 --num-walks 40 --window-size 5
# python3 ./main.py  --input ../../data/freebase/homograph.txt --output ../../data/freebase/madw_128.emb --dimensions 128 --workers 56 --walk-length 100 --num-walks 40 --window-size 5
