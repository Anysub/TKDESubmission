#!/bin/bash

# run.sh

# --
#

lr="0.01 0.003 0.001"
prep="node_embedding"
aggr="attention edge_emb_attn"
edge="identity residual attention"

count="10"
for l in $lr; do
for p in $prep; do
for a in $aggr; do
for e in $edge; do
python3 ./train.py \
    --problem-path ../../data/freebase/ \
    --problem yago \
    --epochs 1000 \
    --batch-size 2048 \
    --lr-init $l \
    --lr-schedule constant\
    --dropout 0.5\
    --batchnorm\
    --prep-class $p \
    --edgeupt-class $e \
    --aggregator-class $a \
    --log-interval 1\
    > "experiment/freebase/fb_"$count".txt" 
let count++
done
done
done
done

