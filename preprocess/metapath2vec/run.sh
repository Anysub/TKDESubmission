#!/bin/bash

# ./metapath2vec -train ../../data/freebase/MAM.walk -output ../../data/freebase/MAM_64.emb -size 64 -threads 40 
# ./metapath2vec -train ../../data/freebase/MDM.walk -output ../../data/freebase/MDM_64.emb -size 64 -threads 40 
# ./metapath2vec -train ../../data/freebase/MWM.walk -output ../../data/freebase/MWM_64.emb -size 64 -threads 40 

# ./metapath2vec -train ../../data/freebase/MAM.walk -output ../../data/freebase/MAM_32.emb -size 32 -threads 40 
# ./metapath2vec -train ../../data/freebase/MDM.walk -output ../../data/freebase/MDM_32.emb -size 32 -threads 40 
# ./metapath2vec -train ../../data/freebase/MWM.walk -output ../../data/freebase/MWM_32.emb -size 32 -threads 40 

# ./metapath2vec -train ../../data/freebase/MAM.walk -output ../../data/freebase/MAM_256.emb -size 256 -threads 40 
# ./metapath2vec -train ../../data/freebase/MDM.walk -output ../../data/freebase/MDM_256.emb -size 256 -threads 40 
# ./metapath2vec -train ../../data/freebase/MWM.walk -output ../../data/freebase/MWM_256.emb -size 256 -threads 40 

# ./metapath2vec -train ../../data/dblp2/APA.walk -output ../../data/dblp2/apa_128.emb -size 128 -threads 40 --window 5
./metapath2vec -train ../../data/dblp2/APCPA.walk -output ../../data/dblp2/apcpa_128.emb -size 128 -threads 40 --window 5