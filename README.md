###### CLING

A semisupervised learning model for heterogeneous information networks (HINs)

#### Dependency
ujson == 1.35
pytorch >= 1.5.0

#### Preprocess
0. (DBLP dataset only) Download Glove word embdding: http://nlp.stanford.edu/data/glove.840B.300d.zip
1. Generate node embeddings by node2vec or metapath2vec:
    a.    preprocess/yelp.py -> gen_homograph()
          preprocess/main.py --input ../../data/yelp/homograph.txt --output ../../data/ yelp/RUBK_128.emb --dimensions 128 --workers 56 --walk-length 100            --num-walks 40 --window-size 5
    b.    preprocess/yelp.py -> gen_walk(path='../data/yelp/',                                      walk_length=100,n_walks=40)
          preprocess/metapath2vec -train ../../data/yelp/BRKRB.walk -output ../../data/yelp/BRKRB_128.emb -size 128 -threads 40

2. Fuse edge features:
    preprocess/yelp.py -> dump_yelp_edge_emb(path='../data/yelp/')

3. Compute index for GraphSage-like minibatch sampling strategy:
    preprocess/yelp.py -> gen_edge_adj_random(path='../data/yelp/',edge_dim=130)

#### Run
driver: train.py

scripts for running 10 random seeds: run_yelp.py
options listed in the run_***.py file.

#### LICENSE
MIT

