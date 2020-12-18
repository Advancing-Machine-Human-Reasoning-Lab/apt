#!/bin/bash

# python  train-models-on-ppdb.py  twitter-ppdb  roberta  roberta-base  models/twitter-ppdb/roberta-base/
# python  train-models-on-ppdb.py  twitter-ppdb  roberta  roberta-base  models/twitter-ppdb-app/roberta-base/  True

# python  test-models-on-app.py  roberta  models/twitter-ppdb/roberta-base/  models/predictions/twitter-ppdb/roberta-base/
# python  test-models-on-app.py  roberta  models/twitter-ppdb-app/roberta-base/  models/predictions/twitter-ppdb-app/roberta-base/

for i in 0 1 2
do
    python  nap_generation.py  0  ppnmt1  3  $i &
done