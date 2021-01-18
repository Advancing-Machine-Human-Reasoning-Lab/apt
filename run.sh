#!/bin/bash

# python  train-models-on-ppdb.py  twitter-ppdb  roberta  roberta-base  models/twitter-ppdb/roberta-base/
# python  test-models-on-app.py  roberta  models/twitter-ppdb/roberta-base/  models/predictions/twitter-ppdb/roberta-base/

# python  train-models-on-ppdb.py  twitter-ppdb  roberta  roberta-base  models/twitter-ppdb-app/roberta-base/  app/train
# python  test-models-on-app.py  roberta  models/twitter-ppdb-app/roberta-base/  models/predictions/twitter-ppdb-app/roberta-base/

# python  train-models-on-ppdb.py  twitter-ppdb  roberta  roberta-base  models/twitter-ppdb-nap-msrp/roberta-base/  nap/msrp1/msrp1
# python  test-models-on-app.py  roberta  models/twitter-ppdb-nap-msrp/roberta-base/  models/predictions/twitter-ppdb-nap-msrp/roberta-base/

# python  train-models-on-ppdb.py  twitter-ppdb  roberta  roberta-base  models/twitter-ppdb-nap-msrp-balanced/roberta-base/  nap/msrp1/msrp1  balanced
# python  test-models-on-app.py  roberta  models/twitter-ppdb-nap-msrp-balanced/roberta-base/  models/predictions/twitter-ppdb-nap-msrp-balanced/roberta-base/

# python  train-models-on-ppdb.py  twitter-ppdb  roberta  roberta-base  models/twitter-ppdb-nap-msrp-equal/roberta-base/  nap/msrp1/msrp1-apt,nap/msrp1/msrp1-nmi  balanced
# python  test-models-on-app.py  roberta  models/twitter-ppdb-nap-msrp-equal/roberta-base/  models/predictions/twitter-ppdb-nap-msrp-equal/roberta-base/

# python  train-models-on-ppdb.py  twitter-ppdb  roberta  roberta-base  models/twitter-ppdb-app-nap-msrp/roberta-base/  app/train,nap/msrp1/msrp1
# python  test-models-on-app.py  roberta  models/twitter-ppdb-app-nap-msrp/roberta-base/  models/predictions/twitter-ppdb-app-nap-msrp/roberta-base/


# python  train-models-on-ppdb.py  twitter-ppdb  roberta  roberta-base  models/twitter-ppdb-nap-twitterppdb/roberta-base/  nap/twitterppdb1/twitterppdb1
# python  test-models-on-app.py  roberta  models/twitter-ppdb-nap-twitterppdb/roberta-base/  models/predictions/twitter-ppdb-nap-twitterppdb/roberta-base/

# python  train-models-on-ppdb.py  twitter-ppdb  roberta  roberta-base  models/twitter-ppdb-nap-twitterppdb-balanced/roberta-base/  nap/twitterppdb1/twitterppdb1  balanced
# python  test-models-on-app.py  roberta  models/twitter-ppdb-nap-twitterppdb-balanced/roberta-base/  models/predictions/twitter-ppdb-nap-twitterppdb-balanced/roberta-base/

# python  train-models-on-ppdb.py  twitter-ppdb  roberta  roberta-base  models/twitter-ppdb-nap-twitterppdb-equal/roberta-base/  nap/twitterppdb1/twitterppdb1-apt,nap/twitterppdb1/twitterppdb1-nmi  balanced
# python  test-models-on-app.py  roberta  models/twitter-ppdb-nap-twitterppdb-equal/roberta-base/  models/predictions/twitter-ppdb-nap-twitterppdb-equal/roberta-base/

# python  train-models-on-ppdb.py  twitter-ppdb  roberta  roberta-base  models/twitter-ppdb-app-nap-twitterppdb/roberta-base/  app/train,nap/twitterppdb1/twitterppdb1
# python  test-models-on-app.py  roberta  models/twitter-ppdb-app-nap-twitterppdb/roberta-base/  models/predictions/twitter-ppdb-app-nap-twitterppdb/roberta-base/

# python  train-models-on-ppdb.py  twitter-ppdb  roberta  roberta-base  models/twitter-ppdb-nap-msrp-nap-twitterppdb/roberta-base/  nap/msrp1/msrp1,nap/twitterppdb1/twitterppdb1
# python  test-models-on-app.py  roberta  models/twitter-ppdb-nap-msrp-nap-twitterppdb/roberta-base/  models/predictions/twitter-ppdb-nap-msrp-nap-twitterppdb/roberta-base/

python  train-models-on-ppdb.py  twitter-ppdb  roberta  roberta-base  models/twitter-ppdb-app-nap-msrp-nap-twitterppdb/roberta-base/  app/train,nap/msrp1/msrp1,nap/twitterppdb1/twitterppdb1
python  test-models-on-app.py  roberta  models/twitter-ppdb-app-nap-msrp-nap-twitterppdb/roberta-base/  models/predictions/twitter-ppdb-app-nap-msrp-nap-twitterppdb/roberta-base/



# for i in 0 1 2
# do
#     python  nap_generation.py  0  twitter-ppdb1  3  $i &
# done