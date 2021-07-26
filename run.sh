# #!/bin/bash

# echo "########## TWITTER-PPDB ##########"
# python  train-models-on-ppdb.py  twitter-ppdb  roberta  roberta-large  models/twitter-ppdb/roberta-large/
python  test-models.py  roberta  models/twitter-ppdb/roberta-large/  models/predictions/twitter-ppdb/roberta-large/
# python  test-models-on-app.py  roberta  models/twitter-ppdb/roberta-large/  models/predictions/twitter-ppdb/roberta-large/

# echo "########## TWITTER-PPDB + APP ##########"
# python  train-models-on-ppdb.py  twitter-ppdb  roberta  roberta-large  models/twitter-ppdb-app/roberta-large/  app/train
# python  test-models-on-app.py  roberta  models/twitter-ppdb-app/roberta-large/  models/predictions/twitter-ppdb-app/roberta-large/

# echo "########## TWITTER-PPDB + NAP-MSRP##########"
# python  train-models-on-ppdb.py  twitter-ppdb  roberta  roberta-large  models/twitter-ppdb-nap-msrp/roberta-large/  nap/msrp1/msrp1
# python  test-models-on-app.py  roberta  models/twitter-ppdb-nap-msrp/roberta-large/  models/predictions/twitter-ppdb-nap-msrp/roberta-large/

# # python  train-models-on-ppdb.py  twitter-ppdb  roberta  roberta-large  models/twitter-ppdb-nap-msrp-balanced/roberta-large/  nap/msrp1/msrp1  balanced
# # python  test-models-on-app.py  roberta  models/twitter-ppdb-nap-msrp-balanced/roberta-large/  models/predictions/twitter-ppdb-nap-msrp-balanced/roberta-large/

# # python  train-models-on-ppdb.py  twitter-ppdb  roberta  roberta-large  models/twitter-ppdb-nap-msrp-equal/roberta-large/  nap/msrp1/msrp1-apt,nap/msrp1/msrp1-nmi  balanced
# # python  test-models-on-app.py  roberta  models/twitter-ppdb-nap-msrp-equal/roberta-large/  models/predictions/twitter-ppdb-nap-msrp-equal/roberta-large/

# echo "########## TWITTER-PPDB + APP + NAP-MSRP ##########"
# python  train-models-on-ppdb.py  twitter-ppdb  roberta  roberta-large  models/twitter-ppdb-app-nap-msrp/roberta-large/  app/train,nap/msrp1/msrp1
# python  test-models-on-app.py  roberta  models/twitter-ppdb-app-nap-msrp/roberta-large/  models/predictions/twitter-ppdb-app-nap-msrp/roberta-large/


# echo "########## TWITTER-PPDB + NAP-TWITTERPPDB ##########"
python  train-models-on-ppdb.py  twitter-ppdb  roberta  models/twitter-ppdb-nap-twitterppdb/roberta-large/  models/twitter-ppdb-nap-twitterppdb/roberta-large/  nap/twitterppdb1/twitterppdb1
python  test-models-on-app.py  roberta  models/twitter-ppdb-nap-twitterppdb/roberta-large/  models/predictions/twitter-ppdb-nap-twitterppdb/roberta-large/

# # python  train-models-on-ppdb.py  twitter-ppdb  roberta  roberta-large  models/twitter-ppdb-nap-twitterppdb-balanced/roberta-large/  nap/twitterppdb1/twitterppdb1  balanced
# # python  test-models-on-app.py  roberta  models/twitter-ppdb-nap-twitterppdb-balanced/roberta-large/  models/predictions/twitter-ppdb-nap-twitterppdb-balanced/roberta-large/

# # python  train-models-on-ppdb.py  twitter-ppdb  roberta  roberta-large  models/twitter-ppdb-nap-twitterppdb-equal/roberta-large/  nap/twitterppdb1/twitterppdb1-apt,nap/twitterppdb1/twitterppdb1-nmi  balanced
# # python  test-models-on-app.py  roberta  models/twitter-ppdb-nap-twitterppdb-equal/roberta-large/  models/predictions/twitter-ppdb-nap-twitterppdb-equal/roberta-large/

# echo "########## TWITTER-PPDB + APP + NAP-TWITTERPPDB ##########"
# python  train-models-on-ppdb.py  twitter-ppdb  roberta  roberta-large  models/twitter-ppdb-app-nap-twitterppdb/roberta-large/  app/train,nap/twitterppdb1/twitterppdb1
# python  test-models-on-app.py  roberta  models/twitter-ppdb-app-nap-twitterppdb/roberta-large/  models/predictions/twitter-ppdb-app-nap-twitterppdb/roberta-large/

# echo "########## TWITTER-PPDB + NAP-MSRP + NAP-TWITTERPPDB ##########"
# python  train-models-on-ppdb.py  twitter-ppdb  roberta  roberta-large  models/twitter-ppdb-nap-msrp-nap-twitterppdb/roberta-large/  nap/msrp1/msrp1,nap/twitterppdb1/twitterppdb1
# python  test-models-on-app.py  roberta  models/twitter-ppdb-nap-msrp-nap-twitterppdb/roberta-large/  models/predictions/twitter-ppdb-nap-msrp-nap-twitterppdb/roberta-large/

# echo "########## TWITTER-PPDB + APP + NAP-MSRP + NAP-TWITTERPPDB ##########"
# python  train-models-on-ppdb.py  twitter-ppdb  roberta  roberta-large  models/twitter-ppdb-app-nap-msrp-nap-twitterppdb/roberta-large/  app/train,nap/msrp1/msrp1,nap/twitterppdb1/twitterppdb1
# python  test-models-on-app.py  roberta  models/twitter-ppdb-app-nap-msrp-nap-twitterppdb/roberta-large/  models/predictions/twitter-ppdb-app-nap-msrp-nap-twitterppdb/roberta-large/



# # for i in 0 1 2
# # do
# #     python  nap_generation.py  0  twitter-ppdb1  3  $i &
# # done