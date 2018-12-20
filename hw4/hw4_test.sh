wget -O emb.model 'https://www.dropbox.com/s/f7ejetmtthqmkod/emb.model?dl=1'
wget -O model.h5 'https://www.dropbox.com/s/xdsb6tzby2wc1y8/ml-hw4.h5?dl=1'
python3 hw4_test.py $1 $2 $3
