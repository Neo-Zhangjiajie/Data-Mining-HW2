python3 experiment.py -d red-wine-quality -bs 32 -s 1@16 -e 401 -lrde 200 -lr 0.002 -ki 0 -i 0 -wd 0.0001 --print_rule --save_best
python3 experiment.py -d housing_data -bs 32 -s 1@16 -e 1001 -lrde 200 -lr 0.002 -ki 0 -i 0 -wd 0.0001 --print_rule --save_best
python3 experiment.py -d online_news_popularity -bs 256 -s 1@16 -e 1001 -lrde 200 -lr 0.002 -ki 0 -i 0 -wd 0.0001 --print_rule --save_best