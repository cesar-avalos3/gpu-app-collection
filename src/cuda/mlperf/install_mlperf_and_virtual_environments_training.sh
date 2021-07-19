
##################################
#   Install the training part    #
##################################

set -e

mkdir mlperf_training
cd mlperf_training
git clone https://github.com/mlcommons/training.git

##############################
# RNN Translator
##############################

cd rnn_translator
wget https://gist.githubusercontent.com/vsajip/4673395/raw/0504ce930e6dc6b02e4955a07d91ad462e0ba80b/pyvenvex.py

python3 pyvenvex.py virtual_environment
. virtual_environment/bin/activate

set +e

pip install torch===1.7.1+cu110 -f https://download.pytorch.org/whl/torch_stable.html
pip install sacrebleu

. download_dataset.sh

# To run this benchmark
# python 

deactivate

cd ..

set -e

##############################
# BERT
##############################

# 365 GB of data
# See if this is a problem
cd language_model/tensorflow/bert
wget https://gist.githubusercontent.com/vsajip/4673395/raw/0504ce930e6dc6b02e4955a07d91ad462e0ba80b/pyvenvex.py

python3 pyvenvex.py virtual_environment
. virtual_environment/bin/activate

set +e
pip install nvidia-pyindex
pip install nvidia-tensorflow

cd cleanup_scripts
mkdir wiki
cd wiki
wget https://dumps.wikimedia.org/enwiki/20200101/enwiki-20200101-pages-articles-multistream.xml.bz2
bzip2 -d enwiki-20200101-pages-articles-multistream.xml.bz2
cd ..
git clone https://github.com/attardi/wikiextractor.git
python wikiextractor/WikiExtractor.py wiki/enwiki-20200101-pages-articles-multistream.xml
. process_wiki '<text/*/wiki_??'
python extract_test_set_articles.py


deactivate

cd ..

