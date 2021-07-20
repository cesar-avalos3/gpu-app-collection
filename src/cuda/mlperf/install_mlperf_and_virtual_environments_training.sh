
##################################
#   Install the training part    #
##################################

echo "######################################"
echo "#     Installing MLPerf training     #"
echo "######################################"

set -e

mkdir mlperf_training
cd mlperf_training
git clone https://github.com/mlcommons/training.git

BASE_MLPERF_DIR=$PWD/training

cd training

##############################
# RNN Translator
##############################

RNN_DEPEND=$BASE_MLPERF_INFERENCE_DIR/rnn_training_depend.done

if [ -e "$BASE_MLPERF_DIR/virtual_environment_pytorch/bin/activate" ]; then
    echo "Pytorch virtual env. exists, skipping"
else
    cd $BASE_MLPERF_DIR
    wget https://gist.githubusercontent.com/vsajip/4673395/raw/0504ce930e6dc6b02e4955a07d91ad462e0ba80b/pyvenvex.py

    python3 pyvenvex.py virtual_environment
    . virtual_environment/bin/activate
fi

RNN_DEPEND=$BASE_MLPERF_INFERENCE_DIR/rnn_training_depend.done

if [ -f "$RNN_DEPEND" ]; then
    echo "RNN Translation dependencies fulfilled, skipping"
else
    set +e
    pip install torch===1.7.1+cu110 -f https://download.pytorch.org/whl/torch_stable.html
    pip install sacrebleu

    . download_dataset.sh
    echo "RNN Translation dependencies fulfilled" > $RNN_DEPEND
fi

deactivate

cd ..

##############################
# BERT
##############################

# 365 GB of data
if [[ -z "${DOWNLOAD_BIG_BERT_TRAINING_DATASET}"]]; then
    cd $BASE_MLPERF_DIR
    set -e

    wget https://gist.githubusercontent.com/vsajip/4673395/raw/0504ce930e6dc6b02e4955a07d91ad462e0ba80b/pyvenvex.py

    python3 pyvenvex.py virtual_environment_tensorflow_1_15
    . virtual_environment_tensorflow_1_15/bin/activate

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
fi

deactivate

cd ..

