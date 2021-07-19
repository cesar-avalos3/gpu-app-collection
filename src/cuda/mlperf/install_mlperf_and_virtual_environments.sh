#Clone mlcommons inference repo

echo "These workloads are to be used as performance reference only."

# exit when any command fails
set -e

if  test -e "$PWD/mlperf_inference"; then
    echo "mlperf inference directory exists, skipping this part"
else
    echo $PWD/mlperf_inference
    mkdir mlperf_inference
    cd mlperf_inference
    git clone https://github.com/mlcommons/inference.git
fi

cd $PWD/mlperf_inference/inference
BASE_MLPERF_INFERENCE_DIR=$PWD

if [ -f "$PWD/virtual_environment/bin/activate" ]; then
    echo "Virtual environment exists; skipping this part"
else
    wget https://gist.githubusercontent.com/vsajip/4673395/raw/0504ce930e6dc6b02e4955a07d91ad462e0ba80b/pyvenvex.py
    #################################################
    # Single Pytorch environment for all workloads  #
    #################################################
    python3 pyvenvex.py virtual_environment
fi

. virtual_environment/bin/activate

cd language/bert
#python --version

#################################################
# At this point we are in the virtual environment
#################################################

# BERT
 
BERT_DEPEND=$BASE_MLPERF_INFERENCE_DIR/bert_inference_depend.done

if [ -f "$BERT_DEPEND" ]; then
    echo "BERT dependencies fulfilled, skipping"
else
    # First install loadgen
    set +e
    cd ../../
    git submodule update --init third_party/pybind
    cd loadgen
    pip install absl-py numpy 	
    python setup.py bdist_wheel  
    pip install --force-reinstall dist/mlperf_loadgen-*.whl

    # Go back to bert, install dependencies and make
    cd ../language/bert
    pip install onnx==1.8.1 transformers==2.4.0 onnxruntime==1.2.0 numpy==1.18.0
    pip install torch===1.7.1+cu110 torchvision===0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
    pip install nvidia-pyindex
    pip install nvidia-tensorflow
    make setup
    echo "BERT dependencies fulfilled" > $BERT_DEPEND
fi 

# RESNET

cd $BASE_MLPERF_INFERENCE_DIR/vision/classification_and_detection

RESNET_DEPEND=$BASE_MLPERF_INFERENCE_DIR/resnet_inference_depend.done

if [ -f "$RESNET_DEPEND" ]; then
    echo "RESNET dependencies fulfilled, skipping"
else
    wget https://zenodo.org/record/4588417/files/resnet50-19c8e357.pth .
    wget https://zenodo.org/record/2535873/files/resnet50_v1.pb .
    wget https://zenodo.org/record/4589637/files/resnet50_INT8bit_quantized.pt .
    wget https://zenodo.org/record/2592612/files/resnet50_v1.onnx .
    pip install opencv-python
    pip install pycocotools
    pip install future
    python setup.py install
    echo "RESNET dependencies fulfilled" > $RESNET_DEPEND
fi

# 3DUNET

cd $BASE_MLPERF_INFERENCE_DIR/vision/medical_imaging/3d-unet-brats19/

UNET_DEPEND=$BASE_MLPERF_INFERENCE_DIR/unet_inference_depend.done

if [ -f "$UNET_DEPEND" ]; then
    echo "RESNET dependencies fulfilled, skipping"
else
    if [[ -z "${DOWNLOAD_DATA_DIR}"]]; then
        make setup
        make preprocess_data
        echo "3DUNET dependencies fulfilled" > $UNET_DEPEND
    else
        echo "######################################################################"
        echo "3d-Unet requires the BRATS dataset to be set via the DOWNLOAD_DATA_DIR env. variable"
        echo "######################################################################"        
    fi
fi

# SSD

cd $BASE_MLPERF_INFERENCE_DIR/vision/classification_and_detection

SSD_DEPEND=$BASE_MLPERF_INFERENCE_DIR/ssd_inference_depend.done

if [ -f "$SSD_DEPEND" ]; then
    echo "SSD dependencies fulfilled, skipping"
else
    mkdir cocos
    cd cocos
    mkdir cocos-full
    mkdir cocos-300-300
    mkdir cocos-1200-1200
    cd cocos-full
    wget http://images.cocodataset.org/zips/val2017.zip
    wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
    unzip annotations_trainval2017.zip
    unzip val2017.zip
    cd $BASE_MLPERF_INFERENCE_DIR/tools/upscale_coco
    python upscale_coco.py --inputs $BASE_MLPERF_INFERENCE_DIR/vision/classification_and_detection/cocos/cocos-full --outputs $BASE_MLPERF_INFERENCE_DIR/vision/classification_and_detection/cocos/cocos-300-300 --size 300 300 --format jpg
    python upscale_coco.py --inputs $BASE_MLPERF_INFERENCE_DIR/vision/classification_and_detection/cocos/cocos-full --outputs $BASE_MLPERF_INFERENCE_DIR/vision/classification_and_detection/cocos/cocos-1200-1200 --size 1200 1200 --format jpg
    cd $BASE_MLPERF_INFERENCE_DIR/vision/classification_and_detection

fi

deactivate

: '
##########################################
# Setup the virtual environment for GNMT #
##########################################
set -e

cd ../../translation/gnmt/tensorflow

wget https://gist.githubusercontent.com/vsajip/4673395/raw/0504ce930e6dc6b02e4955a07d91ad462e0ba80b/pyvenvex.py

python3 pyvenvex.py virtual_environment
. virtual_environment/bin/activate

set +e

pip install absl-py numpy 	
cd ../../loadgen
pip install --force-reinstall dist/mlperf_loadgen-*.whl

cd ../translation/gnmt/tensorflow

# Install tensorflow 1.15
# If we dont get the nvidia tensorflow
# Ampere doesnt work

pip install nvidia-pyindex
pip install nvidia-tensorflow

. download_trained_model.sh
. download_dataset.sh
#. verify_dataset.sh - Pausing this for now, checksums are screwy, blame zenodo

# To run the benchmark
# python run_task.py --run=performance --batch_size=32

deactivate

cd ../../../




# Install the training workloads

# cd ../../../../../

. install_mlperf_and_virtual_environments_training.sh
'
echo "Done"