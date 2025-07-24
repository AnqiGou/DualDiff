# Installation
\# Install git
conda install -c anaconda git
\# Clone DualDiff code
git clone https://github.com/DualDiff.git
cd DualDiff
\# Prepare a font file; Arial Unicode MS is recommended, you need to download it on your own
mv your/path/to/arialuni.ttf ./font/Arial_Unicode.ttf
\# Create a new environment and install packages as follows:
conda create --name dualdiff python==3.10
pip install -r requirements.txt
conda activate dualdiff

# Dataset Preparation
\# To get training dateset
git clone https://www.modelscope.cn/datasets/iic/AnyWord-3M.git
\# To get benchmark
git clone https://www.modelscope.cn/datasets/iic/AnyText-benchmark.git

# Inference
python inference.py

# Evaluation
sh eval/eval_ocr.sh
sh eval/eval_clip.sh
sh eval/eval_fid.sh

# Training
Core model code is available, training scripts will be released soon.