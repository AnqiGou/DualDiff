# Installation
\# Install git <br>
conda install -c anaconda git <br>
<br>
\# Clone DualDiff code <br>
git clone https://github.com/DualDiff.git <br>
cd DualDiff <br>
<br>
\# Prepare a font file; Arial Unicode MS is recommended, you need to download it on your own <br>
mv your/path/to/arialuni.ttf ./font/Arial_Unicode.ttf <br>
<br>
\# Create a new environment and install packages as follows: <br>
conda create --name dualdiff python==3.10 <br>
pip install -r requirements.txt <br>
conda activate dualdiff <br>

# Dataset Preparation
\# To get training dateset <br>
git clone https://www.modelscope.cn/datasets/iic/AnyWord-3M.git <br>
<br>
\# To get benchmark <br>
git clone https://www.modelscope.cn/datasets/iic/AnyText-benchmark.git <br>

# Inference
python inference.py

# Evaluation
sh eval/eval_ocr.sh<br>
sh eval/eval_clip.sh<br>
sh eval/eval_fid.sh

# Training
Core model code is available, training scripts will be released soon.
