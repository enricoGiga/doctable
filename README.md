______________________________________________________________________
<div style="text-align:center">

# 🤖  Doctable





______________________________________________________________________



</div>

## 💎 Installation with `poetry`

Installation is as simple as running:

```bash
poetry install
```

```bash
# Clone repo
git clone https://github.com/enricoGiga/doctable.git

# Go to repo directory
cd doctable

# Download poetry if you don't have it already
see: https://python-poetry.org/docs/#installation

# Install project dependencies
poetry install

```

## Download the required models
In the current section is explained how and where download all the required models
### Download the Paddle models:
* navigate inside `data/models`
* `wget http://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_infer.tar && tar xf en_PP-OCRv3_det_infer.tar`
* `wget http://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_rec_infer.tar && tar xf en_PP-OCRv3_rec_infer.tar`
* `wget https://paddleocr.bj.bcebos.com/ppstructure/models/slanet/en_ppstructure_mobile_v2.0_SLANet_infer.tar && tar xf en_ppstructure_mobile_v2.0_SLANet_infer.tar` 

## Usage
Define the **PROJECT_DIR** virtual environment, it is the root of your project
### Usage: Layout-parser with Detectron:
* Download the configuration file from [here](https://layout-parser.readthedocs.io/en/latest/notes/modelzoo.html): chose: TableBank dataset faster_rcnn_R_50_FPN_3x)
* To avoid downloading the weights from url (download it manually, save it inside the [models](data%2Fmodels) path and change the WEIGHTS inside the configuration file)

### Usage: How to test table detection:
* See [detection_with_layout_parser.ipynb](notebooks%2Fdetection_with_layout_parser.ipynb)

### Usage: How to test table recognition:
* Test the results of the recognition model on your cropped table image:
* See [recognition_with_paddle.ipynb](notebooks%2Frecognition_with_paddle.ipynb)

### How to test table detection + table recognition:
* You can test the table detection and recognition  either on image or pdf, call the method: `ailab_table_extraction(path: str)`
* See [detection_and_recognition.ipynb](notebooks%2Fdetection_and_recognition.ipynb)


