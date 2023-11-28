______________________________________________________________________
<div style="text-align:center">


#   DOCTABLE
## A simple tool to extract tables from pdf and images

______________________________________________________________________

</div>

This repository contains the code to extract tables from pdf and images. 
The table extraction is done in two steps:
1. Table detection: the table is detected and cropped from the original image
    - The table detection is done using the YOLOv8s Table Detection model.
    - You can find reference to the model [here](https://huggingface.co/foduucom/table-detection-and-extraction)
2. Table recognition: the table is recognized and the text is extracted
   - I used the PaddleOCR models to recognize the structure and the text of the table.

## 💎 Installation with `poetry`

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

### Usage: How to test table detection:
* See [detection.ipynb](notebooks%2Fdetection.ipynb)

### Usage: How to test table recognition:
* See [recognition.ipynb](notebooks%2Frecognition.ipynb)

### How to test table detection + table recognition:
* See how to try the whole pipeline on a single image or pdf [here](notebooks%2Fdetection%2Brecognition.ipynb)
# DEMO
You can test your own images or pdfs by simply running a demo with streamlit:
* To run streamlit lunch:`streamlit run .\doctable\streamlit_demo\app.py`
[![Demo Video](http://img.youtube.com/vi/hb2zUQJB1d4/0.jpg)](https://www.youtube.com/watch?v=hb2zUQJB1d4)
