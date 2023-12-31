<div style="text-align:center">

# DOCTABLE
## A simple tool to extract tables from pdf and images

</div>

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![Test Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen)

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Demo](#demo)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Overview
This repository contains the code to extract tables from pdf and images. The table extraction is done in two steps:
1. Table detection: the table is detected and cropped from the original image
    - The table detection is done using the YOLOv8s Table Detection model.
    - You can find reference to the model [here](https://huggingface.co/foduucom/table-detection-and-extraction)
2. Table recognition: the table is recognized and the text is extracted
   - I used the PaddleOCR models to recognize the structure and the text of the table.

## Demo
You can test your own images or pdfs by simply running a demo with streamlit:
* To run streamlit lunch:`streamlit run .\doctable\streamlit_demo\app.py`
[![Thumbnail for Demo Video](http://img.youtube.com/vi/XT3klGwHV0E/0.jpg)](https://www.youtube.com/watch?v=XT3klGwHV0E)

## Installation
### Prerequisites
- Python 3.8 or higher
- Poetry

### Steps
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
## Quick Start

Here is a simple example of how to use Doctable to extract tables from an image:

```python
from src.doctable import Doctable

# Initialize Doctable
doctable = Doctable()

# Path to your image (can be jpg, png, pdf, etc.)
img_path = "/path/to/your/image.jpg"

# Extract pages, each page contains a list of tables, 
# if the path is an image there will be only one page, 
# otherwise there will be one page for each page in the pdf.
pages = doctable.table_extraction(img_path)

# Print the recognition results for each table in each page
for page in pages:
   for table in page.tables:
      print(table.recognition_results["text"])
```

## Usage
This project provides several Jupyter notebooks that demonstrate how to use the table detection and recognition features. 

### How to test table detection:
The [detection.ipynb](notebooks%2Fdetection.ipynb) notebook demonstrates how to use the table detection feature. It includes examples of detecting tables in various types of images and PDFs.

### How to test table recognition:
The [recognition.ipynb](notebooks%2Frecognition.ipynb) notebook demonstrates how to use the table recognition feature. It includes examples of recognizing the structure and text of detected tables.

### How to test table detection + table recognition:
The [detection%2Brecognition.ipynb](notebooks%2Fdetection%2Brecognition.ipynb) notebook demonstrates how to use both the table detection and recognition features together. It includes examples of detecting tables in an image or PDF, and then recognizing the structure and text of the detected tables.


## Contributing
Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

Please make sure to update tests as appropriate.

## License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for more details.

## Contact
For any questions or support, please contact us at [enrico.gigante@gmail.com](mailto:enrico.gigante@gmail.com).