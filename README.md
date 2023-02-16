# SiamIris-v1
This is the repository of the research work: **SiamIris: An Open Source Siamese Network for Iris Recognition**.

- IEEE version of the paper is available here (upon approval)
- ArXiv version can be found here (upon submission)

## Requirements
This repo works with TensorFlow 2, and was tested on versions 2.2, 2.4 and 2.9. To install all the requirements please use:

```bash
pip install -r requirements.txt
```

## Model's Weights
The best results were obtained with the ResNet50 backbone and using periocular iris images a the input. This has the advantage of not needing segmentation nor localization for the Iris recognition Pipeline.

The weights can be downloaded from [this link]( https://www.dropbox.com/s/lk9ctfgjte0yv5l/weights-SiamIris.zip?dl=0); however, the password must be solicited to the author's emails:

- Daniel Benalcazar: dbenalcazar@ug.uchile.cl
- Juan Tapia: juan.tapia-farias@h-da.de

Please unzip the file and place the folder **weights/** on the root folder of this repo.

## How to use
Use following code to initialize the model. You must specify if the backbone is **'renet50'** or **'mobilenetv2'**.

```python
from siamiris_embedding import siamiris_embedding

# Initialize SiamIris Object
SiamIris = siamiris_embedding(backbone)
```
Then, to get the embedding of an iris image, please use:

```python
# Read iris image:
iris = cv2.cvtColor(cv2.imread('path/to/iris/image'), cv2.COLOR_BGR2RGB)

# Image on correct format:
iris = SiamIris.process_image(iris)

# Get embedding:
emb = SiamIris.get_embedding(iris)
```
Finally, to compare the embeddings of two images, please use:

```python
# Get euclidean distance of two embeddings:
dist = SiamIris.compare(emb1, emb2)
```

For an illustrative example, please refer to the jupyter notebook: **SiamIris Example.ipynb**

## Test on ND-LG4000-LR
You must solicit the Notre Dame Dataset directly to the University of Notre Dame. Then, you must place all the images in a single folder and change the format to PNG. The 10.959 images used in this work are listed in: "./ND-LG4000-LR/lists_images/all_iris.txt". It is very important to leave only the 10.959 listed images in the dataset folder because the provided code uses the index of each image for the comparisons. For example, compare image 0001 with image 0125.

Then, use the code in **iris_recognition.py** to evaluate the models on the test or validation splits.

```bash
python iris_recognition.py --backbone=resnet50 \\
  --opt_dir=/path/to/results/dir/              \\
  --dataset=/path/to/ND-LG4000-LR/png/images/  \\
  --split=test  
```

The code will perform the iris recognition tests and display the DET curve.

## Cite us
(coming soon)
