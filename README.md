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

The weights can be downloaded from this link (upon uploading); however, the password must be solicited to the author's emails:

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

## Cite us
(coming soon)
