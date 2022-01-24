# Facial Expression Recognition Using Prior Attention

## Table of contents

- [General info](#general-info)
- [Background](#Background)
- [Repository Description](#repository-description)
- [Attention Mechanism](#Attention-Mechanism)
- [Architecture](#Architecture)
- [Requirement](#Requirement)
- [Further development ideas](#further-development-ideas)
- [References](#References)
- [Notes](#Notes)

## General info

In this project, we propose an approach based on an attention mechanism in the prepossessing phase, even before training our network. which requires low resource consumption and is still able to achieve great performance over previous models on the FERG database.

## Background

Over the last few years, there is an increase in the desire to use deep learning tools to understand human behavior patterns. One of the most trendy challenges in this field is facial expression recognition. Facial expressions are a form of nonverbal communication. The immediate association when talking about human behavior is facial expression and, it's can tell us a lot about human emotions. The challenge is predicting the underlying image to one of seven different emotions (happiness, sadness, anger, fear, disgust, surprise, neutral).

## Repository Description

| Filename                    | description                                                                                       |
| --------------------------- | ------------------------------------------------------------------------------------------------- |
| `emotion-recognision.ipynb` | The main file in google colab format, including the prepossessing. to open import to google colab |
| `emotion-recognision.py`    | The main file in Python format                                                                    |
| `prepossessing.py`          | The prepossessing only in a python format                                                         |
| `dataset.py`                | Python file consists of the implementation of the dataset object.                                 |
| `architecture.py `          | Python file consists of the implementation of the proposed architecture.                          |
| `res `                      | Folder consists of all the images from the project                                                |
| `requirement.txt`           | File containing all the packages we used in this project                                          |
## Attention Mechanism
We proposed an attention mechanism that relies on a prior belief that much of the clues come from two major regions- the mouse region and the eyes region. Considering this fact we divide the images into two blocks.
- Upper Block consists of the eyes and eyebrows.
- Lower Block consists of the mouse.  
We extract those two blocks by a pretrained model from OpenCv library and others helper functions we implemented.

The fully explenation is in the report.

<p align="center">
  <img src=".\res\FlowChartAttention.png" width="350" alt="accessibility text">
</p>

## Architecture

The proposed architecture used for this work is further explained in the report

<p align="center">
  <img src=".\res\Architecture.png" width="350" alt="accessibility text">
</p>

## Requirement

To run this project, you need to install several packages. For convenience, we created a `requirement.txt` file consists of all the packages used in this projcet.

In order to install all the packages in the `requirement.txt` file, simply use to command `pip install -r requirements.txt`.

## Further development ideas

1. Try to expand our work to more facial expression datasets.
2. Try different uses of the attention mechanism.

## References

- FERG dataset source: [FERG dataset](http://grail.cs.washington.edu/projects/deepexpr/ferg-2d-db.html)

## Notes

- The data, and the predictor are too big to be uploaded to GitHub. You can found them in the following links:
  [train](https://drive.google.com/file/d/1wwtsQ1cCfpP132pGP7HZ5Ot7nmUZvimt/view?usp=sharing)
  [validation](https://drive.google.com/file/d/1q5qOGdZ0zkmZgv5Avyc1OrWa-FYNQX3S/view?usp=sharing),
  [test](https://drive.google.com/file/d/1pXyXMXUk08lZlnmqj-7hmD7xnKyM4Q7x/view?usp=sharing),
  [raw-images](https://drive.google.com/drive/folders/1FjyYvSZAEPQaoROEpr5FLtK2yGmLkt6x?usp=sharing),
  [predictor](https://drive.google.com/drive/folders/1o1DtnFnSwdRO8o23eW-a9jo_6cbY7ZA6?usp=sharing)
  (for the raw images and the predictor you need to download the files and put them in a folder with the exact same name as in the original folder)
