# Hanzi Vision OCR
> Project started 6/25/25<br>
> Last updated on 8/29/25

> Authors/Contributors: Keagan Kautzer

## Overview
This repository is home to my Hanzi OCR Project. It contains three individual subdirectories (`character_classifier`, `frontend`, and `server`) each outlined briefly below, and in much greater detail in each of their respective directories.

### `character_classifier`
This is the main directory for all model-related data and information. This is where training and evaluation scripts, raw training / evaluation / test data, logs, and previous model weights are stored. The model utilizes a tweaked GoogLeNet architecture for grayscale input and a dynamic number of output classes, as this is the best available trade-off for time and accuracy given the current project and computing constraints.

### `frontend`
This is the main directory for all frontend web-interface relevant code. The frontend is a fairly simple and straightforward React app - it allows users to either upload and evaluate an image from their computer, or to evaluate a drawing from an on-screen canvas. This directory contains all of the pages, components, routing, and styling seen on the frontend webpage.

### `server`
This is the main directory for behind-the-scenes connections between the frontend interface and the trained models. It is a Python Flask API, allowing for the frontend to quickly and easily communicate with and evaluate images based on trained character models.

## Current Goals
The current core objectives for this project are as follows:

1. Scale training to 750, 1000, and 1500 characters, rather than the current 500
2. Build a frontend interface that allows users to (a) upload and evaluate an image and (b) draw in and evaluate writing from an on-screen canvas / drawing
3. Incorporate novice-level handwriting into the training data (requires building a new dataset from scratch)

## Future Directions
Some brief future directions for this project are as follows:
1. Image Segmentation
    - This will significantly expand the project's functionality and real-world application, allowing for additional future directions to open up (transcription, translation, etc.)
    - Small updates to the frontend and server are required - namely, separate single-character and multi-character processing - otherwise, largely the exact same
2. Model Architecture Comparison
    - As stated in the overview, the recognition model currently utilizes a tweaked GoogLeNet architecture. In the future (with more resource availability), we can compare and evaluate the various aspects of many popular architectures (like ResNet and VGG)
    - Additionally, we can implement training processes to fully understand the effects of using more modern technologies like Vision Transformers rather than CNNs
