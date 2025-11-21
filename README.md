# Hanzi Vision OCR
> Project started 6/25/25<br>
> Last updated on 11/20/25

> Authors/Contributors: Keagan Kautzer<br>
> _Note: This repo will be sparsely updated throughout the academic year, as I focus primarily on my coursework_

## Overview
This repository is home to my Hanzi OCR Project. It contains three individual subdirectories (`model`, `frontend`, and `server`) each outlined briefly below.

### `model`
This is the main directory for all model-related data and information. This is where training and evaluation scripts, raw training / evaluation / test data, logs, and previous model weights are stored. The model utilizes a tweaked GoogLeNet (Inception-v1) architecture for grayscale input and a dynamic number of output classes, as this is the best available trade-off for time and accuracy given the current project and computing constraints. In the future, additional models will be trained using other underlying architectures.

### `frontend`
This is the main directory for all frontend web-interface relevant code. The frontend is a fairly simple and straightforward React app - it allows users to either upload and evaluate an image from their computer, or to evaluate a drawing from an on-screen canvas. This directory contains all of the pages, components, routing, and styling seen on the frontend webpage.

### `server`
This is the main directory for behind-the-scenes connections between the frontend interface and the trained models. It is a Python Flask API, allowing for the frontend to quickly and easily communicate with and evaluate images based on trained character models.

## Current Goals
The current core objectives for this project are as follows:

1. Scale training to 1000, 1500, and 2000 characters, rather than the current 500
2. Build a mobile interface / app (very similar to web interface from `frontend`)
3. Implement additional architectures to train and evaluate new models on

## Future Directions
Some brief future directions for this project are as follows:
1. Image Segmentation
    - This will significantly expand the project's functionality and real-world application, allowing for additional future directions to open up (transcription, translation, etc.)
    - Small updates to the frontend and server may be required (separate single-character and multi-character processing). Otherwise, the interactive process is largely the same.
2. Incorporate novice-level handwriting into the training data
