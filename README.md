# Hanzi Vision OCR
> Project started 06/26/25<br>
> README last updated on 12/25/25<br>
> Project last updated on 12/25/25

> Authors/Contributors: Keagan Kautzer<br>
> _Note: This repo will be sparsely updated throughout the academic year, as I focus primarily on my coursework_

## Overview
This repository is home to my Hanzi OCR Project. It contains three individual subdirectories (`model`, `frontend`, and `server`) each outlined briefly below.

### `model`
This is the main directory for all model-related data and information. This is where training and evaluation scripts, raw training / evaluation / test data, logs, and previous model weights are stored. The model utilizes a tweaked Inception-v1 (GoogLeNet) architecture for grayscale input and a dynamic number of output classes, as this is the best available trade-off for time and accuracy given the current project constraints. In the future, additional models may be trained using other underlying architectures.

### `analysis`
This is the main direcotry for all data analysis operations. The main component is a script to generate various plots and tables based on model metadata and training data. Currently, figures are generated for:

* Metadata Table (Name, Architecture, Number of Characters, Highest Accuracy, Highest Accuracy Epoch, and Number of Epochs)
* Number of Characters vs. Highest Accuracy
* Epoch vs. Accuracy (Individual)
* Epoch vs. Accuracy (Combined / Aggregate)
* Epoch vs. Average Accuracy

As deemed necessary, the script may be updated to generate additional figures. Furthermore, a future update will implement necessary p-value calculations to complement the existing figures.

### `frontend`
This is the main directory for all frontend web-interface relevant code. The frontend is a straightforward and intuitive React app - it allows users to either upload and evaluate an image from their computer, or to evaluate a drawing from an on-screen canvas. This directory contains all of the pages, components, routing, and styling seen on the frontend webpage.

### `server`
This is the main directory for behind-the-scenes connections between the frontend interface and the trained models. It is a Python Flask API, allowing for the frontend to quickly and easily communicate with and evaluate images based on trained character models.

## Current Goals
The current core objectives for this project are as follows:

1. Scale training to additional character sets (one model for roughly every 500 characters). Currently, models trained for 100 epochs use character sets consisting of the most popular {500, 750, 1000, 1500, 2000, 2500, 4000, 5000} Hanzi. The best-accuracy version of each of these models are available for testing on the frontend interface.
2. Create a series of tests to run before commits are made to this repository, ensuring no breaking changes are incorrectly pushed.
3. Build a mobile interface / app (very similar to web interface from `frontend`)

## Future Directions
Some brief future directions for this project are as follows:
1. Image Segmentation
    - This will significantly expand the project's functionality and real-world application, allowing for additional future directions to open up (transcription, translation, etc.)
    - Small updates to the frontend and server may be required (separate single-character and multi-character processing). Otherwise, the interactive process is largely the same.
2. Examine the accuracy impact of incorporating novice-level handwriting into the training data
3. Implement additional architectures to train and evaluate new models on
