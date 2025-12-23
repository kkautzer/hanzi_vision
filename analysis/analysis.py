import pandas as pd

import os
import json

save_path_root = None

## num. characters vs. highest accuracy analysis
def nCharsVsAccuracy():
    # read metadata files from `./models/exports/metadata` to get highest accuracy for each model
    models = os.listdir('./model/exports/metadata')
    
    # create mapping between model name and highest accuracy
    model_accuracies = {}
    for model in models:
        with open(f'./model/exports/metadata/{model}', 'r') as f:
            metadata = json.load(f)
            model_accuracies[metadata['nchars']] = metadata['highest_accuracy']
    
    # create graph of num. characters vs. highest accuracy
    
    # show (+ save) plot
    
    pass

## epochs vs. accuracy analysis for each character set
def epochsVsAccuracy():
    # read training_data.csv to get accuracy vs. epoch data for each model
    training_data = pd.read_csv('./model/exports/training_data.csv')
    
    # sort by model name
    
    # create graph of accuracy vs. epoch for each character set
    
    # show (+ save) plot
    
    pass

## main method - include a switch to save plots or just show them
if __name__ == "__main__":
    save_plots = False
    if save_plots:
        save_path_root = './analysis/plots/'

    nCharsVsAccuracy()
    epochsVsAccuracy()