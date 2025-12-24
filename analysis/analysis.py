import pandas as pd
import matplotlib.pyplot as plt
import os
import json

save_path_root = None

## num. characters vs. highest accuracy analysis
def nCharsVsAccuracyTable():
    if save_path_root:
        save_path = f'{save_path_root}/chars_accuracy_table.png'
    else:
        save_path = None
        
    # read metadata files from `./models/exports/metadata` to get highest accuracy for each model
    filenames = os.listdir('./model/exports/metadata')
    
    # create mapping between nchars and highest accuracy
    char_accuracy_data = pd.DataFrame(
        columns=['architecture', 'model_name', 'nchars', 'max_val_accuracy', 'max_val_epoch', 'epochs']
    )
    column_translate = {
        'architecture': "Underlying Architecture",
        "model_name": "Model Name",
        "nchars": "Number of Characters",
        "max_val_accuracy": "Highest Accuracy",
        "max_val_epoch": "Highest Accuracy Epoch",
        "epochs": "Epochs"
    }
    
    # read metadata & build dataframe
    for fn in filenames:
        with open(f'./model/exports/metadata/{fn}', 'r', encoding='utf-8') as f:
            char_accuracy_data.loc[len(char_accuracy_data)] = json.load(f)

    # update column names
    char_accuracy_data.rename(columns=column_translate, inplace=True)    
        
    # create table
    fig, ax = plt.subplots(figsize=(8, 2.5))
    ax.axis('off')
    table = plt.table(
        cellText=char_accuracy_data.values,
        colLabels=char_accuracy_data.columns,
        loc='center',
        cellLoc='center'
    )    
    
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1, 1.4)

    plt.tight_layout()
    
    if save_path is not None:
        os.makedirs(save_path_root, exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f'Saved figure to {save_path}!')
    
    plt.show()
        

## epochs vs. accuracy analysis for each character set
def epochsVsAccuracyCurve():
    print("--- NOT YET IMPLEMENTED ---")
    return

    # read training_data.csv to get accuracy vs. epoch data for each model
    training_data = pd.read_csv('./model/exports/training_data.csv')
    
    # divide dataframe by model name, then sort increasing by epoch
    
    # for each [exported] model, create a graph of accuracy vs. epoch

    # create an aggregate graph of accuracy vs. epoch (one line per model)
    
    # show (+ save) plot
    
    pass

## main method - include a switch to save plots or just show them
if __name__ == "__main__":
    save_plots = True
    if save_plots:
        save_path_root = './analysis/figures'

    nCharsVsAccuracyTable()
    # epochsVsAccuracyCurve()