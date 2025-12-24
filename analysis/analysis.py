import pandas as pd
import matplotlib.pyplot as plt
import os
import json

## num. characters vs. highest accuracy analysis
def nCharsVsAccuracyTable(save_path_root=None, show_plots=True):
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
    
    if show_plots:
        plt.show()   
    
    # clear plot
    plt.clf() 

## epochs vs. accuracy analysis for each character set
def epochsVsAccuracyCurve(save_path_root=None, show_plots=True):
    if save_path_root:
        save_path = f'{save_path_root}/epoch_accuracy_curves'
    else:
        save_path = None
        
    # read training_data.csv to get accuracy vs. epoch data for each model
    tdf = pd.read_csv('./model/exports/training_data.csv')
    model_names = tdf['name'].unique()
    
    for name in model_names:
        mdf = tdf.loc[tdf['name'] == name]
        plt.plot(mdf['epoch'], mdf['val_accuracy'])
        plt.title(f'Epoch vs. Validation Accuracy ({name})')
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.ylim([0, 100])
        
        plt.tight_layout()

        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(f'{save_path}/{name}.png', dpi=300)
            print(f'Saved figure to {save_path}/{name}.png!')
            
        if show_plots:
            plt.show()
            
        # clear plot
        plt.clf() 
    
    # create an aggregate graph of accuracy vs. epoch (one line per model)
    
## main method - include a switch to save plots or just show them
if __name__ == "__main__":
    
    save_plots = True
    show_plots = True
    
    if save_plots:
        save_path_root = './analysis/figures'
    else:
        save_path_root = None

    nCharsVsAccuracyTable(save_path_root=save_path_root, show_plots=show_plots)
    epochsVsAccuracyCurve(save_path_root=save_path_root, show_plots=show_plots)