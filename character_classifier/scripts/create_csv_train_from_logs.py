import os
import re
import json
import ast


## NOTE THIS SCRIPT WILL OVERWRITE THE EXISTING DATA, USE CAUTION WHEN RUNNING ##

def create_csv():
    # each list entry will be appended to a .csv
    export_data = [["name", "nchars", "lr", "epoch", "train_loss", "val_accuracy", "thresholded"]] 

    # open folder of log files
    for filename in os.listdir("./character_classifier/logs"):
        # read log file and save each entry in data
        with open(f"./character_classifier/logs/{filename}", "r", encoding='utf-8') as log:
            # since there is a consistent formatting and text throughout the log files, we can
            # read the logs in a specific way:
                # if the line starts with Model Configuration:, set name, lr, nchars 
                # if the line starts with Epoch [ax/yz], update the epoch # and training loss
                # if the line starts with Validation Accuracy:, set the val_accuracy
            # then, we can push to export_data, and reset values for epoch, loss, and accuracy
            data = [None] * 7 # ["name", "nchars", "lr", "epoch", "train_loss", "val_accuracy"]
            max_val = float(0)
            max_val_epoch = 0
            for line in log.readlines(): # read file lines
                # split based on datetime at start of message
                line = re.split(r"\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{6}\]", line)[-1].strip()
                if (line.startswith("Model Configuration: ")):
                    params_dict_str = line.split("Model Configuration: ")[-1]
                    # parse as a dictionary
                    params_dict = ast.literal_eval(params_dict_str)
                    
                    data[0] = f"\"{params_dict['model_name']}\"" ## name
                    data[1] = str(params_dict['num_characters']) ## nchars
                    data[2] = str(params_dict['learning_rate']) ## learning rate
                    data[6] = str(params_dict.get('thresholded', False) ) ## if thresholded images were used for training
                elif (line.startswith("Epoch [")): 
                    data[3] = line.split("Epoch [")[-1]. split('/')[0] ## epoch
                    data[4] = line.split("Epoch [")[-1].split(", Loss: ")[-1] ## loss
                elif (line.startswith("Validation Accuracy: ")):
                    data[5] = line.split("Validation Accuracy: ")[-1].split("%")[0] ## accuracy
                    if float(data[5]) > max_val:
                        max_val = float(data[5])
                        max_val_epoch = int(data[3])
                    ## now have all data for this epoch, so push a copy to export_data
                    export_data.append(data.copy())
                    
        with open(f'./character_classifier/models/metadata/{data[0][1:-1]}-metadata.json', 'w', encoding='utf-8') as f:
            metadata_json = {
                "model_name": data[0][1:-1], # remove leading/trailing quotations
                "nchars": int(data[1]),
                "epochs": int(data[3] or 0),
                "max_val_accuracy": max_val,
                "max_val_epoch": max_val_epoch,
                "threshold": data[6]
            }
            json.dump(metadata_json, f, indent=4)
        ## put most recent data to the metadata file 
                    
    # open csv log file to write [TEST FILE, CHANGE WHEN CONFIRMED WORKING]
    with open("./character_classifier/exports/training_data.csv", "w", encoding='utf-8') as csv: 
        csv.writelines(f"{(",").join(entry)}\n" for entry in export_data)
        print("Log data successfully written to `./character_classifier/data/data_export.csv`")
        
if os.path.isfile("./character_classifier/exports/training_data.csv"):
    run = input("CSV file already exists, running this script will overwrite existing data. Continue? (Y/N): ")

    # check before overwriting
    if run.strip().lower()=="y" or run.strip().lower()=="yes":    
        create_csv()
    else:
        print("Canceled CSV file creation")