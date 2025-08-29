from datetime import datetime
import os
import shutil
from torchvision import datasets


def load_whitelist(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]
    
def filter_dataset(src, dst, whitelist):
    os.makedirs(dst, exist_ok=True)
    copied_count = 0
    
    for set_type in os.listdir(src):
        copied_count_s = 0
        for folder_name in os.listdir(src+"/"+set_type):
            if folder_name in whitelist:
                src_path = os.path.join(src+'/'+set_type, folder_name)
                dst_path = os.path.join(dst+'/'+set_type, folder_name)
                if os.path.isdir(src_path):
                    shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
                    print(f"\tCharacter folder '{folder_name}' successfully copied ({copied_count_s+1} / {len(whitelist)})")
                    copied_count += 1
                    copied_count_s += 1
                
        print(f"[{datetime.now()}] Copied {copied_count_s} folders from '{src+'/'+set_type}' to '{dst+'/'+set_type}'")
        copied_count_s = 0
    print(f"[{datetime.now()}] Successfully initialized directories (copied {copied_count} folders from '{src}' to '{dst}')")
    
def create_class_list(whitelist):
    classes = whitelist[:]
    classes.sort()
    # classes = [ name+"\n" for name in classes]
    with open(f"character_classifier/classes/top-{len(whitelist)}-classes.txt", 'w', encoding='utf-8') as f:
        print(classes)
        f.writelines(f"{c}\n" for c in classes)
    print(f"[{datetime.now()}] Successfully created class list for {len(whitelist)} characters")
    
def create_filtered_set(whitelist_file):
    whitelist = load_whitelist(whitelist_file)
        
    source_dir = 'character_classifier/data/processed'
    target_dir = f"character_classifier/data/filtered/top-{len(whitelist)}"

    # if a directory already exists for this set of top-x chars, then return immediately,
    # since there no need to waste 5-10 minutes overwriting data with exact copies of it
    if os.path.isdir(target_dir):
        print("\tData directories for this charset already exist!")
        if os.path.isfile(f"character_classifier/classes/top-{len(whitelist)}-classes.txt"):
            print("\tClass list file for this charset already exists!")
            return
        else:
            create_class_list(whitelist)
            return
    else:
        filter_dataset(source_dir, target_dir, whitelist)
        create_class_list(whitelist)
    
if __name__=="__main__":
    create_filtered_set(whitelist_file="character_classifier/data/whitelist.txt")