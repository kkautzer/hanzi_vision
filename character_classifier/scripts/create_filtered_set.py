from datetime import datetime
import os
import shutil


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
    
    
def create_filtered_set(whitelist_file):
    whitelist = load_whitelist(whitelist_file)
        
    source_dir = 'data/processed'
    target_dir = f"data/filtered/top-{len(whitelist)}"

    # if a directory already exists for this set of top-x chars, then return immediately,
    # since there no need to waste 5-10 minutes overwriting data with exact copies of it
    if os.path.isdir(target_dir):
        print("\tData directories for this charset already exist!")
        return
    else:
        filter_dataset(source_dir, target_dir, whitelist)
    
if __name__=="__main__":
    create_filtered_set(whitelist_file="data/whitelist.txt")