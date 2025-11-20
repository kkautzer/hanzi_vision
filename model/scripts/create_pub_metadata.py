import shutil
import os

whitelist = []

with open('./model/models/public_models_whitelist.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        whitelist.append(line.strip())
  
for name in whitelist:
    shutil.copy(f'./model/models/metadata/{name}-metadata.json', f'./model/exports/metadata_public/{name}-metadata.json')
