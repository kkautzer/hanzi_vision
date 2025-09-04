import shutil
import os

whitelist = []

with open('./character_classifier/models/public_models_whitelist.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        whitelist.append(line.strip())
  
for name in whitelist:
    shutil.copy(f'./character_classifier/models/metadata/{name}-metadata.json', f'./character_classifier/exports/metadata_public/{name}-metadata.json')
