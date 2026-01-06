import csv
import os

# # # print(os.listdir('./model/data/processed/test'))

with open('./model/data/hanzi_db_full.csv', mode='r', encoding='utf-8') as infile:
    reader = csv.reader(infile)
    hanzi_list = [row for row in reader]


missing = 0
found = 0


with open(f'./model/data/hanzi_db.csv', mode='w', newline='', encoding='utf-8') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(hanzi_list[0])  # write header

    for row in hanzi_list[1:]:
        if not os.path.exists(f'./model/data/processed/test/{row[1]}'):
            # print(f'Missing file for character: {row[1]}')
            missing = missing + 1
        else:      
            writer.writerow(row)

            found = found + 1

print('--- Summary ---')
print(f'Total found: {found}')
print(f'Total missing: {missing}')
