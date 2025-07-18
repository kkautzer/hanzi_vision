
def generate_whitelist(characters):
    with open('character_classifier/data/hanzi_db.csv', 'r', encoding='utf-8') as f:
        whitelist = [line.strip() for line in f.readlines()[1:characters+1] if line.strip()]
    with open('character_classifier/data/whitelist.txt', 'w', encoding='utf-8') as f:
        data = (f"{line.split(',')[1]}\n" for line in whitelist)
        f.writelines(data)

    
if __name__=="__main__":
    whitelist = generate_whitelist(5) # generate the 25 most common characters
