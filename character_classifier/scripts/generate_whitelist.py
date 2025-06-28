
def generate_whitelist(characters):
    with open('data/hanzi_db.csv', 'r', encoding='utf-8') as f:
        whitelist = [line.strip() for line in f.readlines()[1:characters+1] if line.strip()]
    with open('data/whitelist.txt', 'w', encoding='utf-8') as f:
        data = (f"{line.split(',')[1]}\n" for line in whitelist)
        f.writelines(data)

    
if __name__=="__main__":
    whitelist = generate_whitelist(25) # generate the 500 most common characters
