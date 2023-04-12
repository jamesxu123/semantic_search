import torch
from transformers import AutoTokenizer, AutoModel
from annoy import AnnoyIndex
from lib import extract_conversations
from glob import glob
import json

device = torch.device("mps")
t = AnnoyIndex(768, 'angular')
# model_ckpt = "sentence-transformers/multi-qa-mpnet-base-dot-v1"


model_ckpt = "sentence-transformers/msmarco-distilbert-base-tas-b"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = AutoModel.from_pretrained(model_ckpt).to(device)


def cls_pooling(model_output):
    return model_output.last_hidden_state[:, 0]


def get_embeddings(text_list):
    encoded_input = tokenizer(text_list, padding=True, truncation=True, return_tensors="pt").to(device)
    encoded_input = {k: v for k, v in encoded_input.items()}
    model_output = model(**encoded_input)
    return cls_pooling(model_output)


def build():
    print("[Starting processing]")
    files = glob("./data/hackduke_slack/**/*.json", recursive=True)
    result = [extract_conversations(i) for i in files]
    counter = 0
    db = {}
    for file in result:
        db[counter] = {}
        text, path = file
        db[counter]["path"] = path
        embed = get_embeddings(text)
        t.add_item(counter, torch.transpose(embed, 0, 1))
        counter += 1
        if counter % 20 == 0:
            print(f"* processed {counter / len(result) * 100}%")

    t.build(50)
    t.save("index.ann")
    with open("db.txt", "w") as file:
        json.dump(db, file)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # build()
    t.load("index.ann")
    file = open("db.txt", "r")
    db = json.load(file)
    file.close()

    while True:
        search = get_embeddings(input("Enter your search query: "))
        result = t.get_nns_by_vector(torch.transpose(search, 0, 1), 3, include_distances=True)
        print(result)
        for i in result[0]:
            print(db[str(i)])
