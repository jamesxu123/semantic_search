import torch
from annoy import AnnoyIndex
from lib import extract_conversations
from glob import glob
import json
from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder
import operator

cross_enc_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512, device="mps")

device = torch.device("mps")
t = AnnoyIndex(768, 'dot') # dot is better for long-form
# # model_ckpt = "sentence-transformers/multi-qa-mpnet-base-dot-v1"


# model_ckpt = "sentence-transformers/msmarco-distilbert-base-tas-b"
model = SentenceTransformer("msmarco-distilbert-base-tas-b")
# tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
# model = AutoModel.from_pretrained(model_ckpt).to(device)


def cls_pooling(model_output):
    return model_output.last_hidden_state[:, 0]

annoy_filename = "msmarco_dot_additional.ann"


# def get_embeddings(text_list):
#     encoded_input = tokenizer(text_list, padding=True, truncation=True, return_tensors="pt").to(device)
#     encoded_input = {k: v for k, v in encoded_input.items()}
#     model_output = model(**encoded_input)
#     return cls_pooling(model_output)


def build():
    print("[Starting processing]")
    files = glob("./data/hackduke_slack/**/*.json", recursive=True)
    result = [extract_conversations(i) for i in files]
    counter = 0
    db = {}
    embeddings = model.encode([pair[0] for pair in result], show_progress_bar=True, convert_to_numpy=True, device="mps")
    for embed, path in zip(embeddings, files):
        db[counter] = {}
        db[counter]["path"] = path
        t.add_item(counter, embed)
        counter += 1
        if counter % 20 == 0:
            print(f"* processed {counter / len(result) * 100}%")

    t.build(200)
    t.save(annoy_filename)
    with open("db.txt", "w") as file:
        json.dump(db, file)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # build()
    t.load(annoy_filename)
    file = open("db.txt", "r")
    db = json.load(file)
    file.close()
    running = True;
    while running:
        query = input("Enter your search query: ")
        if query == 'QUIT':
            running = False
        else:
            search = model.encode(query, convert_to_numpy=True)
            result = t.get_nns_by_vector(search, 20, include_distances=True)
            print(result)
            contents = []
            text = []
            for i in result[0]:
                txt = extract_conversations(db[str(i)]['path'])[0]
                text.append(txt)
                contents.append((query, txt))
            scores = cross_enc_model.predict(contents)
            reranked = sorted(zip(scores, text), key=lambda x: x[0], reverse=True)
            for entry in map(operator.itemgetter(1), reranked[:2]):
                print('-----------------')
                print(entry)
