from typing import Iterable
from pypdf import PdfReader
import json
from whoosh.index import create_in, open_dir
from whoosh.fields import *
from whoosh.qparser import QueryParser


schema = Schema(path=ID(stored=True), content=TEXT)
INDEX_DIR = "whoosh_index"

def read_text_from_pdf(path):
    reader = PdfReader(path)
    texts = map(lambda p: p.extract_text(), reader.pages)
    return ' '.join(texts), path

def extract_conversations(file):
    conversations = []
    with open(file) as f:
        json_content = json.load(f)
        for message in json_content:
            if 'subtype' not in message and 'text' in message:
                # print(message)
                name = message['user_profile']['real_name'] if 'user_profile' in message else "unknown user"
                conversations.append(f"{name}: {message['text']}")
    return '\n'.join(conversations), file


def build_whoosh_index(files: Iterable[str], indexdir):
    ix = create_in(indexdir, schema)
    writer = ix.writer()
    data = map(extract_conversations, files)
    results = list(map(lambda row: writer.add_document(path=row[1], content=row[0]), data))
    writer.commit()

def get_whoosh_ix(folder: str):
    return open_dir(folder)

# files = glob("./data/hackduke_slack/**/*.json", recursive=True)

# build_whoosh_index(files, INDEX_DIR)

class SearchResult:
    def __init__(self, path, content) -> None:
        self.path = path
        self.content = content

def search_by_term(INDEX_DIR, term):
    ix = get_whoosh_ix(INDEX_DIR)
    qp = QueryParser("content", schema=ix.schema)
    q = qp.parse(term)
    search_results = []
    with ix.searcher() as searcher:
        results = searcher.search(q, limit=10)
        for result in results:
            path = result["path"]
            search_results.append(SearchResult(path, extract_conversations(path)))
    return search_results