# from glob import glob
# from lib import build_whoosh_index, get_whoosh_ix
# from whoosh.qparser import QueryParser

# files = glob("./data/hackduke_slack/**/*.json", recursive=True)
# INDEX_DIR = "whoosh_index"

# # build_whoosh_index(files, INDEX_DIR)

# ix = get_whoosh_ix(INDEX_DIR)
# qp = QueryParser("content", schema=ix.schema)
# q = qp.parse(u"fund code")
# with ix.searcher() as searcher:
#     results = searcher.search(q, limit=10)
#     for result in results:
#         path = result["path"]