from lib import extract_conversations
from glob import glob
import statistics

files = glob("./data/hackduke_slack/**/*.json", recursive=True)
result = [extract_conversations(i)[0] for i in files]

print(statistics.quantiles(map(lambda x: len(x.split(" ")), result), n=10))
print(statistics.quantiles(range(1000), n=10))