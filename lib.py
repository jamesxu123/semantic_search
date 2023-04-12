from pypdf import PdfReader
import json

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