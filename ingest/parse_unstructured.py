import fitz
import os
import json
from email import policy
from email.parser import BytesParser

def parse_pdf(path):
    with fitz.open(path) as doc:
        text = "".join(page.get_text() for page in doc)
    return text

def parse_eml(path):
    with open(path, 'rb') as f:
        msg = BytesParser(policy=policy.default).parse(f)
    return msg.get_body(preferencelist=('plain')).get_content()

def parse_files():
    output = []
    for filename in os.listdir("data/unstructured"):
        if filename.endswith(".pdf"):
            body = parse_pdf(f"data/unstructured/{filename}")
            doc = {"doc_id": filename, "source_type": "pdf", "body": body}
            output.append(doc)
        elif filename.endswith(".eml"):
            body = parse_eml(f"data/unstructured/{filename}")
            doc = {"doc_id": filename, "source_type": "email", "body": body}
            output.append(doc)
    with open("parsed_docs.jsonl", "w") as f:
        for doc in output:
            f.write(json.dumps(doc) + "\n")

if __name__ == "__main__":
    parse_files()