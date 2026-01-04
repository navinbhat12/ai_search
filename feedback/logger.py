import json

def log_feedback(query, doc_id, rating, comment):
    with open("feedback_log.jsonl", "a") as f:
        entry = {"query": query, "doc_id": doc_id, "rating": rating, "comment": comment}
        f.write(json.dumps(entry) + "\n")