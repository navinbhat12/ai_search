from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct
from langchain.text_splitter import RecursiveCharacterTextSplitter
import json

model = SentenceTransformer('all-MiniLM-L6-v2')
client = QdrantClient(
    url="https://454be4f9-9809-4ec2-a6c4-aa61752fbb1b.us-east4-0.gcp.cloud.qdrant.io",
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0._xg2d_mS1qMrJ38neK735_1U3sq9pYuL4U1F1Vo_FjU"
)
client.recreate_collection(
    collection_name="docs",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE)
)

def embed_and_store():
    with open("./parsed_docs.jsonl") as f:
        lines = [json.loads(line) for line in f]

    print(f"Loaded {len(lines)} documents")
    
    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500)
    
    point_id = 0
    for i, doc in enumerate(lines):
        # Split document into chunks
        chunks = text_splitter.split_text(doc['body'])
        
        for chunk in chunks:
            # Generate embedding for each chunk
            embedding = model.encode(chunk)
            
            # Store chunk with metadata
            client.upsert(
                collection_name="docs",
                points=[PointStruct(
                    id=point_id,
                    vector=embedding.tolist(),
                    payload={
                        'doc_id': doc['doc_id'],
                        'chunk': chunk,
                        'metadata': doc.get('metadata', {})
                    }
                )]
            )
            point_id += 1
            
        print(f"Processed doc {i}: {doc['doc_id']} into {len(chunks)} chunks")


if __name__ == "__main__":
    embed_and_store()
