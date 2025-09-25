from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct
from qdrant_client.http.models import Filter, FieldCondition, MatchAny
import os
import uuid
from openai import OpenAI
from sentence_transformers import SentenceTransformer

class QdrantManager:
    def __init__(self, batch_size=50, use_openai=True):
        self.client = QdrantClient(path="storage/qdrant")
        self.use_openai = use_openai
        self.batch_size = batch_size
        if self.use_openai:
            self.embedding_model = "text-embedding-3-small"
            self.openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.vector_size = 1536
        else:
            self.st_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.vector_size = self.st_model.get_sentence_embedding_dimension()

    def _get_embeddings(self, texts):
        if self.use_openai:
            resp = self.openai.embeddings.create(input=texts, model=self.embedding_model)
            return [d.embedding for d in resp.data]
        else:
            return self.st_model.encode(texts, convert_to_numpy=True).tolist()

    def init_collection(self, collection_name):
        try:
            self.client.get_collection(collection_name)
        except:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size, distance=Distance.COSINE
                ),
            )

    def add_document(self, collection_name, chunks, metadata=None):
        self.init_collection(collection_name)
        meta = metadata or {}

        # embed in batches to avoid big payloads
        for i in range(0, len(chunks), self.batch_size):
            batch_chunks = chunks[i : i + self.batch_size]
            embeddings = self._get_embeddings(batch_chunks)

            points = []
            for text, emb in zip(batch_chunks, embeddings):
                # unique ID per chunk
                point_id = str(uuid.uuid4())
                points.append(
                    PointStruct(id=point_id, vector=emb, payload={**meta, "text": text})
                )
            self.client.upsert(collection_name=collection_name, points=points)

    def query(self, collection_name, query, top_k=10, pdf_ids=None):
        q_emb = self._get_embeddings([query])[0]

        filter_condition = None
        if pdf_ids:
            filter_condition = Filter(
                must=[FieldCondition(key="pdf_id", match=MatchAny(any=pdf_ids))]
            )

        hits = self.client.search(
            collection_name=collection_name,
            query_vector=q_emb,
            limit=top_k,
            query_filter=filter_condition,
        )

        return [h.payload["text"] for h in hits]
