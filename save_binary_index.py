from datasets import load_dataset
import numpy as np
from faiss import IndexBinaryFlat, write_index_binary
from sentence_transformers.util import quantize_embeddings

dataset = load_dataset("mixedbread-ai/wikipedia-2023-11-embed-en-pre-1", split="train")
embeddings = np.array(dataset["emb"], dtype=np.float32)

ubinary_embeddings = quantize_embeddings(embeddings, "ubinary")
index = IndexBinaryFlat(1024)
index.add(ubinary_embeddings)
write_index_binary(index, "wikipedia_ubinary_faiss_1m.index")
