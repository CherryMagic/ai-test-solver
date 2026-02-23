# create_faiss.py
import faiss
import numpy as np
import json
import os
from sentence_transformers import SentenceTransformer

# -------------------------
# CONFIG
# -------------------------
CHUNKS_PATH = "data/chunks.json"
EMBEDDINGS_PATH = "data/embeddings.npy"
FAISS_INDEX_PATH = "data/faiss.index"
EMBEDDER_MODEL = "intfloat/multilingual-e5-small"

# -------------------------
# BUILD AND SAVE INDEX
# -------------------------
def build_index():
    """Build FAISS index from embeddings and chunks"""
    print("="*60)
    print("🔨 Building FAISS Index")
    print("="*60)
    
    # Check if files exist
    if not os.path.exists(CHUNKS_PATH):
        print(f"❌ Error: Chunks file not found at {CHUNKS_PATH}")
        print("   Please run prepare_data.py first")
        return False
    
    if not os.path.exists(EMBEDDINGS_PATH):
        print(f"❌ Error: Embeddings file not found at {EMBEDDINGS_PATH}")
        print("   Please run prepare_data.py first")
        return False
    
    try:
        # Load chunks
        print(f"📚 Loading chunks from {CHUNKS_PATH}...")
        with open(CHUNKS_PATH, encoding="utf-8") as f:
            all_chunks = json.load(f)
        print(f"   Loaded {len(all_chunks)} chunks")
        
        # Load embeddings
        print(f"📊 Loading embeddings from {EMBEDDINGS_PATH}...")
        embeddings = np.load(EMBEDDINGS_PATH)
        print(f"   Loaded embeddings with shape: {embeddings.shape}")
        
        # Get dimension
        dimension = embeddings.shape[1]
        print(f"   Embedding dimension: {dimension}")
        
        # Create FAISS index
        print("🔧 Creating FAISS index...")
        index = faiss.IndexFlatL2(dimension)  # L2 distance (Euclidean)
        
        # Add vectors to index
        print(f"   Adding {len(embeddings)} vectors to index...")
        index.add(embeddings)
        
        # Save index to disk
        print(f"💾 Saving index to {FAISS_INDEX_PATH}...")
        
        # Create data directory if it doesn't exist
        os.makedirs(os.path.dirname(FAISS_INDEX_PATH), exist_ok=True)
        
        faiss.write_index(index, FAISS_INDEX_PATH)
        
        print(f"✅ FAISS index created successfully!")
        print(f"   Total vectors: {index.ntotal}")
        print(f"   Index type: {type(index).__name__}")
        print(f"   Index saved to: {FAISS_INDEX_PATH}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error building index: {e}")
        import traceback
        traceback.print_exc()
        return False

# -------------------------
# LOAD INDEX + EMBEDDER
# -------------------------
def load_retriever():
    """Load FAISS index, chunks, and embedder"""
    print("🔍 Loading retriever components...")
    
    if not os.path.exists(FAISS_INDEX_PATH):
        print(f"❌ FAISS index not found at {FAISS_INDEX_PATH}")
        print("   Please run build_index() first")
        return None, None, None
    
    try:
        # Load FAISS index
        print(f"   Loading index from {FAISS_INDEX_PATH}...")
        index = faiss.read_index(FAISS_INDEX_PATH)
        print(f"   Index loaded with {index.ntotal} vectors")
        
        # Load chunks
        print(f"   Loading chunks from {CHUNKS_PATH}...")
        with open(CHUNKS_PATH, encoding="utf-8") as f:
            all_chunks = json.load(f)
        print(f"   Loaded {len(all_chunks)} chunks")
        
        # Load embedder
        print(f"   Loading embedding model: {EMBEDDER_MODEL}...")
        embedder = SentenceTransformer(EMBEDDER_MODEL)
        print(f"   Embedder loaded successfully")
        
        return index, all_chunks, embedder
        
    except Exception as e:
        print(f"❌ Error loading retriever: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

# -------------------------
# RETRIEVE CONTEXT
# -------------------------
def retrieve_context(question, index, all_chunks, embedder, k=3):
    """Retrieve relevant context for a question"""
    if index is None or all_chunks is None or embedder is None:
        print("❌ Retriever components not loaded properly")
        return ""
    
    try:
        print(f"🔎 Searching for: '{question}'")
        
        # Encode question
        question_embedding = embedder.encode([question])
        print(f"   Question embedding shape: {question_embedding.shape}")
        
        # Search in index
        distances, indices = index.search(np.array(question_embedding), k)
        
        # Collect retrieved chunks
        retrieved_chunks = []
        print(f"   Found {k} results:")
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(all_chunks):  # Make sure index is valid
                chunk_preview = all_chunks[idx][:100] + "..." if len(all_chunks[idx]) > 100 else all_chunks[idx]
                print(f"      Result {i+1}: distance={dist:.4f}, chunk {idx}: {chunk_preview}")
                retrieved_chunks.append(all_chunks[idx])
        
        # Combine chunks
        context = "\n\n".join(retrieved_chunks)
        print(f"   Retrieved context length: {len(context)} characters")
        
        return context
        
    except Exception as e:
        print(f"❌ Error retrieving context: {e}")
        import traceback
        traceback.print_exc()
        return ""

# -------------------------
# MAIN EXECUTION
# -------------------------
if __name__ == "__main__":
    # Build the index
    success = build_index()
    
    if success:
        print("\n🧪 Testing retrieval...")
        # Load and test
        index, chunks, embedder = load_retriever()
        
        if index is not None:
            # Test with a sample question
            test_question = "What is the main topic of this document?"
            context = retrieve_context(test_question, index, chunks, embedder, k=2)
            
            print("\n📝 Retrieved context preview:")
            print("-" * 40)
            print(context[:500] + "..." if len(context) > 500 else context)
            print("-" * 40)
            
            print("\n✅ FAISS index is ready to use!")
    else:
        print("\n❌ Index creation failed. Make sure prepare_data.py ran successfully first.")
