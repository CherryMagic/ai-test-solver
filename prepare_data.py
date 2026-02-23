# prepare_data.py
import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
import warnings
warnings.filterwarnings('ignore')

# -------------------------
# CONFIG
# -------------------------
PDF_PATH = "data/pd.pdf"  # Your PDF file
CHUNKS_PATH = "data/chunks.json"
EMBEDDINGS_PATH = "data/embeddings.npy"
EMBEDDER_MODEL = "intfloat/multilingual-e5-small"  # Matches your create_faiss.py

# Chunking parameters
CHUNK_SIZE = 500  # Size of each chunk in characters
CHUNK_OVERLAP = 50  # Overlap between chunks

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from PDF file
    """
    print(f"📄 Loading PDF: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    # Combine all pages
    full_text = "\n".join([doc.page_content for doc in documents])
    print(f"   Extracted {len(documents)} pages, {len(full_text)} characters")
    return full_text

def extract_text_from_txt(txt_path: str) -> str:
    """
    Extract text from text file
    """
    print(f"📄 Loading text file: {txt_path}")
    with open(txt_path, 'r', encoding='utf-8') as f:
        text = f.read()
    print(f"   Extracted {len(text)} characters")
    return text

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
    """
    Split text into overlapping chunks
    """
    print(f"🔪 Chunking text...")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    
    chunks = text_splitter.split_text(text)
    
    print(f"   Created {len(chunks)} chunks")
    print(f"   Average chunk length: {sum(len(c) for c in chunks)//len(chunks)} chars")
    
    # Optional: Print first chunk as sample
    if chunks:
        print(f"\n📝 Sample chunk (first 200 chars):")
        print(f"   {chunks[0][:200]}...")
    
    return chunks

def create_embeddings(chunks, model_name: str = EMBEDDER_MODEL):
    """
    Create embeddings for chunks using sentence-transformers
    """
    print(f"🤖 Loading embedding model: {model_name}")
    embedder = SentenceTransformer(model_name)
    
    print(f"   Creating embeddings for {len(chunks)} chunks...")
    embeddings = embedder.encode(chunks, show_progress_bar=True)
    
    print(f"   Embedding shape: {embeddings.shape}")
    print(f"   Embedding dimension: {embeddings.shape[1]}")
    
    return embeddings, embedder

def save_chunks_and_embeddings(chunks, embeddings):
    """
    Save chunks and embeddings to disk
    """
    # Create data directory if it doesn't exist
    os.makedirs(os.path.dirname(CHUNKS_PATH), exist_ok=True)
    
    # Save chunks as JSON
    print(f"💾 Saving chunks to {CHUNKS_PATH}")
    with open(CHUNKS_PATH, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    
    # Save embeddings as numpy array
    print(f"💾 Saving embeddings to {EMBEDDINGS_PATH}")
    np.save(EMBEDDINGS_PATH, embeddings)
    
    print(f"✅ Saved {len(chunks)} chunks and embeddings of shape {embeddings.shape}")

def process_pdf_to_embeddings(pdf_path: str = PDF_PATH):
    """
    Complete pipeline: PDF -> chunks -> embeddings -> save
    """
    print("="*60)
    print("🚀 Starting PDF processing pipeline")
    print("="*60)
    
    # Step 1: Extract text from PDF
    if not os.path.exists(pdf_path):
        print(f"❌ Error: PDF file not found at {pdf_path}")
        print(f"   Please make sure '{pdf_path}' exists in the current directory")
        return False
    
    text = extract_text_from_pdf(pdf_path)
    
    # Step 2: Chunk the text
    chunks = chunk_text(text)
    
    if not chunks:
        print("❌ No chunks created. Check your PDF file.")
        return False
    
    # Step 3: Create embeddings
    embeddings, _ = create_embeddings(chunks)
    
    # Step 4: Save to disk
    save_chunks_and_embeddings(chunks, embeddings)
    
    print("="*60)
    print("✅ Pipeline complete! You can now run create_faiss.py")
    print("="*60)
    return True

def process_text_file_to_embeddings(txt_path: str):
    """
    Alternative: Process a text file instead of PDF
    """
    print("="*60)
    print(f"🚀 Processing text file: {txt_path}")
    print("="*60)
    
    # Step 1: Extract text from file
    if not os.path.exists(txt_path):
        print(f"❌ Error: File not found at {txt_path}")
        return False
    
    text = extract_text_from_txt(txt_path)
    
    # Step 2: Chunk the text
    chunks = chunk_text(text)
    
    # Step 3: Create embeddings
    embeddings, _ = create_embeddings(chunks)
    
    # Step 4: Save to disk
    save_chunks_and_embeddings(chunks, embeddings)
    
    print("✅ Pipeline complete!")
    return True

def verify_files():
    """
    Verify that all necessary files were created
    """
    print("\n🔍 Verifying created files:")
    
    files_to_check = [CHUNKS_PATH, EMBEDDINGS_PATH]
    all_exist = True
    
    for file_path in files_to_check:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            if file_path.endswith('.npy'):
                # For embeddings, also check shape
                try:
                    data = np.load(file_path)
                    print(f"   ✅ {file_path} - {size:,} bytes, shape: {data.shape}")
                except:
                    print(f"   ⚠️ {file_path} - {size:,} bytes (but couldn't load)")
            else:
                # For JSON, check number of chunks
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        chunks = json.load(f)
                    print(f"   ✅ {file_path} - {size:,} bytes, {len(chunks)} chunks")
                except:
                    print(f"   ⚠️ {file_path} - {size:,} bytes")
        else:
            print(f"   ❌ {file_path} - NOT FOUND")
            all_exist = False
    
    return all_exist

def test_embedding_model():
    """
    Quick test to make sure the embedding model works
    """
    print("\n🧪 Testing embedding model...")
    embedder = SentenceTransformer(EMBEDDER_MODEL)
    
    test_sentences = ["This is a test sentence.", "Another test sentence."]
    test_embeddings = embedder.encode(test_sentences)
    
    print(f"   ✅ Model loaded successfully")
    print(f"   Test embedding shape: {test_embeddings.shape}")
    return True

# -------------------------
# MAIN EXECUTION
# -------------------------
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare chunks and embeddings for FAISS')
    parser.add_argument('--pdf', type=str, default=PDF_PATH, 
                       help=f'Path to PDF file (default: {PDF_PATH})')
    parser.add_argument('--txt', type=str, 
                       help='Path to text file (use instead of PDF)')
    parser.add_argument('--chunk_size', type=int, default=CHUNK_SIZE,
                       help=f'Chunk size in characters (default: {CHUNK_SIZE})')
    parser.add_argument('--chunk_overlap', type=int, default=CHUNK_OVERLAP,
                       help=f'Chunk overlap (default: {CHUNK_OVERLAP})')
    
    args = parser.parse_args()
    
    # Update config with command line args
    CHUNK_SIZE = args.chunk_size
    CHUNK_OVERLAP = args.chunk_overlap
    
    # Test embedding model first
    if not test_embedding_model():
        print("❌ Embedding model test failed")
        exit(1)
    
    # Process file
    if args.txt:
        success = process_text_file_to_embeddings(args.txt)
    else:
        success = process_pdf_to_embeddings(args.pdf)
    
    if success:
        # Verify files were created
        if verify_files():
            print("\n🎉 All files created successfully!")
            print("\nNext step: Run 'python create_faiss.py' to build the FAISS index")
        else:
            print("\n⚠️ Some files are missing. Check the errors above.")