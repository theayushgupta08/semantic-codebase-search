import os
import shutil
import pytest
from src.embeddings import get_embedding
from src.db import DatabaseManager

def test_integration():
    db_path = ".test_code_search_db"
    
    # Ensure a clean slate for the test
    if os.path.exists(db_path):
        shutil.rmtree(db_path)
        
    try:
        # Skip if API key is not present, to prevent failing in CI/local runs without auth
        if "GEMINI_API_KEY" not in os.environ:
            pytest.skip("GEMINI_API_KEY not set")
            
        # 1. Test GenAI embedding generation
        text = "def hello_world():\n    print('Hello World!')"
        vector = get_embedding(text)
        assert len(vector) == 3072
        
        # 2. Test DB storage and schema compliance
        db = DatabaseManager(db_path=db_path)
        
        chunk = {
            "chunk_id": "test_chunk_1",
            "file_path": "test.py",
            "name": "hello_world",
            "type": "function",
            "content": text,
            "lines": "1-2",
            "vector": vector
        }
        
        db.upsert_chunks([chunk])
        
        # 3. Test vector similarity search
        query = "print hello"
        query_vector = get_embedding(query)
        
        results = db.search_similar(query_vector, top_k=1)
        
        assert len(results) > 0
        assert results[0]["id"] == "test_chunk_1"
        assert results[0]["content"] == text
        assert "hello_world" in results[0]["metadata"]
        
    finally:
        # Cleanup test db
        if os.path.exists(db_path):
            shutil.rmtree(db_path)
