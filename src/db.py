"""
LanceDB vector manager.
"""
import json
import hashlib
from typing import List, Dict
import lancedb
import pyarrow as pa

# PyArrow schema for storing code chunks
SCHEMA = pa.schema([
    pa.field("id", pa.string()),
    pa.field("file_path", pa.string()),
    pa.field("content", pa.string()),
    pa.field("metadata", pa.string()),
    pa.field("vector", pa.list_(pa.float32(), 3072)) # gemini-embedding-2 dimensions
])

class DatabaseManager:
    def __init__(self, db_path: str = ".code_search_db"):
        self.db_path = db_path
        self.db = lancedb.connect(self.db_path)
        self.table_name = "code_chunks"
        
    def _get_table(self):
        """Retrieve or create the lancedb table."""
        if self.table_name not in self.db.table_names():
            return self.db.create_table(self.table_name, schema=SCHEMA)
        return self.db.open_table(self.table_name)
        
    def upsert_chunks(self, chunks: List[Dict]):
        """Add or update embedding records to the vector database."""
        table = self._get_table()
        
        records = []
        for chunk in chunks:
            content_hash = hashlib.sha256(chunk["content"].encode('utf-8')).hexdigest()
            metadata = {
                "name": chunk.get("name", ""),
                "type": chunk.get("type", ""),
                "lines": chunk.get("lines", ""),
                "hash": content_hash
            }
            # Only add to records if we actually have a vector to insert
            if "vector" in chunk:
                records.append({
                    "id": chunk.get("chunk_id", ""),
                    "file_path": chunk.get("file_path", ""),
                    "content": chunk.get("content", ""),
                    "metadata": json.dumps(metadata),
                    "vector": chunk["vector"]
                })
            
        if records:
            # Perform upsert based on the 'id' field
            # Use add() for the initial insertion to prevent LanceDB spill errors on empty tables
            try:
                if table.count_rows() == 0:
                    table.add(records)
                else:
                    table.merge_insert("id") \
                        .when_matched_update_all() \
                        .when_not_matched_insert_all() \
                        .execute(records)
            except Exception as e:
                print(f"Database insertion error: {e}")
                
    def delete_chunks(self, chunk_ids: List[str]):
        """Delete specific chunks from the database by their IDs."""
        if not chunk_ids:
            return
            
        table = self._get_table()
        # LanceDB uses sql syntax for deletions
        id_list_str = ", ".join([f"'{cid}'" for cid in chunk_ids])
        try:
            table.delete(f"id IN ({id_list_str})")
        except Exception as e:
            print(f"Failed to delete chunks: {e}")
                
    def search_similar(self, query_vector: List[float], top_k: int = 5) -> List[Dict]:
        """Search the database for closest matches."""
        table = self._get_table()
        
        results = table.search(query_vector).limit(top_k).to_list()
        return results
