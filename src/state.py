import json
import os
from typing import Dict, List, Any

class StateManager:
    def __init__(self, state_file: str = ".index_state.json"):
        self.state_file = state_file
        self.state: Dict[str, Dict[str, Any]] = self.load_state()

    def load_state(self) -> Dict[str, Dict[str, Any]]:
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r") as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def save_state(self):
        with open(self.state_file, "w") as f:
            json.dump(self.state, f, indent=2)

    def get_file_state(self, file_path: str) -> Dict[str, Any]:
        """Return the state dict (hash and chunk_ids) for a given file path."""
        return self.state.get(file_path)

    def update_file_state(self, file_path: str, file_hash: str, chunk_ids: List[str]):
        """Update or insert state information for a processed file."""
        self.state[file_path] = {
            "hash": file_hash,
            "chunk_ids": chunk_ids
        }
        self.save_state()

    def remove_file_state(self, file_path: str):
        """Remove a file from the index tracking state."""
        if file_path in self.state:
            del self.state[file_path]
            self.save_state()
            
    def get_all_tracked_files(self) -> List[str]:
        """Return a list of all currently tracked files."""
        return list(self.state.keys())
