import os
import tempfile
from src.parser import extract_code_chunks

def test_extract_code_chunks():
    code = '''
class MyClass:
    """Class docstring."""
    
    def my_method(self):
        """Method docstring."""
        print("Hello")

def my_function():
    print("World")
'''
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        temp_path = f.name
        
    try:
        chunks = extract_code_chunks(temp_path)
        
        assert len(chunks) == 3
        
        types = [c["type"] for c in chunks]
        names = [c["name"] for c in chunks]
        
        assert "class" in types
        assert "method" in types
        assert "function" in types
        
        assert "MyClass" in names
        assert "MyClass.my_method" in names
        assert "my_function" in names
        
        for chunk in chunks:
            assert chunk["file_path"] == temp_path
            assert "chunk_id" in chunk
            assert "lines" in chunk
            assert "content" in chunk
            assert "def" in chunk["content"] or "class" in chunk["content"]
            
    finally:
        os.remove(temp_path)

def test_invalid_file():
    chunks = extract_code_chunks("non_existent_file.py")
    assert chunks == []

def test_syntax_error():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write("def invalid_syntax( \n print('hi')")
        temp_path = f.name
        
    try:
        chunks = extract_code_chunks(temp_path)
        assert isinstance(chunks, list)
    finally:
        os.remove(temp_path)

def test_invalid_utf8():
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".py", delete=False) as f:
        f.write(b"def my_func():\n\tprint('hi')\n\xff\xff\xff")
        temp_path = f.name
        
    try:
        chunks = extract_code_chunks(temp_path)
        assert isinstance(chunks, list)
    finally:
        os.remove(temp_path)
