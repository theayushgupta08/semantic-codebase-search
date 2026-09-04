"""
AST parsing module using tree-sitter.
Supports multi-language code chunking.
"""
import os
import uuid
from typing import List, Dict, Optional
import tree_sitter

# Import all supported grammars
try:
    import tree_sitter_python
    import tree_sitter_javascript
    import tree_sitter_typescript
    import tree_sitter_java
    import tree_sitter_go
    import tree_sitter_cpp
    import tree_sitter_scala
    import tree_sitter_html
    import tree_sitter_css
except ImportError as e:
    print(f"Warning: Missing tree-sitter bindings. Run pip install again. Error: {e}")

LANGUAGE_MAP = {
    ".py": ("python", tree_sitter_python.language),
    ".js": ("javascript", tree_sitter_javascript.language),
    ".jsx": ("javascript", tree_sitter_javascript.language),
    ".ts": ("typescript", tree_sitter_typescript.language_typescript),
    ".tsx": ("tsx", tree_sitter_typescript.language_tsx),
    ".java": ("java", tree_sitter_java.language),
    ".go": ("go", tree_sitter_go.language),
    ".cpp": ("cpp", tree_sitter_cpp.language),
    ".cc": ("cpp", tree_sitter_cpp.language),
    ".cxx": ("cpp", tree_sitter_cpp.language),
    ".hpp": ("cpp", tree_sitter_cpp.language),
    ".h": ("cpp", tree_sitter_cpp.language),
    ".scala": ("scala", tree_sitter_scala.language),
    ".html": ("html", tree_sitter_html.language),
    ".css": ("css", tree_sitter_css.language),
}

# Define which AST node types represent a semantic chunk per language
CHUNK_NODES = {
    "python": {"function_definition", "class_definition"},
    "javascript": {"function_declaration", "class_declaration", "method_definition", "arrow_function", "generator_function_declaration"},
    "typescript": {"function_declaration", "class_declaration", "method_definition", "arrow_function", "interface_declaration"},
    "tsx": {"function_declaration", "class_declaration", "method_definition", "arrow_function", "interface_declaration"},
    "java": {"method_declaration", "class_declaration", "constructor_declaration", "interface_declaration"},
    "go": {"function_declaration", "method_declaration", "type_declaration"},
    "cpp": {"function_definition", "class_specifier", "struct_specifier"},
    "scala": {"function_definition", "class_definition", "object_definition", "trait_definition"},
    "html": {"script_element", "style_element", "element"},  # In HTML we'll extract major elements
    "css": {"rule_set"}
}

def _get_node_text(node, source_bytes: bytes) -> str:
    return source_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace")

def _init_parser(lang_fn, lang_name: str):
    try:
        # tree-sitter >= 0.22
        language = tree_sitter.Language(lang_fn())
        parser = tree_sitter.Parser(language)
    except Exception:
        # fallback for older versions
        language = tree_sitter.Language(lang_fn(), lang_name)
        parser = tree_sitter.Parser()
        parser.set_language(language)
    return parser

def extract_code_chunks(file_path: str) -> List[Dict]:
    """
    Parse a file and extract structural code components across multiple languages.
    Falls back to whole-file chunking if the language is unsupported or parse fails.
    """
    chunks = []
    
    if not os.path.exists(file_path):
        return chunks

    try:
        with open(file_path, "rb") as f:
            source_bytes = f.read()
    except Exception:
        return chunks
        
    try:
        source_code = source_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return chunks

    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    # Fallback to whole-file chunking for unsupported extensions
    if ext not in LANGUAGE_MAP:
        return _fallback_chunking(file_path, source_code)

    lang_name, lang_fn = LANGUAGE_MAP[ext]
    target_nodes = CHUNK_NODES.get(lang_name, set())

    try:
        parser = _init_parser(lang_fn, lang_name)
        tree = parser.parse(source_bytes)
    except Exception as e:
        # If parsing fails, fallback
        return _fallback_chunking(file_path, source_code)
        
    def traverse(node, current_parent=None):
        if node.type in target_nodes:
            # Attempt to extract a reasonable name
            name_node = node.child_by_field_name("name")
            if not name_node:
                # Sometimes the name is the first identifier
                for child in node.children:
                    if child.type == "identifier":
                        name_node = child
                        break
            
            name = _get_node_text(name_node, source_bytes) if name_node else "anonymous"
            full_name = f"{current_parent}.{name}" if current_parent else name
            
            # Start and end lines (1-indexed)
            start_line = node.start_point[0] + 1
            end_line = node.end_point[0] + 1
            
            chunk = {
                "chunk_id": str(uuid.uuid4()),
                "file_path": file_path,
                "name": full_name,
                "type": node.type,
                "content": _get_node_text(node, source_bytes),
                "lines": f"{start_line}-{end_line}"
            }
            chunks.append(chunk)
            
            # Keep track of class/object context
            if "class" in node.type or "type" in node.type or "interface" in node.type:
                current_parent = name
                
        # HTML gets very noisy if we traverse every single nested element. 
        # Only recurse if we are not currently in a target node, OR if it's a structural wrapper.
        # For simplicity across languages, we recurse into everything.
        for child in node.children:
            traverse(child, current_parent)

    traverse(tree.root_node)
    
    # If the parser yielded absolutely no chunks (e.g. empty classes, or no functions just scripts)
    if not chunks and source_code.strip():
        return _fallback_chunking(file_path, source_code)
    
    return chunks

def _fallback_chunking(file_path: str, source_code: str) -> List[Dict]:
    """
    Fallback chunker that splits a file into a single chunk if it's not AST-parseable,
    ensuring it's still searchable. (Can be improved to split by line counts).
    """
    lines = source_code.splitlines()
    if not lines:
        return []
        
    return [{
        "chunk_id": str(uuid.uuid4()),
        "file_path": file_path,
        "name": os.path.basename(file_path),
        "type": "file_chunk",
        "content": source_code,
        "lines": f"1-{len(lines)}"
    }]
