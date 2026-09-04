import os
import hashlib
import json
from typing import List, Dict
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, MofNCompleteColumn

from src.parser import extract_code_chunks
from src.embeddings import get_embedding
from src.db import DatabaseManager
from src.state import StateManager

# Load env variables manually for API keys
if os.path.exists(".env"):
    with open(".env", "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                key, val = line.split("=", 1)
                os.environ[key] = val

app = typer.Typer(help="Semantic Codebase Search CLI")
console = Console()

def get_file_hash(file_path: str) -> str:
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

def get_source_files(directory: str) -> List[str]:
    source_files = []
    supported_exts = {".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".go", ".cpp", ".cc", ".cxx", ".hpp", ".h", ".scala", ".html", ".css"}
    for root, _, files in os.walk(directory):
        if ".git" in root or "venv" in root or "__pycache__" in root or ".code_search_db" in root or "node_modules" in root:
            continue
        for file in files:
            _, ext = os.path.splitext(file)
            if ext.lower() in supported_exts:
                source_files.append(os.path.join(root, file))
    return source_files

def get_dir_size(path='.'):
    total = 0
    with os.scandir(path) as it:
        for entry in it:
            if entry.is_file():
                total += entry.stat().st_size
            elif entry.is_dir():
                total += get_dir_size(entry.path)
    return total

@app.command()
def index(
    path: str = typer.Argument(".", help="Path to the directory to index"),
):
    """
    Index a codebase for semantic search.
    """
    path = os.path.abspath(path)
    if not os.path.exists(path):
        console.print(f"[red]Error: Path '{path}' does not exist.[/red]")
        raise typer.Exit(code=1)
        
    console.print(f"[bold green]Indexing codebase at:[/bold green] {path}")
    
    db_path = os.path.join(path, ".code_search_db")
    state_path = os.path.join(path, ".index_state.json")
    
    db = DatabaseManager(db_path=db_path)
    state_manager = StateManager(state_file=state_path)
    
    current_files = get_source_files(path)
    tracked_files = state_manager.get_all_tracked_files()
    
    new_files = []
    modified_files = []
    unchanged_files = 0
    file_hashes = {}
    
    with console.status("[bold cyan]Scanning directory and calculating hashes..."):
        for fpath in current_files:
            fhash = get_file_hash(fpath)
            file_hashes[fpath] = fhash
            state = state_manager.get_file_state(fpath)
            
            if not state:
                new_files.append(fpath)
            elif state.get("hash") != fhash:
                modified_files.append(fpath)
            else:
                unchanged_files += 1
                
        deleted_files = [f for f in tracked_files if f not in current_files]
        
    console.print(f"Found [green]{len(new_files)} new[/green], [yellow]{len(modified_files)} modified[/yellow], [red]{len(deleted_files)} deleted[/red], and {unchanged_files} unchanged files.")
    
    total_to_process = len(new_files) + len(modified_files) + len(deleted_files)
    
    if total_to_process == 0:
        console.print("[yellow]Index is already up to date.[/yellow]")
        return
        
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Updating index...", total=total_to_process)
        
        # 1. Process deleted files
        for fpath in deleted_files:
            state = state_manager.get_file_state(fpath)
            if state and "chunk_ids" in state:
                db.delete_chunks(state["chunk_ids"])
            state_manager.remove_file_state(fpath)
            progress.advance(task)
            
        # 2. Process modified files
        for fpath in modified_files:
            state = state_manager.get_file_state(fpath)
            if state and "chunk_ids" in state:
                db.delete_chunks(state["chunk_ids"])
                
            chunks = extract_code_chunks(fpath)
            chunk_ids = []
            valid_chunks = []
            for chunk in chunks:
                try:
                    chunk["vector"] = get_embedding(chunk["content"])
                    chunk_ids.append(chunk["chunk_id"])
                    valid_chunks.append(chunk)
                except Exception as e:
                    console.print(f"\n[red]Failed to embed chunk in {fpath}: {e}[/red]")
                    
            if valid_chunks:
                db.upsert_chunks(valid_chunks)
                
            state_manager.update_file_state(fpath, file_hashes[fpath], chunk_ids)
            progress.advance(task)
            
        # 3. Process new files
        for fpath in new_files:
            chunks = extract_code_chunks(fpath)
            chunk_ids = []
            valid_chunks = []
            for chunk in chunks:
                try:
                    chunk["vector"] = get_embedding(chunk["content"])
                    chunk_ids.append(chunk["chunk_id"])
                    valid_chunks.append(chunk)
                except Exception as e:
                    console.print(f"\n[red]Failed to embed chunk in {fpath}: {e}[/red]")
            if valid_chunks:
                db.upsert_chunks(valid_chunks)
            state_manager.update_file_state(fpath, file_hashes[fpath], chunk_ids)
            progress.advance(task)

    console.print("[bold green]Indexing complete.[/bold green]")


@app.command()
def search(
    query: str = typer.Argument(..., help="Semantic query to search for"),
    path: str = typer.Option(".", "--path", "-p", help="Path to the indexed directory"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of results to return"),
):
    """
    Search the indexed codebase.
    """
    path = os.path.abspath(path)
    db_path = os.path.join(path, ".code_search_db")
    
    if not os.path.exists(db_path):
        console.print(f"[red]Error: No index found at '{path}'. Run 'code-search index {path}' first.[/red]")
        raise typer.Exit(code=1)
        
    db = DatabaseManager(db_path=db_path)
    
    with console.status("[bold cyan]Generating query embedding..."):
        try:
            query_vector = get_embedding(query)
        except Exception as e:
            console.print(f"[red]Error generating embedding: {e}[/red]")
            raise typer.Exit(1)
        
    with console.status("[bold cyan]Searching database..."):
        try:
            results = db.search_similar(query_vector, top_k=top_k)
        except Exception as e:
            console.print(f"[red]Error searching database: {e}[/red]")
            raise typer.Exit(1)
        
    if not results:
        console.print("[yellow]No results found.[/yellow]")
        return
        
    console.print(f"\n[bold blue]Top {len(results)} results for:[/bold blue] '{query}'\n")
    
    for i, res in enumerate(results):
        distance = res.get("_distance", 0.0)
        # Convert L2 distance back to a roughly intuitive similarity percentage
        similarity = (1 - (distance**2)/2) * 100 if distance <= 2 else 0
        
        meta = json.loads(res.get("metadata", "{}"))
        file_path = res.get("file_path", "Unknown")
        lines = meta.get("lines", "Unknown")
        name = meta.get("name", "Unknown")
        content = res.get("content", "")
        
        _, ext = os.path.splitext(file_path)
        ext_map = {
            ".py": "python", ".js": "javascript", ".ts": "typescript", 
            ".html": "html", ".css": "css", ".java": "java", ".go": "go",
            ".cpp": "cpp", ".scala": "scala"
        }
        lexer = ext_map.get(ext.lower(), "text")
        syntax = Syntax(content, lexer, theme="monokai", line_numbers=True, word_wrap=True)
        
        header = f"[{i+1}/{len(results)}] [bold cyan]{file_path}[/bold cyan] : [magenta]Lines {lines}[/magenta] ({name})"
        footer = f"Similarity: {similarity:.1f}%"
        
        panel = Panel(
            syntax,
            title=header,
            subtitle=footer,
            title_align="left",
            subtitle_align="right",
            border_style="green"
        )
        console.print(panel)


@app.command()
def status(
    path: str = typer.Option(".", "--path", "-p", help="Path to the indexed directory"),
):
    """
    Show the status of the LanceDB index.
    """
    path = os.path.abspath(path)
    db_path = os.path.join(path, ".code_search_db")
    state_path = os.path.join(path, ".index_state.json")
    
    state_manager = StateManager(state_file=state_path)
    tracked = state_manager.get_all_tracked_files()
    
    db = DatabaseManager(db_path=db_path)
    try:
        table = db._get_table()
        total_chunks = table.count_rows()
    except Exception:
        total_chunks = 0
        
    if os.path.exists(db_path):
        size_bytes = get_dir_size(db_path)
        size_mb = size_bytes / (1024 * 1024)
    else:
        size_mb = 0.0
        
    table_display = Table(title="Index Status", border_style="cyan")
    table_display.add_column("Metric", style="bold blue")
    table_display.add_column("Value", style="magenta")
    
    table_display.add_row("Database Status", "Connected" if os.path.exists(db_path) else "Not Found")
    table_display.add_row("Total Files Tracked", str(len(tracked)))
    table_display.add_row("Total Vector Chunks", str(total_chunks))
    table_display.add_row("Database Disk Size (MB)", f"{size_mb:.2f}")
    
    console.print(table_display)

if __name__ == "__main__":
    app()
