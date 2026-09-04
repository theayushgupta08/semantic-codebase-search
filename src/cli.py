import os
import hashlib
import json
import re
from typing import List, Dict, Optional
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

app = typer.Typer(help="Semantic Codebase Search CLI")
console = Console()

def load_env(target_dir: str = None) -> None:
    """
    Load environment variables from .env files in both the current working
    directory and the target directory, if different.
    """
    env_paths = [os.path.abspath(".env")]
    if target_dir:
        target_env = os.path.abspath(os.path.join(target_dir, ".env"))
        if target_env not in env_paths:
            env_paths.append(target_env)
            
    for env_path in env_paths:
        if os.path.isfile(env_path):
            try:
                with open(env_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            key, val = line.split("=", 1)
                            key = key.strip()
                            val = val.strip().strip("\"'")
                            if key:
                                os.environ[key] = val
            except Exception:
                pass

# Load env variables on startup
load_env()

def validate_api_key(target_path: str = None, api_key: str = None) -> str:
    """
    Validate that a Gemini API key is configured before proceeding.
    Checks explicit parameter, environment variables (GEMINI_API_KEY, GOOGLE_API_KEY), and .env files.
    Exits with code 1 if no valid key is found.
    """
    if api_key and api_key.strip():
        resolved_key = api_key.strip().strip("\"'")
        os.environ["GEMINI_API_KEY"] = resolved_key
        return resolved_key

    if target_path:
        load_env(target_path)
    else:
        load_env()

    api_key_val = (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or "").strip().strip("\"'")
    placeholder_keys = {
        "your_api_key_here",
        "your_gemini_api_key_here",
        "placeholder",
        "xxx",
        "your-key-here",
    }

    if not api_key_val or api_key_val.lower() in placeholder_keys:
        console.print("\n[bold red]Error: Gemini API key not found.[/bold red]")
        console.print("An API key is required to generate embeddings before indexing your codebase.\n")
        console.print("Please set the [bold cyan]GEMINI_API_KEY[/bold cyan] environment variable or add it to a [bold cyan].env[/bold cyan] file.")
        console.print("\n[bold]Configuration Options:[/bold]")
        console.print("  - [bold]Windows (PowerShell):[/bold] [yellow]$env:GEMINI_API_KEY=\"your_api_key_here\"[/yellow] [dim](do NOT use 'set' in PowerShell)[/dim]")
        console.print("  - [bold]Windows (CMD):[/bold]        [yellow]set GEMINI_API_KEY=\"your_api_key_here\"[/yellow]")
        console.print("  - [bold]Linux/macOS:[/bold]          [yellow]export GEMINI_API_KEY=\"your_api_key_here\"[/yellow]")
        console.print("  - [bold].env file:[/bold]            Place a [yellow].env[/yellow] file in the directory being indexed:")
        console.print("                          [yellow]GEMINI_API_KEY=your_api_key_here[/yellow]")
        console.print("  - [bold]CLI Flag:[/bold]             [yellow]code-search index . --api-key \"your_api_key_here\"[/yellow]")
        console.print("\n[dim]Get a free API key at:[/dim] [link=https://aistudio.google.com/]https://aistudio.google.com/[/link]\n")
        raise typer.Exit(code=1)

    return api_key_val

def parse_retry_delay(error_msg: str) -> Optional[str]:
    """Extract suggested retry delay time from API error messages if available."""
    match = re.search(r"retry in ([0-9.]+\s*s(?:econds?)?)", error_msg, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    match = re.search(r"['\"]retryDelay['\"]:\s*['\"]([^'\"]+)['\"]", error_msg)
    if match:
        return match.group(1).strip()
    return None

def handle_embedding_error(e: Exception, file_path: Optional[str] = None) -> None:
    """
    Display a clear, structured error message when embedding fails (e.g. rate limit / quota 429,
    authentication failure, or server error) and terminate the process cleanly.
    """
    err_str = str(e)
    code = getattr(e, "code", None)
    
    is_quota_error = (
        code == 429
        or "429" in err_str
        or "RESOURCE_EXHAUSTED" in err_str
        or "quota" in err_str.lower()
        or "rate limit" in err_str.lower()
    )
    
    is_auth_error = (
        code in (401, 403)
        or "401" in err_str
        or "403" in err_str
        or "PERMISSION_DENIED" in err_str
        or "UNAUTHENTICATED" in err_str
        or "API_KEY_INVALID" in err_str
        or "api key not valid" in err_str.lower()
    )

    console.print()
    if is_quota_error:
        console.print("[bold red]API Quota Exceeded (429 RESOURCE_EXHAUSTED)[/bold red]")
        console.print("You have exceeded your Gemini API request quota or rate limit for [bold cyan]gemini-embedding-2[/bold cyan].\n")
        
        if file_path:
            console.print(f"  - [bold]Failed File:[/bold] [yellow]{file_path}[/yellow]")
            
        retry_delay = parse_retry_delay(err_str)
        if retry_delay:
            console.print(f"  - [bold]Suggested Wait Time:[/bold] [cyan]{retry_delay}[/cyan]")
            
        if "free_tier" in err_str.lower() or "1000" in err_str:
            console.print("  - [bold]Quota Limit:[/bold] Free tier is limited to 1,000 embedding requests per day.")
            
        console.print("\n[bold]Next Steps:[/bold]")
        console.print("  1. [bold]Wait for quota reset:[/bold] Free tier requests reset daily (or wait a few seconds/minutes if per-minute rate limit).")
        console.print("  2. [bold]Switch API Key:[/bold] Use another project or paid key: [yellow]code-search index . --api-key \"YOUR_KEY\"[/yellow]")
        console.print("  3. [bold]Resume Anytime:[/bold] Previously completed files are already saved in the local database! Re-running will seamlessly continue.")
        console.print("\n[dim]Monitor usage & quotas at:[/dim] [link=https://ai.dev/rate-limit]https://ai.dev/rate-limit[/link]\n")
        
    elif is_auth_error:
        console.print("[bold red]API Authentication Error (401/403)[/bold red]")
        console.print("The Gemini API rejected your API key as invalid or unauthorized.\n")
        if file_path:
            console.print(f"  - [bold]Failed File:[/bold] [yellow]{file_path}[/yellow]")
        console.print("\n[bold]Next Steps:[/bold]")
        console.print("  - Verify your key at [link=https://aistudio.google.com/]https://aistudio.google.com/[/link]")
        console.print("  - Set a valid key: [yellow]$env:GEMINI_API_KEY=\"your_key\"[/yellow] or update your [yellow].env[/yellow] file.\n")
        
    else:
        console.print("[bold red]Error Generating Embedding[/bold red]")
        if file_path:
            console.print(f"  - [bold]Failed File:[/bold] [yellow]{file_path}[/yellow]")
        console.print(f"  - [bold]Details:[/bold] [red]{err_str}[/red]\n")
        console.print("[bold]Stopping indexing process to prevent corrupting state.[/bold]")
        console.print("Previously indexed files have been saved in the local database.\n")

    raise typer.Exit(code=1)

def get_file_hash(file_path: str) -> str:
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

IGNORED_DIRECTORIES = {
    ".git", "venv", ".venv", "env", "__pycache__", 
    ".code_search_db", ".test_code_search_db", "node_modules", 
    "target", "build", "dist", "out", "bin", "obj", 
    ".idea", ".vscode", ".pytest_cache", ".next", ".nuxt", "vendor"
}

def get_source_files(directory: str) -> List[str]:
    source_files = []
    supported_exts = {".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".go", ".cpp", ".cc", ".cxx", ".hpp", ".h", ".scala", ".html", ".css"}
    for root, dirs, files in os.walk(directory):
        # Exclude build, cache, and hidden directories from traversal
        dirs[:] = [d for d in dirs if d not in IGNORED_DIRECTORIES and not d.startswith(".")]
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

def ensure_gitignore(directory: str) -> None:
    """
    Ensure that generated files (.code_search_db/ and .index_state.json)
    are listed in the target directory's .gitignore file. Creates .gitignore if missing.
    """
    gitignore_path = os.path.join(directory, ".gitignore")
    required_entries = [
        (".code_search_db", ".code_search_db/"),
        (".index_state.json", ".index_state.json"),
    ]
    
    existing_lines = []
    content = ""
    if os.path.isfile(gitignore_path):
        try:
            with open(gitignore_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                existing_lines = [line.strip() for line in content.splitlines()]
        except Exception:
            return

    entries_to_add = []
    for base_name, entry in required_entries:
        if not any(line == base_name or line == f"{base_name}/" for line in existing_lines):
            entries_to_add.append(entry)

    if not entries_to_add:
        return

    try:
        with open(gitignore_path, "a", encoding="utf-8") as f:
            if content and not content.endswith("\n"):
                f.write("\n")
            if not content:
                f.write("# Semantic Codebase Search\n")
            elif "# Semantic Codebase Search" not in content:
                f.write("\n# Semantic Codebase Search\n")
            for entry in entries_to_add:
                f.write(f"{entry}\n")
    except Exception:
        pass

@app.command()
def index(
    path: str = typer.Argument(".", help="Path to the directory to index"),
    api_key: str = typer.Option(None, "--api-key", help="Gemini API key override"),
):
    """
    Index a codebase for semantic search.
    """
    path = os.path.abspath(path)
    if not os.path.exists(path):
        console.print(f"[red]Error: Path '{path}' does not exist.[/red]")
        raise typer.Exit(code=1)

    # Validate API key before starting indexing
    validate_api_key(path, api_key=api_key)

    # Ensure generated database and state files are in .gitignore
    ensure_gitignore(path)
        
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
                    handle_embedding_error(e, file_path=fpath)
                    
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
                    handle_embedding_error(e, file_path=fpath)
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
    api_key: str = typer.Option(None, "--api-key", help="Gemini API key override"),
):
    """
    Search the indexed codebase.
    """
    path = os.path.abspath(path)
    db_path = os.path.join(path, ".code_search_db")
    
    if not os.path.exists(db_path):
        console.print(f"[red]Error: No index found at '{path}'. Run 'code-search index {path}' first.[/red]")
        raise typer.Exit(code=1)

    validate_api_key(path, api_key=api_key)
        
    db = DatabaseManager(db_path=db_path)
    
    with console.status("[bold cyan]Generating query embedding..."):
        try:
            query_vector = get_embedding(query)
        except Exception as e:
            handle_embedding_error(e)
        
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
