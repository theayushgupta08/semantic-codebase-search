# Semantic Codebase Search 🔍

A blazingly fast, lightweight, and fully local CLI tool that brings intelligent semantic search to any modern codebase (Python, JavaScript, TypeScript, Java, Go, C++, Scala, HTML, and CSS). 

Stop wrestling with `grep` or trying to remember exact variable names. Ask your codebase questions in plain English—like *"Where is the API Key stored?"* or *"How does the tree traversal work?"*—and get instant, highlighted code snippets mapped precisely to their files and line numbers.

---

## 🚀 How It Helps You

Modern codebases are sprawling and complex. Finding the exact function that implements a specific piece of business logic often requires deep domain knowledge or tedious manual searching. 

**Semantic Codebase Search** solves this by:
- **Understanding Intent:** It searches for *meaning*, not just exact keyword matches.
- **AST-Aware Chunking:** Unlike dumb chunkers that split files arbitrarily by line count, this tool parses the Abstract Syntax Tree (AST) to index code precisely by functions, methods, and classes.
- **Incremental Indexing:** It hashes your files locally so that re-indexing a massive codebase takes milliseconds—only processing the files you've actively changed.

### Why not just use built-in IDE AI Agents?
While tools like GitHub Copilot or Cursor are fantastic, they often require you to stay locked inside a specific editor, rely on cloud-syncing your entire codebase to third-party servers, or suffer from limited context windows. 
This CLI tool is:
1. **Editor Agnostic:** Use it in any terminal, alongside Vim, Emacs, VSCode, or whatever you prefer.
2. **Infinitely Scalable:** Backed by LanceDB, it can search millions of vector chunks across massive monorepos in milliseconds.
3. **Fully Yours:** The vector database is stored locally inside your project folder (`.code_search_db`).

---

## 🛠️ Tech Stack

This tool is built on a modern, high-performance stack:
- **[LanceDB](https://lancedb.com/):** An open-source vector database built on Apache Arrow, designed for zero-copy, lightning-fast similarity search directly on your local disk.
- **[Tree-Sitter](https://tree-sitter.github.io/tree-sitter/):** An incremental parsing system that dynamically reads your code's AST to extract precise semantic boundaries (like `arrow_function` in JS, `element` in HTML, or `class_definition` in Python) across 9+ programming languages.
- **[Google Gemini 2.0 (`gemini-embedding-2`)](https://ai.google.dev/):** State-of-the-art embedding models to map your code semantics into 3072-dimensional vector space.
- **[Typer](https://typer.tiangolo.com/) & [Rich](https://rich.readthedocs.io/):** For a beautiful, colorful, and intuitive command-line interface.

---

## 📦 Installation

Since this is a fully packaged Python CLI, you can install it globally on your machine directly from GitHub:

```bash
pip install git+https://github.com/rsl-ayush2gupta/semantic-codebase-search.git
```

*(This will make the `code-search` command available everywhere in your terminal!)*

### Configuration

The tool uses Google's Gemini API to generate embeddings. You need a free API key from [Google AI Studio](https://aistudio.google.com/).

Export the key in your terminal profile (`~/.zshrc`, `~/.bashrc`, or `~/.profile`):
```bash
export GEMINI_API_KEY="your_api_key_here"
```
*(Alternatively, you can place a `.env` file containing `GEMINI_API_KEY=...` in the directory you are indexing).*

---

## 💻 Usage

Navigate to any project on your machine (whether it's a React web app, a Go backend, or a Python script) and start exploring!

### 1. Indexing the Codebase
Build the semantic index for the current directory. It will recursively parse all supported source files (`.py`, `.js`, `.ts`, `.html`, `.css`, `.go`, `.java`, `.cpp`, `.scala`), compute embeddings, and store them inside an isolated `.code_search_db` folder within the project.
```bash
code-search index .
```
*(You can also pass an absolute path like `code-search index /path/to/project`).*

### 2. Semantic Searching
Query your indexed codebase using natural language. The tool will calculate L2 distance across the vector space and return the top matching snippets with gorgeous syntax highlighting.
```bash
code-search search "How is the AST parsing implemented?" --top-k 3
```

### 3. Checking Status
View the health of your local LanceDB connection, disk size, and total chunks tracked:
```bash
code-search status
```

---

## 🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to fork the repository and submit a pull request.
