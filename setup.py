from setuptools import setup, find_packages

setup(
    name="semantic-codebase-search",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "lancedb",
        "tree-sitter>=0.22.0",
        "tree-sitter-python>=0.21.0",
        "tree-sitter-javascript",
        "tree-sitter-typescript",
        "tree-sitter-java",
        "tree-sitter-go",
        "tree-sitter-cpp",
        "tree-sitter-scala",
        "tree-sitter-html",
        "tree-sitter-css",
        "google-genai",
        "rich",
        "typer",
    ],
    entry_points={
        "console_scripts": [
            "code-search=src.cli:app",
        ],
    },
)
