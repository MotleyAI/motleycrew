# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MotleyCrew is a Python framework for building multi-agent AI systems. It orchestrates AI agents and tools from various frameworks (Langchain, LlamaIndex, CrewAI, Autogen) using a knowledge graph backend and provides workflow management for complex AI tasks.

## Development Commands

### Setup and Installation
```bash
pip install poetry  # If not already installed
poetry install      # Install dependencies
```

### Testing
```bash
poetry run pytest                              # Run tests (excludes "fat" tests by default)  
poetry run pytest --cov --cov-report=term-missing  # Run tests with coverage
poetry run pytest -m fat                       # Run resource-intensive tests
make test                                       # Alternative test command
make cov                                        # Test with coverage using Makefile
```

### Code Quality and Linting
```bash
poetry run black .      # Format code
poetry run flake8 .     # Check style
poetry run mypy .       # Type checking
poetry run isort .      # Sort imports
```

### Build and Release
```bash
poetry build    # Build package
make build      # Alternative build command
make clean      # Clean build artifacts
make all        # Clean, test with coverage, and build
```

## Architecture

### Core Components

**MotleyCrew**: Main orchestration class that executes tasks and manages agents using a graph store backend.

**Tasks**: Abstract base classes for describing work units:
- `Task`: Generic task interface with knowledge graph integration
- `SimpleTask`: Basic task implementation with agent and description
- Tasks can be chained using `>>` operator for sequential execution

**Agents**: Wrapper classes for different AI agent frameworks:
- `motleycrew.agents.langchain/`: Langchain agent implementations
- `motleycrew.agents.llama_index/`: LlamaIndex agent implementations
- Support for ReAct, tool-calling, and other agent types

**Knowledge Graph**: Kuzu-based graph store for task orchestration and data persistence:
- `motleycrew.storage.kuzu_graph_store`: Primary graph store implementation
- `motleycrew.storage.graph_store_utils`: Utilities for graph operations
- Used for task dependencies, data flow, and system state

**Tools**: Integration layer for external capabilities:
- `motleycrew.tools.code/`: Code-related tools (Python REPL, linters)
- `motleycrew.tools.image/`: Image generation tools
- Tools implement Langchain's Runnable API for compatibility

### Key Patterns

- All components implement Langchain's Runnable interface for LCEL compatibility
- Tasks are stored and managed in a knowledge graph for complex workflow orchestration  
- Async execution support with configurable backends (asyncio, threading)
- Built-in caching (motleycache) and observability (Lunary) integrations

### Test Structure
- Unit tests in `tests/` organized by component
- Integration tests marked with `fat` marker for resource-intensive scenarios
- `conftest.py` files provide test fixtures and configuration