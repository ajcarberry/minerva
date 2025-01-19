# Minerva: AI-Powered Product Management Assistant

Minerva is an intelligent assistant designed to streamline product management workflows through automated research, documentation, and knowledge management. It combines the power of local LLM capabilities with persistent memory to enhance your product management process.

## Features

- 🧠 Intelligent Knowledge Management
  - Vector store integration for semantic search
  - Automatic context maintenance
  - Document ingestion and processing
  - Real-time file monitoring and updates

- 📝 Document Management
  - Markdown file editing and suggestions
  - Interactive command interface
  - Automated content improvements
  - Backup system for safe editing

- 🤖 AI Integration
  - Local LLM integration through Ollama
  - Context-aware responses
  - Intelligent document analysis
  - Automated improvement suggestions

- ⚙️ Customization
  - Configurable through YAML
  - Extensible architecture
  - Flexible document handling
  - Custom prompts support

## Prerequisites

- Python 3.8 or higher
- [Ollama](https://ollama.ai) installed and running
- Llama 2 model pulled in Ollama

## Installation

1. Install Poetry (package manager):
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

2. Clone the repository:
```bash
git clone https://github.com/ajcarberry/minerva.git
cd minerva
```

3. Install dependencies:
```bash
poetry install
```

4. Install the Llama model in Ollama:
```bash
ollama pull llama2
```

## Usage

1. Start Ollama:
```bash
nohup sh -c 'OLLAMA_HOST=127.0.0.1:11435 ollama serve' > ollama.log 2>&1 &
```

2. Run Minerva:
```bash
poetry run python agent.py
```

### Available Commands

#### Document Management
- `edit <filename>`: Enter edit mode for a specific file
- `context`: Show current document context
- `refresh`: Reload all documents
- `status`: Show current configuration and status

#### Knowledge Management
- `learn <directory>`: Ingest new documents into the knowledge base
- `query <text>`: Search the knowledge base using semantic search
- `suggest <filename>`: Get AI-powered improvement suggestions
- `chat`: Enter interactive chat mode with context awareness

#### System Commands
- `config`: View or modify configuration
- `backup`: Create a backup of current documents
- `exit`: Quit the application

## Configuration

Configuration is stored in `config.yaml`:

```yaml
model:
  name: llama2
  context_window: 4096
  temperature: 0.7

vector_store:
  engine: "faiss"  # Vector store backend
  dimension: 384   # Embedding dimension
  index_path: ".minerva/vector_store"

paths:
  exclude:
    - ".obsidian/*"
    - ".git/*"
  watch_extensions:
    - ".md"

editing:
  backup: true
  backup_dir: ".minerva/backups"
  require_approval: true
```

## Project Structure

```
minerva/
├── .minerva/
│   ├── config.yaml        # Configuration file
│   ├── vector_store/      # Vector store indices
│   └── backups/          # Document backups
├── agent/
│   ├── __init__.py
│   ├── agent.py          # Core agent functionality
│   ├── document.py       # Document management
│   ├── knowledge.py      # Knowledge management
│   └── ui.py            # User interface
├── docs/                # Documentation
└── README.md           # This file
```

## Development

This project uses Poetry for dependency management. To set up the development environment:

1. Install development dependencies:
```bash
poetry install --with dev
```

2. Activate the virtual environment:
```bash
poetry shell
```

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

[MIT License](LICENSE)

## Troubleshooting

### Common Issues

1. **Vector Store Initialization**
   - Ensure proper permissions for the vector store directory
   - Check if the embedding model is downloaded
   - Verify vector store configuration

2. **Ollama Connection**
   - Ensure Ollama is running (`ollama serve`)
   - Check if the Llama model is installed (`ollama list`)
   - Verify the OLLAMA_HOST configuration

3. **Document Processing**
   - Check file permissions
   - Verify paths in config.yaml
   - Ensure proper file encoding

### Getting Help

- Open an issue in the repository
- Check the Ollama documentation
- Review the Poetry documentation

## Roadmap

- [ ] Add support for multiple LLM models
- [ ] Implement concurrent document processing
- [ ] Add web interface
- [ ] Enhance semantic search capabilities
- [ ] Add document version control
- [ ] Implement collaborative editing
- [ ] Add custom workflow automation
- [ ] Enhance API integration capabilities