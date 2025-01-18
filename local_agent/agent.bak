# agent.py
from dataclasses import dataclass
from typing import List, Dict, Optional, Callable
import os
import yaml
import time
import json
import difflib
import threading
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import requests
from prompt_toolkit import prompt
from prompt_toolkit.history import FileHistory

# ============================================================================
# Configuration Classes
# ============================================================================

@dataclass
class ModelConfig:
    """Configuration for the LLM model"""
    name: str
    context_window: int
    temperature: float
    host: str = "http://localhost:11435"

@dataclass
class PathConfig:
    """Configuration for file paths and extensions"""
    exclude: List[str]
    watch_extensions: List[str]

@dataclass
class EditingConfig:
    """Configuration for document editing behavior"""
    backup: bool
    backup_dir: str
    require_approval: bool

class ConfigManager:
    """Handles configuration loading and management"""
    
    def __init__(self, project_path: str):
        self.project_path = project_path
        self.config_path = os.path.join(project_path, '.llama', 'config.yaml')
        self.load_config()

    def load_config(self) -> None:
        """Load configuration from file or create default"""
        try:
            with open(self.config_path, 'r') as f:
                config_data = yaml.safe_load(f)
        except FileNotFoundError:
            config_data = self._create_default_config()
            
        self.model_config = ModelConfig(**config_data['model'])
        self.path_config = PathConfig(**config_data['paths'])
        self.editing_config = EditingConfig(**config_data['editing'])

    def _create_default_config(self) -> Dict:
        """Create and save default configuration"""
        config = {
            'model': {
                'name': 'llama3.2',
                'context_window': 4096,
                'temperature': 0.7
            },
            'paths': {
                'exclude': ['.obsidian/*', '.git/*'],
                'watch_extensions': ['.md']
            },
            'editing': {
                'backup': True,
                'backup_dir': '.llama/backups',
                'require_approval': True
            }
        }
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        with open(self.config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        return config

# ============================================================================
# Document Management
# ============================================================================

class Document:
    """Represents a markdown document"""
    
    def __init__(self, path: str, content: str):
        self.path = path
        self.content = content
        self.filename = os.path.basename(path)
        self._modified_time = os.path.getmtime(path)

    @property
    def is_stale(self) -> bool:
        """Check if the document has been modified on disk"""
        return os.path.getmtime(self.path) > self._modified_time

class DocumentManager:
    """Handles document loading and management"""
    
    def __init__(self, project_path: str, path_config: PathConfig):
        self.project_path = project_path
        self.path_config = path_config
        self.docs_path = os.path.join(project_path, 'docs')
        self.documents: Dict[str, Document] = {}
        self.load_all_documents()

    def load_all_documents(self) -> None:
        """Load all markdown documents in the project"""
        self.documents.clear()
        
        for root, _, files in os.walk(self.docs_path):
            for file in files:
                if file.endswith(tuple(self.path_config.watch_extensions)):
                    path = os.path.join(root, file)
                    if not any(Path(path).match(pattern) for pattern in self.path_config.exclude):
                        self.load_document(path)

    def load_document(self, path: str) -> Document:
        """Load a single document"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            doc = Document(path, content)
            self.documents[path] = doc
            return doc
        except Exception as e:
            raise IOError(f"Failed to load document {path}: {str(e)}")

    def get_document(self, path: str) -> Optional[Document]:
        """Get a document by path, reloading if stale"""
        doc = self.documents.get(path)
        if doc and doc.is_stale:
            doc = self.load_document(path)
        return doc

    def create_document(self, filename: str) -> Document:
        """Create a new document"""
        if not filename.endswith('.md'):
            filename = f"{filename}.md"
            
        file_path = os.path.join(self.docs_path, filename)
        
        if os.path.exists(file_path):
            raise FileExistsError(f"File {filename} already exists")
            
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        content = f"# {filename[:-3].title()}\n\n"
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return self.load_document(file_path)
        except Exception as e:
            raise IOError(f"Failed to create document {filename}: {str(e)}")

# ============================================================================
# File Watching
# ============================================================================

class FileWatcher:
    """Manages file system monitoring"""
    
    def __init__(self, docs_path: str, on_file_changed: Callable[[str], None]):
        self.docs_path = docs_path
        self.on_file_changed = on_file_changed
        self.observer = Observer()
        self._stop_event = threading.Event()
        
    def __enter__(self):
        self.start()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        
    def start(self) -> None:
        """Start watching for file changes"""
        event_handler = self._create_event_handler()
        self.observer.schedule(event_handler, self.docs_path, recursive=True)
        self.observer.start()
        
    def stop(self) -> None:
        """Stop watching for file changes"""
        self._stop_event.set()
        self.observer.stop()
        self.observer.join(timeout=2)
        
    def _create_event_handler(self) -> FileSystemEventHandler:
        """Create event handler for file changes"""
        class MarkdownHandler(FileSystemEventHandler):
            def __init__(self, callback: Callable[[str], None]):
                self.callback = callback
                self._last_event_time = 0
                self._debounce_seconds = 1.0

            def on_modified(self, event):
                if event.src_path.endswith('.md'):
                    current_time = time.time()
                    if current_time - self._last_event_time > self._debounce_seconds:
                        self._last_event_time = current_time
                        self.callback(event.src_path)

        return MarkdownHandler(self.on_file_changed)

# ============================================================================
# LLM Client
# ============================================================================

class LlamaError(Exception):
    """Base exception for LlamaClient errors"""
    pass

class LlamaClient:
    """Handles LLM interactions"""
    
    DEFAULT_SYSTEM_PROMPT = (
        "You are an AI assistant that modifies markdown files. "
        "Your only job is to analyze the provided file content and suggest improvements. "
        "When suggesting improvements, provide explanations followed by a complete revised "
        "version of the file after a '---CONTENT---' marker."
    )
    
    CHAT_SYSTEM_PROMPT = (
        "You are a helpful AI assistant ready to discuss any topic. "
        "You provide clear, informative responses while being engaging and conversational. "
        "You can help with analysis, explanations, creative tasks, or general discussion."
    )

    EXEC_SYSTEM_PROMPT = (
        "You are an executive VP of Product at a major company reviewing the product artifacts from your team. "
        "You are ultimately responsible for the success of the product. "
        "You provide clear, informative feedback while being engaging and conversational. "
    )
    
    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config
        self.message_history = []  # Add message history storage
        
    def query_llama(
        self, 
        prompt: str, 
        persona: str = "default",
        context: Optional[List[Document]] = None,
        preserve_history: bool = True, 
    ) -> Dict:
        """
        Send query to Ollama.
        
        Args:
            prompt: The user's prompt
            persona: Which persona to use ('default', 'chat', 'exec_feedback')
            context: Optional list of documents to include in context
            preserve_history: Whether to use and update message history
        """
        system_prompt = self._build_system_prompt(persona, context)
        
        # Build messages including history if needed
        messages = []
        if preserve_history:
            messages.extend(self.message_history)
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = requests.post(
                f'{self.model_config.host}/api/generate',
                json={
                    "model": self.model_config.name,
                    "prompt": "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages]),
                    "system": system_prompt,
                    "temperature": self.model_config.temperature
                },
                stream=True,
                timeout=30
            )
            response.raise_for_status()
            result = self._handle_streaming_response(response)
            
            # Update history if needed
            if preserve_history:
                self.message_history.append({"role": "user", "content": prompt})
                self.message_history.append({"role": "assistant", "content": result["response"]})
            
            return result
            
        except requests.Timeout:
            raise LlamaError("Request to Llama timed out")
        except requests.RequestException as e:
            raise LlamaError(f"Failed to connect to Llama: {str(e)}")
            
    def _build_system_prompt(self, persona: str, context: Optional[List[Document]] = None) -> str:
        """Build system prompt based on persona and context"""
        # Get base prompt based on persona
        if persona == 'chat':
            system_prompt = self.CHAT_SYSTEM_PROMPT
        elif persona == 'exec_feedback':
            system_prompt = self.EXEC_SYSTEM_PROMPT
        else:
            system_prompt = self.DEFAULT_SYSTEM_PROMPT
        
        # Add context if provided
        if context:
            system_prompt += "\n\nCurrent project context:\n"
            system_prompt += "\n".join([
                f"File: {doc.filename}\n{doc.content}\n---" 
                for doc in context
            ])
            
        return system_prompt
        
    def _handle_streaming_response(self, response: requests.Response) -> Dict:
        """Handle streaming response from Llama"""
        full_response = ""
        
        for line in response.iter_lines():
            if line:
                try:
                    json_response = json.loads(line)
                    if 'response' in json_response:
                        chunk = json_response['response']
                        full_response += chunk
                        print(chunk, end='', flush=True)
                except json.JSONDecodeError:
                    continue
                    
        if not full_response:
            raise LlamaError("Empty response from Llama")
                    
        print()  # Add newline at the end
        return {"response": full_response}

    def clear_history(self) -> None:
        """Clear the message history"""
        self.message_history = []

# ============================================================================
# Main Application
# ============================================================================

class LlamaAgent:
    """Main application class that coordinates all components"""
    
    def __init__(self, project_path: str):
        self.project_path = project_path
        self.config_manager = ConfigManager(project_path)
        
        self.document_manager = DocumentManager(
            project_path,
            self.config_manager.path_config
        )
        
        self.llm_client = LlamaClient(self.config_manager.model_config)
        
        self.file_watcher = FileWatcher(
            os.path.join(project_path, 'docs'),
            self._on_file_changed
        )
        
    def _on_file_changed(self, path: str) -> None:
        """Handle file change events"""
        try:
            self.document_manager.load_document(path)
            print(f"\nFile updated: {path}")
            print("Continue your last command or enter a new one.")
        except Exception as e:
            print(f"\nError reloading file: {str(e)}")

    def start(self) -> None:
        """Start the application"""
        try:
            with self.file_watcher:
                self._run_interactive_session()
        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            self.cleanup()

    def _run_interactive_session(self) -> None:
        """Run the interactive command session"""
        history = FileHistory(os.path.join(self.project_path, '.llama', '.history'))
        
        print("Welcome to LlamaAgent! Type 'help' for available commands.")
        
        while True:
            try:
                command = prompt('> ', history=history).strip()
                
                if not command:
                    continue
                    
                if command == 'exit':
                    break
                    
                self._handle_command(command)
                    
            except KeyboardInterrupt:
                continue
            except EOFError:
                break

    def _handle_command(self, command: str) -> None:
        """Handle user commands"""
        if command == 'help':
            self._show_help()
        elif command == 'list':
            self._handle_list_command()
        elif command == 'archive':
            self._handle_archive_command()
        elif command == 'chat':
            self._handle_chat_command()
        elif command.startswith('new '):
            self._handle_new_command(command[4:])
        elif command.startswith('edit '):
            self._handle_edit_command(command[5:])
        else:
            print("Unknown command. Type 'help' for available commands.")

    def _handle_archive_command(self) -> None:
        """Handle the 'archive' command - clean up backup files"""
        backup_dir = os.path.join(
            self.project_path,
            self.config_manager.editing_config.backup_dir
        )
        
        if not os.path.exists(backup_dir):
            print("\nNo backup directory found.")
            return
            
        backup_count = 0
        try:
            # Walk through backup directory and remove .bak files
            for root, _, files in os.walk(backup_dir):
                for file in files:
                    if file.endswith('.bak'):
                        file_path = os.path.join(root, file)
                        os.remove(file_path)
                        backup_count += 1
                        
            # Remove empty directories
            for root, dirs, files in os.walk(backup_dir, topdown=False):
                for dir_name in dirs:
                    dir_path = os.path.join(root, dir_name)
                    if not os.listdir(dir_path):  # if directory is empty
                        os.rmdir(dir_path)
                        
            if backup_count > 0:
                print(f"\nCleaned up {backup_count} backup files.")
            else:
                print("\nNo backup files found.")
                
        except Exception as e:
            print(f"\nError cleaning up backups: {str(e)}")

    def _handle_chat_command(self) -> None:
        """Handle the 'chat' command - interactive conversation with Llama"""
        print("\nEntering chat mode with Llama. Type 'exit' to return to main menu.")
        print("Context: You can ask Llama questions or discuss topics.")
        
        # Clear history when starting new chat
        self.llm_client.clear_history()
        
        while True:
            try:
                user_input = input('\nchat> ').strip()
                
                if not user_input:
                    continue
                    
                if user_input.lower() == 'exit':
                    print("\nExiting chat mode...")
                    break
                
                try:
                    response = self.llm_client.query_llama(
                        prompt=user_input,
                        persona="chat",
                        preserve_history=True  # Enable history for chat mode
                    )
                except LlamaError as e:
                    print(f"\nError: {str(e)}")
                    
            except KeyboardInterrupt:
                print("\nExiting chat mode...")
                break

    def _handle_list_command(self) -> None:
        """Handle the 'list' command"""
        documents = self.document_manager.documents
        
        if not documents:
            print("\nNo documents found.")
            return
            
        print("\nAvailable documents:")
        # Sort documents by filename for consistent display
        for path, doc in sorted(documents.items(), key=lambda x: x[1].filename):
            # Calculate relative path from docs directory for cleaner display
            rel_path = os.path.relpath(path, self.document_manager.docs_path)
            # Get first line of content for preview
            preview = doc.content.split('\n')[0][:60] + ('...' if len(doc.content.split('\n')[0]) > 60 else '')
            print(f"  {rel_path}")
            print(f"    {preview}")

    def _show_help(self) -> None:
        """Show available commands"""
        print("\nAvailable commands:")
        print("  new <filename>: Create a new markdown file")
        print("  edit <filename>: Edit an existing file")
        print("  list: Show all documents")
        print("  chat: Start an interactive chat with Llama")
        print("  archive: Clean up all backup files")
        print("  help: Show this help message")
        print("  exit: Exit the application")

    def _handle_new_command(self, filename: str) -> None:
        """Handle the 'new' command"""
        try:
            doc = self.document_manager.create_document(filename)
            print(f"\nCreated new file: {doc.filename}")
            print("Use 'edit' command to start editing")
        except (FileExistsError, IOError) as e:
            print(f"\nError: {str(e)}")

    def _handle_edit_command(self, filename: str) -> None:
        """Handle the 'edit' command"""
        if not filename.endswith('.md'):
            filename = f"{filename}.md"
            
        file_path = os.path.join(self.project_path, 'docs', filename)
        doc = self.document_manager.get_document(file_path)
        
        if not doc:
            print(f"\nError: File {filename} not found!")
            return
            
        print(f"\nEditing {filename}")
        print("\nAvailable commands:")
        print("  suggest: Get suggestions from Llama")
        print("  suggest <focus>: Get focused suggestions")
        print("  view: View current content")
        print("  save <content>: Save new content")
        print("  back: Return to main menu")
        
        while True:
            try:
                cmd = prompt('edit> ').strip()
                
                if not cmd:
                    continue
                    
                if cmd == 'back':
                    break
                    
                if cmd == 'view':
                    print("\nCurrent content:")
                    print(doc.content)
                    continue
                    
                if cmd.startswith('suggest'):
                    self._handle_suggest_command(cmd, doc)
                    continue
                    
                if cmd.startswith('save '):
                    self._handle_save_command(cmd[5:], doc)
                    continue
                    
                print("\nUnknown command. Type 'back' to return to main menu.")
                    
            except KeyboardInterrupt:
                break
    def _handle_suggest_command(self, cmd: str, doc: Document) -> None:
        """Handle the 'suggest' command in edit mode"""
        try:
            # Extract focus area if provided
            focus = "overall structure and content"
            if len(cmd) > 7:
                focus = cmd[8:]
            
            prompt_text = (
                f"Review this file and suggest improvements regarding {focus}. "
                f"Provide a complete revised version of the file after the suggestions, "
                f"starting with '---CONTENT---'."
            )
            
            print(f"\nGetting suggestions focused on {focus}...")
            response = self.llm_client.query_llama(
                prompt=prompt_text,
                context=[doc],  # Pass document context for suggestions
                persona="default"
            )
            
            print("\nWould you like to save these changes? (yes/no)")
            user_input = input('> ').lower()
            if user_input.startswith('y'):
                if '---CONTENT---' in response['response']:
                    new_content = response['response'].split('---CONTENT---', 1)[1].strip()
                    self._save_with_backup(doc, new_content)
                    print("\nChanges saved!")
                else:
                    print("\nError: Could not find content marker in response")
            
        except LlamaError as e:
            print(f"\nError getting suggestions: {str(e)}")

    def _handle_save_command(self, content: str, doc: Document) -> None:
        """Handle the 'save' command in edit mode"""
        try:
            self._save_with_backup(doc, content)
            print("\nChanges saved!")
        except IOError as e:
            print(f"\nError saving changes: {str(e)}")

    def _save_with_backup(self, doc: Document, new_content: str) -> None:
        """Save changes with backup if enabled"""
        if self.config_manager.editing_config.backup:
            # Create backup
            backup_dir = os.path.join(
                self.project_path,
                self.config_manager.editing_config.backup_dir
            )
            os.makedirs(backup_dir, exist_ok=True)
            
            backup_path = os.path.join(
                backup_dir,
                f"{doc.filename}.{int(time.time())}.bak"
            )
            
            with open(doc.path, 'r') as src, open(backup_path, 'w') as dst:
                dst.write(src.read())

        # Save new content
        with open(doc.path, 'w') as f:
            f.write(new_content)
            
        # Reload the document
        self.document_manager.load_document(doc.path)

    def cleanup(self) -> None:
        """Cleanup resources"""
        # File watcher is handled by context manager
        pass

if __name__ == "__main__":
    agent = LlamaAgent(os.path.dirname(os.path.abspath(__file__)))
    agent.start()