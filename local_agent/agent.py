# agent.py
from typing import List, Dict, Optional, Callable
import os
import time
import json
import difflib
import threading
import asyncio
from .config import PathConfig, EditingConfig, AgentModelConfig, ConfigManager
from .vector_store import VectorStore
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import requests
from prompt_toolkit import prompt
from prompt_toolkit.history import FileHistory

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
# Agent LLM Client
# ============================================================================

class AgentLLMError(Exception):
    """Base exception for AgentLLMClient errors"""
    pass

class AgentLLMClient:
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
        "You can help with analysis, explanations, creative tasks, or general discussion.\n\n"
        "When provided with document content, you should:\n"
        "1. Reference relevant information from the provided documents\n"
        "2. Stay accurate to the document content when citing it\n"
        "3. Clearly indicate when you're using information from documents\n"
        "4. Provide helpful responses even when documents aren't relevant\n"
        "5. Feel free to combine information from multiple documents\n"
        "6. Acknowledge if the documents don't address the question\n"
        "7. Use phrases like 'According to [document]...' when citing"
    )

    EXEC_SYSTEM_PROMPT = (
        "You are an executive VP of Product at a major company reviewing the product artifacts from your team. "
        "You are ultimately responsible for the success of the product. "
        "You provide clear, informative feedback while being engaging and conversational. "
    )
    
    def __init__(self, agent_config: AgentModelConfig, project_path: str):
        self.agent_config = agent_config
        self.project_path = project_path
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
                f'{self.agent_config.host}/api/generate',
                json={
                    "model": self.agent_config.name,
                    "prompt": "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages]),
                    "system": system_prompt,
                    "temperature": self.agent_config.temperature
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
            raise AgentLLMError("Request to Llama timed out")
        except requests.RequestException as e:
            raise AgentLLMError(f"Failed to connect to Llama: {str(e)}")
            
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
            system_prompt += "\n\nRELEVANT DOCUMENTS:\n"
            for doc in context:
                rel_path = os.path.relpath(doc.path, os.path.join(self.project_path, 'docs'))
                system_prompt += f"\n[{rel_path}]\n"
                lines = doc.content.split('\n')
                
                # Add document content with clear structure
                if lines:
                    # Add title if first line is a header
                    if lines[0].startswith('#'):
                        system_prompt += f"Title: {lines[0].lstrip('#').strip()}\n"
                    
                    # Add content
                    system_prompt += "Content:\n"
                    system_prompt += '\n'.join(lines)
                    system_prompt += "\n---\n"
            
            # Add explicit instructions for using context
            system_prompt += "\nPlease reference these documents when they contain relevant information "
            system_prompt += "by saying 'According to [filename]' or similar phrases."
            
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
            raise AgentLLMError("Empty response from Llama")
                    
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
        
        self.llm_client = AgentLLMClient(
            self.config_manager.agent_config,
            project_path=project_path
        )
        
        self.vector_store = VectorStore(
            project_path=project_path,
            embedding_config=self.config_manager.embedding_config
        )
        
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
        
        print("Welcome to LlamaAgent!")
        self._show_help()
        
        while True:
            try:
                command = prompt('> ', history=history).strip()
                
                if not command:
                    continue
                if command == 'exit':
                    break
                    
                asyncio.run(self._handle_command(command))
                    
            except KeyboardInterrupt:
                continue
            except EOFError:
                break
    
    async def _handle_command(self, command: str) -> None:
        """Single command handler for both sync and async commands"""
        if command == 'help':
            self._show_help()
        elif command.startswith('edit'):
            await self._handle_edit_command(command[5:])
        elif command.startswith('new'):
            self._handle_new_command(command[4:])
        elif command == 'chat':
            await self._handle_chat_command()
        elif command == 'tools':
            await self._handle_tools_command()
        else:
            print("Unknown command. Type 'help' for available commands.")
        return None

    def _show_help(self) -> None:
        """Show available commands"""
        print("\nAvailable commands:")
        print("  new <filename>: Create a new markdown file")
        print("  edit <filename>: Edit an existing file")
        print("  chat: Start an interactive chat with Llama")
        print("  tools: Access troubleshooting and maintenance tools")
        print("  help: Show available commands")
        print("  exit: Exit the application\n")

    async def _handle_chat_command(self) -> None:
        """Handle the 'chat' command - interactive conversation with Llama with vector store support"""
        print("\nEntering chat mode with Llama. Type 'exit' to return to main menu.")
        print("Context: You can ask questions about your documents or discuss any topic.")
        print("I'll automatically search for relevant information in your documents.")
        
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
                    # Search vector store for relevant documents
                    print("\nSearching knowledge base...", end='', flush=True)
                    relevant_docs = await self.vector_store.query(user_input, n_results=3)
                    
                    if relevant_docs:
                        print(f"\rFound {len(relevant_docs)} relevant documents:")
                        for i, doc in enumerate(relevant_docs, 1):
                            similarity = 1 - doc['distance']  # Convert distance to similarity score
                            if similarity >= 0.7:  # Only show highly relevant docs
                                print(f"  {i}. {doc['metadata']['filename']} (relevance: {similarity:.0%})")
                    else:
                        print("\rNo highly relevant documents found.")
                    
                    # Build context from relevant documents
                    context_docs = []
                    for doc in relevant_docs:
                        similarity = 1 - doc['distance']
                        if similarity >= 0.7:  # Only use highly relevant docs
                            doc_path = Path(doc['metadata']['path'])
                            if doc_path.exists():
                                context_docs.append(self.document_manager.get_document(str(doc_path)))
                    
                    # Get LLM response with context
                    response = self.llm_client.query_llama(
                        prompt=user_input,
                        persona="chat",
                        context=context_docs,  # Pass relevant documents as context
                        preserve_history=True  # Enable history for chat mode
                    )
                    
                    # Add citations if documents were used
                    if context_docs:
                        print("\nRelevant files referenced:")
                        for doc in context_docs:
                            rel_path = os.path.relpath(doc.path, self.document_manager.docs_path)
                            print(f"- {rel_path}")
                        
                except Exception as e:
                    print(f"\nError: {str(e)}")
                    
            except KeyboardInterrupt:
                print("\nExiting chat mode...")
                break

    def _handle_new_command(self, filename: str) -> None:
        """Handle the 'new' command"""
        try:
            doc = self.document_manager.create_document(filename)
            print(f"\nCreated new file: {doc.filename}")
            print("Use 'edit' command to start editing")
        except (FileExistsError, IOError) as e:
            print(f"\nError: {str(e)}")

    async def _handle_edit_command(self, filename: str) -> None:
        """Handle the 'edit' command"""
        if not filename.endswith('.md'):
            filename = f"{filename}.md"
            
        file_path = os.path.join(self.project_path, 'docs', filename)
        doc = self.document_manager.get_document(file_path)
        
        if not doc:
            print(f"\nError: File {filename} not found!")
            return
            
        print(f"\nEditing {filename}")
        self._show_edit_help()
        
        while True:
            try:
                # Use ainput or a non-event-loop-based input method
                cmd = input('edit> ').strip()
                
                if not cmd:
                    continue
                if cmd == 'back':
                    break
                if cmd == 'view':
                    print("\nCurrent content:")
                    print(doc.content)
                    continue
                if cmd.startswith('suggest'):
                    await self._handle_suggest_command(cmd, doc)
                    continue
                if cmd.startswith('save '):
                    await self._handle_save_command(cmd[5:], doc)
                    continue
                if cmd == 'help':
                    self._show_edit_help()
                    continue
                print("\nUnknown command. Type 'back' to return to main menu.")
                    
            except KeyboardInterrupt:
                break

    def _show_edit_help(self) -> None:
        """Show help information for edit mode"""
        print("\nAvailable commands:")
        print("  suggest: Get suggestions from Llama")
        print("  suggest <focus>: Get focused suggestions")
        print("  view: View current content")
        print("  save <content>: Save new content")
        print("  help: Show available commands")
        print("  back: Return to main menu\n")

    async def _handle_suggest_command(self, cmd: str, doc: Document) -> None:
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
                    await self._save_with_backup(doc, new_content)
                    print("\nChanges saved!")
                else:
                    print("\nError: Could not find content marker in response")
            
        except AgentLLMError as e:
            print(f"\nError getting suggestions: {str(e)}")

    async def _handle_save_command(self, content: str, doc: Document) -> None:
        """Handle the 'save' command in edit mode"""
        try:
            await self._save_with_backup(doc, content)
            print("\nChanges saved!")
        except IOError as e:
            print(f"\nError saving changes: {str(e)}")

    async def _save_with_backup(self, doc: Document, new_content: str) -> None:
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

    async def _handle_tools_command(self) -> None:
        """Handle the 'tools' command - troubleshooting and maintenance tools"""
        print("\nEntering tools mode.")
        self._show_tools_help()
        
        while True:
            try:
                cmd = input('tools> ').strip()
                
                if not cmd:
                    continue
                if cmd == 'back':
                    break
                if cmd == 'update':
                    await self._update_knowledge()
                    continue
                if cmd == 'reset':
                    await self._reset_knowledge()
                    continue
                if cmd.startswith('search '):
                    await self._search_knowledge(cmd[7:])
                    continue
                if cmd == 'archive':
                    await self._handle_archive_command()
                    continue
                if cmd == 'list':
                    await self._handle_list_command()
                    continue
                if cmd == 'help':
                    self._show_tools_help()
                    continue
                print("\nUnknown command. Type 'back' to return to main menu.")
                    
            except KeyboardInterrupt:
                break

    def _show_tools_help(self) -> None:
        """Show available commands"""
        print("\nAvailable commands:")
        print("  update: Update knowledge base embeddings")
        print("  reset: Reset knowledge base")
        print("  search <query>: Search through documents")
        print("  archive: Clean up backup files")
        print("  list: List all documents")
        print("  help: Show available commands")
        print("  back: Return to main menu\n")

    async def _handle_list_command(self) -> None:
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

    async def _search_knowledge(self, query: str) -> None:
        """Search the vector store for relevant documents"""
        try:
            print("\nSearching knowledge base...")
            results = await self.vector_store.query(query)
            
            if not results:
                print("No relevant documents found.")
                return
                
            print(f"\nFound {len(results)} relevant documents:")
            for i, result in enumerate(results, 1):
                print(f"\n{i}. {result['metadata']['filename']} (similarity: {1 - result['distance']:.2f})")
                print("-" * 40)
                print(result['document'][:300] + "..." if len(result['document']) > 300 else result['document'])
        except Exception as e:
            print(f"\nError searching knowledge base: {str(e)}")
            
    async def _update_knowledge(self) -> None:
        """Update the vector store with document embeddings"""
        try:
            print("\nUpdating knowledge base...")
            docs_path = os.path.join(self.project_path, 'docs')
            await self.vector_store.add_documents(docs_path)
            print("Knowledge base updated successfully!")
        except Exception as e:
            print(f"\nError updating knowledge base: {str(e)}")

    async def _reset_knowledge(self) -> None:
        """Reset the vector store, clearing all documents"""
        try:
            print("\nResetting knowledge base...")
            self.vector_store.reset()
            print("Knowledge base reset successfully!")
        except Exception as e:
            print(f"\nError resetting knowledge base: {str(e)}")

    async def _handle_archive_command(self) -> None:
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

    def cleanup(self) -> None:
        """Cleanup resources"""
        # File watcher is handled by context manager
        pass

if __name__ == "__main__":
    agent = LlamaAgent(os.path.dirname(os.path.abspath(__file__)))
    agent.start()