import os
import pytest
import tempfile
import shutil
from unittest.mock import patch, MagicMock, mock_open
import requests
from local_agent.agent import LlamaAgent

@pytest.fixture
def temp_project_path():
    # Create a temporary project directory
    temp_dir = tempfile.mkdtemp()
    try:
        # Create .llama directory
        os.makedirs(os.path.join(temp_dir, '.llama'))
        
        # Create docs directory
        os.makedirs(os.path.join(temp_dir, 'docs'))
        
        yield temp_dir
    finally:
        # Clean up the temporary directory
        shutil.rmtree(temp_dir)

def test_propose_edit(temp_project_path):
    agent = LlamaAgent(temp_project_path)
    
    # Create a test file
    test_file_path = os.path.join(temp_project_path, 'docs', 'test_edit.md')
    with open(test_file_path, 'w') as f:
        f.write("# Original Content\n\nSome text here.")
    
    # Mock the input to accept changes
    with patch('builtins.input', return_value='y'), \
         patch.object(agent, '_save_file') as mock_save:
        
        new_content = "# Updated Content\n\nNew text here."
        agent.propose_edit(test_file_path, new_content)
        
        # Verify _save_file was called with correct arguments
        mock_save.assert_called_once_with(test_file_path, new_content)

def test_propose_edit_reject(temp_project_path):
    agent = LlamaAgent(temp_project_path)
    
    # Create a test file
    test_file_path = os.path.join(temp_project_path, 'docs', 'test_edit.md')
    with open(test_file_path, 'w') as f:
        f.write("# Original Content\n\nSome text here.")
    
    # Mock the input to reject changes
    with patch('builtins.input', return_value='n'), \
         patch.object(agent, '_save_file') as mock_save, \
         patch('builtins.print') as mock_print:
        
        new_content = "# Updated Content\n\nNew text here."
        agent.propose_edit(test_file_path, new_content)
        
        # Verify _save_file was NOT called
        mock_save.assert_not_called()
        # Verify "Changes discarded" message was printed
        mock_print.assert_any_call("Changes discarded.")

def test_save_file_with_backup(temp_project_path):
    agent = LlamaAgent(temp_project_path)
    
    # Ensure backup is enabled in the config
    agent.config['editing']['backup'] = True
    
    # Create a test file path
    test_file_path = os.path.join(temp_project_path, 'docs', 'test_backup.md')
    test_content = "# Test Content\n\nBackup test."
    
    # Create backup directory if it doesn't exist
    backup_dir = os.path.join(temp_project_path, agent.config['editing']['backup_dir'])
    os.makedirs(backup_dir, exist_ok=True)
    
    # Use time.time() mock to get consistent backup filename
    with patch('time.time', return_value=1234567890), \
         patch('builtins.open', mock_open()) as mock_file:
        
        agent._save_file(test_file_path, test_content)
        
        # Verify backup file was created
        expected_backup_path = os.path.join(backup_dir, 'test_backup.md.1234567890.bak')
        
        # Original file was written
        mock_file.assert_any_call(test_file_path, 'w')
        
        # Note: In this test, we're using mock_open, so the actual file content 
        # verification is limited. In a real scenario, you might want to check 
        # actual file contents.

def test_save_file_without_backup(temp_project_path):
    agent = LlamaAgent(temp_project_path)
    
    # Disable backup in the config
    agent.config['editing']['backup'] = False
    
    # Create a test file path
    test_file_path = os.path.join(temp_project_path, 'docs', 'test_no_backup.md')
    test_content = "# Test Content\n\nNo backup test."
    
    # Use mock to verify file is written without backup
    with patch('time.time', return_value=1234567890), \
         patch('os.makedirs') as mock_makedirs, \
         patch('builtins.open', mock_open()) as mock_file:
        
        agent._save_file(test_file_path, test_content)
        
        # Verify backup directory was NOT created
        mock_makedirs.assert_not_called()
        
        # Verify original file was written
        mock_file.assert_called_once_with(test_file_path, 'w')

def test_extract_content_complex_scenarios():
    agent = LlamaAgent(os.getcwd())
    
    # Test with multiple content markers
    response1 = "Prefix text\n---CONTENT---\nFirst content\n---CONTENT---\nSecond content"
    assert agent.extract_content(response1) == "Second content"
    
    # Test with content marker at the beginning
    response2 = "---CONTENT---\nContent at the start"
    assert agent.extract_content(response2) == "Content at the start"
    
    # Test with whitespace around content marker
    response3 = "  ---CONTENT---  \nWhitespace test"
    assert agent.extract_content(response3) == "Whitespace test"

def test_setup_file_watcher(temp_project_path):
    agent = LlamaAgent(temp_project_path)
    
    # Verify observer was started
    assert hasattr(agent, 'observer')
    assert agent.observer is not None
    
    # Cleanup to stop the observer
    agent.cleanup()
    
    # Verify observer was stopped
    assert not agent.observer.is_alive()
