import os
import pytest
import yaml
import tempfile
import shutil
from unittest.mock import patch, MagicMock, mock_open
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

def test_get_staged_path(temp_project_path):
    agent = LlamaAgent(temp_project_path)
    original_path = os.path.join(temp_project_path, 'docs', 'test.md')
    staged_path = agent.get_staged_path(original_path)
    
    assert staged_path == os.path.join(temp_project_path, 'docs', '.staged_test.md')

def test_save_staged_changes(temp_project_path):
    agent = LlamaAgent(temp_project_path)
    original_path = os.path.join(temp_project_path, 'docs', 'test.md')
    new_content = "# Test Content\n\nThis is a test."
    
    agent.save_staged_changes(original_path, new_content)
    
    staged_path = agent.get_staged_path(original_path)
    assert os.path.exists(staged_path)
    
    with open(staged_path, 'r') as f:
        assert f.read() == new_content

def test_extract_content():
    agent = LlamaAgent(os.getcwd())
    
    # Test with content marker
    response1 = "Some initial text\n---CONTENT---\nActual content here"
    assert agent.extract_content(response1) == "Actual content here"
    
    # Test without content marker
    response2 = "Some response without marker"
    assert agent.extract_content(response2) == "Some response without marker"

@patch('requests.post')
def test_query_llama_integration(mock_post, temp_project_path):
    # Simulate a successful Ollama API response
    mock_response = MagicMock()
    mock_response.iter_lines.return_value = [
        b'{"response":"First "}',
        b'{"response":"chunk "}',
        b'{"response":"of text"}'
    ]
    mock_post.return_value = mock_response
    
    agent = LlamaAgent(temp_project_path)
    
    # Patch print to prevent output during test
    with patch('builtins.print'):
        result = agent.query_llama("Test prompt", [])
    
    assert 'First chunk of text' in result['response']
    
    # Verify the request was made with correct parameters
    mock_post.assert_called_once()
    args, kwargs = mock_post.call_args
    assert kwargs['json']['model'] == agent.config['model']['name']
    assert 'Test prompt' in kwargs['json']['prompt']

def test_create_new_file_success(temp_project_path):
    agent = LlamaAgent(temp_project_path)
    
    # Create a new markdown file
    result = agent.create_new_file('test_new_file')
    
    assert result is True
    
    # Check file was created in the correct location
    file_path = os.path.join(temp_project_path, 'docs', 'test_new_file.md')
    assert os.path.exists(file_path)
    
    # Check file content
    with open(file_path, 'r') as f:
        content = f.read()
        assert content == '# Test_New_File\n\n'

def test_create_new_file_existing(temp_project_path):
    agent = LlamaAgent(temp_project_path)
    
    # First creation should succeed
    result1 = agent.create_new_file('existing_file')
    assert result1 is True
    
    # Second creation with same name should fail
    with patch('builtins.print'):  # Suppress print output
        result2 = agent.create_new_file('existing_file')
    assert result2 is False
