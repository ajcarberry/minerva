import os
import pytest
from unittest.mock import patch, MagicMock, call
from local_agent.__main__ import main
import local_agent.agent  # Import the entire module to use module-level patching

def test_main_entry_point(tmp_path):
    # Directly patch the LlamaAgent import in __main__
    with patch('local_agent.__main__.LlamaAgent') as MockAgent, \
         patch.object(os, 'getcwd', return_value=str(tmp_path)):
        
        # Create a mock for interactive_session and cleanup
        mock_agent_instance = MagicMock()
        MockAgent.return_value = mock_agent_instance
        mock_agent_instance.interactive_session = MagicMock()
        mock_agent_instance.cleanup = MagicMock()
        
        # Call main
        main()
        
        # Verify LlamaAgent was called with the correct path
        MockAgent.assert_called_once_with(str(tmp_path))
        
        # Verify interactive_session and cleanup were called
        mock_agent_instance.interactive_session.assert_called_once()
        mock_agent_instance.cleanup.assert_called_once()