"""Tests for the clients module."""

import pytest
from unittest.mock import MagicMock, patch

from bedrock_prompt_management_mcp_server.clients import get_bedrock_client, BedrockPromptManagementClient


class TestGetBedrockClient:
    """Tests for the get_bedrock_client function."""
    
    @patch("bedrock_prompt_management_mcp_server.clients.boto3")
    def test_get_bedrock_client_with_params(self, mock_boto3):
        """Test client creation with explicit parameters."""
        # Setup mock
        mock_session = MagicMock()
        mock_client = MagicMock()
        mock_boto3.Session.return_value = mock_session
        mock_session.client.return_value = mock_client
        
        # Call the function
        client = get_bedrock_client(region_name="us-west-2", profile_name="test-profile")
        
        # Check the result
        assert client == mock_client
        mock_boto3.Session.assert_called_once_with(profile_name="test-profile", region_name="us-west-2")
        mock_session.client.assert_called_once_with(service_name="bedrock-agent")
    
    @patch("bedrock_prompt_management_mcp_server.clients.boto3")
    @patch("bedrock_prompt_management_mcp_server.clients.os")
    def test_get_bedrock_client_with_env_vars(self, mock_os, mock_boto3):
        """Test client creation with environment variables."""
        # Setup mock
        mock_os.getenv.side_effect = lambda key: {"AWS_REGION": "us-east-1", "AWS_PROFILE": "default"}[key]
        mock_session = MagicMock()
        mock_client = MagicMock()
        mock_boto3.Session.return_value = mock_session
        mock_session.client.return_value = mock_client
        
        # Call the function
        client = get_bedrock_client()
        
        # Check the result
        assert client == mock_client
        mock_boto3.Session.assert_called_once_with(profile_name="default", region_name="us-east-1")
        mock_session.client.assert_called_once_with(service_name="bedrock-agent")
    
    @patch("bedrock_prompt_management_mcp_server.clients.boto3")
    def test_get_bedrock_client_error(self, mock_boto3):
        """Test error handling in client creation."""
        # Setup mock to raise an exception
        mock_session = MagicMock()
        mock_boto3.Session.return_value = mock_session
        mock_session.client.side_effect = Exception("Connection error")
        
        # Call the function and check that it raises the exception
        with pytest.raises(Exception) as excinfo:
            get_bedrock_client(region_name="us-west-2")
        
        # Check the error message
        assert "Connection error" in str(excinfo.value)


class TestBedrockPromptManagementClient:
    """Tests for the BedrockPromptManagementClient class."""
    
    @patch("bedrock_prompt_management_mcp_server.clients.get_bedrock_client")
    def test_init_with_custom_client(self, mock_get_client):
        """Test initialization with a custom client."""
        # Setup mock
        mock_bedrock_client = MagicMock()
        
        # Create client
        client = BedrockPromptManagementClient(bedrock_client=mock_bedrock_client)
        
        # Check that the client was set correctly
        assert client.bedrock_client == mock_bedrock_client
        # Ensure get_bedrock_client was not called
        mock_get_client.assert_not_called()
    
    @patch("bedrock_prompt_management_mcp_server.clients.get_bedrock_client")
    def test_init_with_params(self, mock_get_client):
        """Test initialization with parameters."""
        # Setup mock
        mock_bedrock_client = MagicMock()
        mock_get_client.return_value = mock_bedrock_client
        
        # Create client
        client = BedrockPromptManagementClient(region_name="us-west-2", profile_name="test-profile")
        
        # Check that the client was set correctly
        assert client.bedrock_client == mock_bedrock_client
        mock_get_client.assert_called_once_with(region_name="us-west-2", profile_name="test-profile") 