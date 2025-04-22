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
    
    def test_list_prompts(self):
        """Test list_prompts method."""
        # Setup mock
        mock_bedrock_client = MagicMock()
        mock_paginator = MagicMock()
        mock_page1 = {"promptSummaries": [{"id": "prompt-1"}]}
        mock_page2 = {"promptSummaries": [{"id": "prompt-2"}]}
        mock_bedrock_client.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.return_value = [mock_page1, mock_page2]
        
        # Create client and call method
        client = BedrockPromptManagementClient(bedrock_client=mock_bedrock_client)
        result = client.list_prompts(max_results=50)
        
        # Check the result
        assert len(result) == 2
        assert result[0]["id"] == "prompt-1"
        assert result[1]["id"] == "prompt-2"
        mock_bedrock_client.get_paginator.assert_called_once_with("list_prompts")
        mock_paginator.paginate.assert_called_once_with(maxResults=50)
    
    def test_get_prompt(self):
        """Test get_prompt method."""
        # Setup mock
        mock_bedrock_client = MagicMock()
        mock_response = {"id": "prompt-12345", "name": "Test Prompt"}
        mock_bedrock_client.get_prompt.return_value = mock_response
        
        # Create client and call method
        client = BedrockPromptManagementClient(bedrock_client=mock_bedrock_client)
        result = client.get_prompt(prompt_id="prompt-12345")
        
        # Check the result
        assert result == mock_response
        mock_bedrock_client.get_prompt.assert_called_once_with(promptIdentifier="prompt-12345")
    
    def test_get_prompt_with_version(self):
        """Test get_prompt method with version."""
        # Setup mock
        mock_bedrock_client = MagicMock()
        mock_response = {"id": "prompt-12345", "name": "Test Prompt", "version": "1"}
        mock_bedrock_client.get_prompt.return_value = mock_response
        
        # Create client and call method
        client = BedrockPromptManagementClient(bedrock_client=mock_bedrock_client)
        result = client.get_prompt(prompt_id="prompt-12345", version_number=1)
        
        # Check the result
        assert result == mock_response
        mock_bedrock_client.get_prompt.assert_called_once_with(
            promptIdentifier="prompt-12345", promptVersion=1
        )
    
    def test_create_prompt(self):
        """Test create_prompt method."""
        # Setup mock
        mock_bedrock_client = MagicMock()
        mock_response = {"id": "prompt-12345", "name": "Test Prompt"}
        mock_bedrock_client.create_prompt.return_value = mock_response
        
        # Create client and call method
        client = BedrockPromptManagementClient(bedrock_client=mock_bedrock_client)
        result = client.create_prompt(
            name="Test Prompt",
            description="A test prompt",
            model_id="anthropic.claude-3-sonnet-20240229-v1:0",
            template="This is a test prompt with {{variable}}",
        )
        
        # Check the result
        assert result == mock_response
        mock_bedrock_client.create_prompt.assert_called_once()
        # Check the arguments
        call_args = mock_bedrock_client.create_prompt.call_args[1]
        assert call_args["name"] == "Test Prompt"
        assert call_args["description"] == "A test prompt"
        assert call_args["modelId"] == "anthropic.claude-3-sonnet-20240229-v1:0"
        assert call_args["promptConfiguration"]["promptType"] == "TEXT"
        assert call_args["promptConfiguration"]["textPromptConfig"]["text"] == "This is a test prompt with {{variable}}"
    
    def test_update_prompt(self):
        """Test update_prompt method."""
        # Setup mock
        mock_bedrock_client = MagicMock()
        mock_response = {"id": "prompt-12345", "name": "Updated Prompt"}
        mock_bedrock_client.update_prompt.return_value = mock_response
        
        # Create client and call method
        client = BedrockPromptManagementClient(bedrock_client=mock_bedrock_client)
        result = client.update_prompt(
            prompt_id="prompt-12345",
            description="Updated description",
            model_id="anthropic.claude-3-haiku-20240307-v1:0",
            template="Updated prompt text",
        )
        
        # Check the result
        assert result == mock_response
        mock_bedrock_client.update_prompt.assert_called_once()
        # Check the arguments
        call_args = mock_bedrock_client.update_prompt.call_args[1]
        assert call_args["promptIdentifier"] == "prompt-12345"
        assert call_args["description"] == "Updated description"
        assert call_args["modelId"] == "anthropic.claude-3-haiku-20240307-v1:0"
        assert call_args["promptConfiguration"]["promptType"] == "TEXT"
        assert call_args["promptConfiguration"]["textPromptConfig"]["text"] == "Updated prompt text"
    
    def test_create_prompt_version(self):
        """Test create_prompt_version method."""
        # Setup mock
        mock_bedrock_client = MagicMock()
        mock_response = {"id": "prompt-12345", "version": "1"}
        mock_bedrock_client.create_prompt_version.return_value = mock_response
        
        # Create client and call method
        client = BedrockPromptManagementClient(bedrock_client=mock_bedrock_client)
        result = client.create_prompt_version(
            prompt_id="prompt-12345",
            description="Version 1",
        )
        
        # Check the result
        assert result == mock_response
        mock_bedrock_client.create_prompt_version.assert_called_once_with(
            promptIdentifier="prompt-12345",
            description="Version 1"
        ) 