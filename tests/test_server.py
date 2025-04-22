"""Tests for the server module."""

import json
import pytest
from unittest.mock import MagicMock, patch, ANY

from bedrock_prompt_management_mcp_server.server import (
    list_prompts_tool,
    get_prompt_tool,
    create_prompt_tool,
    update_prompt_tool,
    create_prompt_version_tool,
    prompts_resource,
)


@pytest.fixture
def mock_prompt_client():
    """Mock the Bedrock Prompt Management client."""
    with patch("bedrock_prompt_management_mcp_server.server.prompt_client") as mock_client:
        yield mock_client


class TestListPromptsTool:
    """Tests for the list_prompts_tool function."""

    @pytest.mark.asyncio
    async def test_list_prompts_success(self, mock_prompt_client):
        """Test successful listing of prompts."""
        # Setup mock response
        mock_response = {
            "promptSummaries": [
                {
                    "id": "prompt-12345",
                    "name": "Test Prompt",
                    "description": "A test prompt",
                    "createdAt": "2023-01-01T00:00:00Z",
                    "updatedAt": "2023-01-02T00:00:00Z",
                    "version": "DRAFT"
                }
            ]
        }
        mock_prompt_client.bedrock_client.list_prompts.return_value = mock_response

        # Call the function
        result = await list_prompts_tool(max_results=10)
        
        # Check the result
        assert json.loads(result) == mock_response
        mock_prompt_client.bedrock_client.list_prompts.assert_called_once_with(maxResults=10)

    @pytest.mark.asyncio
    async def test_list_prompts_error(self, mock_prompt_client):
        """Test handling of errors when listing prompts."""
        # Setup mock to raise an exception
        mock_prompt_client.bedrock_client.list_prompts.side_effect = Exception("Test error")
        
        # Call the function
        result = await list_prompts_tool()
        
        # Check the result contains the error
        assert "error" in json.loads(result)
        assert "Test error" in json.loads(result)["error"]


class TestGetPromptTool:
    """Tests for the get_prompt_tool function."""
    
    @pytest.mark.asyncio
    async def test_get_prompt_success(self, mock_prompt_client):
        """Test successful retrieval of a prompt."""
        # Setup mock response
        mock_response = {
            "id": "prompt-12345",
            "name": "Test Prompt",
            "description": "A test prompt",
            "variants": [
                {
                    "name": "default",
                    "templateType": "TEXT",
                    "templateConfiguration": {
                        "text": {
                            "text": "This is a test prompt with {{variable}}"
                        }
                    },
                    "modelId": "anthropic.claude-3-sonnet-20240229-v1:0",
                    "inferenceConfiguration": {
                        "temperature": 0.7,
                        "topP": 0.9,
                        "maxTokens": 1000
                    }
                }
            ],
            "createdAt": "2023-01-01T00:00:00Z",
            "updatedAt": "2023-01-02T00:00:00Z",
            "version": "DRAFT"
        }
        mock_prompt_client.bedrock_client.get_prompt.return_value = mock_response
        
        # Call the function
        result = await get_prompt_tool(prompt_id="prompt-12345")
        
        # Check the result
        assert json.loads(result) == mock_response
        
        # Use mock.call to check arguments with ANY for optional parameters
        mock_prompt_client.bedrock_client.get_prompt.assert_called_once()
        call_args = mock_prompt_client.bedrock_client.get_prompt.call_args[1]
        assert call_args["promptIdentifier"] == "prompt-12345"
    
    @pytest.mark.asyncio
    async def test_get_prompt_with_version(self, mock_prompt_client):
        """Test retrieval of a prompt with a specific version."""
        # Setup mock response
        mock_response = {"id": "prompt-12345", "version": "1"}
        mock_prompt_client.bedrock_client.get_prompt.return_value = mock_response
        
        # Call the function
        result = await get_prompt_tool(prompt_id="prompt-12345", version_number="1")
        
        # Check the result
        assert json.loads(result) == mock_response
        
        # Check call arguments
        mock_prompt_client.bedrock_client.get_prompt.assert_called_once()
        call_args = mock_prompt_client.bedrock_client.get_prompt.call_args[1]
        assert call_args["promptIdentifier"] == "prompt-12345"
        assert call_args["promptVersion"] == "1"
    
    @pytest.mark.asyncio
    async def test_get_prompt_error(self, mock_prompt_client):
        """Test handling of errors when retrieving a prompt."""
        # Setup mock to raise an exception
        mock_prompt_client.bedrock_client.get_prompt.side_effect = Exception("Test error")
        
        # Call the function
        result = await get_prompt_tool(prompt_id="prompt-12345")
        
        # Check the result contains the error
        assert "error" in json.loads(result)
        assert "Test error" in json.loads(result)["error"]


class TestCreatePromptTool:
    """Tests for the create_prompt_tool function."""
    
    @pytest.mark.asyncio
    async def test_create_prompt_minimal(self, mock_prompt_client):
        """Test creating a prompt with minimal parameters."""
        # Setup mock response
        mock_response = {
            "id": "prompt-12345",
            "name": "Test Prompt",
            "description": "A test prompt"
        }
        mock_prompt_client.bedrock_client.create_prompt.return_value = mock_response
        
        # Call the function
        result = await create_prompt_tool(
            name="Test Prompt",
            description="A test prompt",
            model_id="anthropic.claude-3-sonnet-20240229-v1:0",
            prompt_text="This is a test prompt with {{variable}}"
        )
        
        # Check the result
        assert json.loads(result) == mock_response
        
        # Check the function was called with the right parameters
        mock_prompt_client.bedrock_client.create_prompt.assert_called_once()
        call_args = mock_prompt_client.bedrock_client.create_prompt.call_args[1]
        assert call_args["name"] == "Test Prompt"
        assert call_args["description"] == "A test prompt"
        assert len(call_args["variants"]) == 1
        assert call_args["variants"][0]["modelId"] == "anthropic.claude-3-sonnet-20240229-v1:0"
        assert call_args["variants"][0]["templateType"] == "TEXT"
        assert call_args["variants"][0]["templateConfiguration"]["text"]["text"] == "This is a test prompt with {{variable}}"
    
    @pytest.mark.asyncio
    async def test_create_prompt_full(self, mock_prompt_client):
        """Test creating a prompt with all parameters."""
        # Setup mock response
        mock_response = {"id": "prompt-12345"}
        mock_prompt_client.bedrock_client.create_prompt.return_value = mock_response
        
        # Call the function
        result = await create_prompt_tool(
            name="Test Prompt",
            description="A test prompt",
            model_id="anthropic.claude-3-sonnet-20240229-v1:0",
            prompt_text="This is a test prompt with {{variable}}",
            temperature=0.7,
            top_p=0.9,
            max_tokens=1000,
            client_token="token123",
            customer_encryption_key_arn="arn:aws:kms:us-west-2:123456789012:key/abcdef",
            default_variant="custom",
            tags={"env": "test"}
        )
        
        # Check the result
        assert json.loads(result) == mock_response
        
        # Check the function was called with the right parameters
        mock_prompt_client.bedrock_client.create_prompt.assert_called_once()
        call_args = mock_prompt_client.bedrock_client.create_prompt.call_args[1]
        assert call_args["name"] == "Test Prompt"
        assert call_args["defaultVariant"] == "custom"
        assert call_args["clientToken"] == "token123"
        assert call_args["customerEncryptionKeyArn"] == "arn:aws:kms:us-west-2:123456789012:key/abcdef"
        assert call_args["tags"] == {"env": "test"}
        
        # Check inference configuration
        variant = call_args["variants"][0]
        assert "inferenceConfiguration" in variant
        assert variant["inferenceConfiguration"]["temperature"] == 0.7
        assert variant["inferenceConfiguration"]["topP"] == 0.9
        assert variant["inferenceConfiguration"]["maxTokens"] == 1000
    
    @pytest.mark.asyncio
    async def test_create_prompt_error(self, mock_prompt_client):
        """Test handling of errors when creating a prompt."""
        # Setup mock to raise an exception
        mock_prompt_client.bedrock_client.create_prompt.side_effect = Exception("Test error")
        
        # Call the function
        result = await create_prompt_tool(
            name="Test Prompt",
            description="A test prompt",
            model_id="anthropic.claude-3-sonnet-20240229-v1:0",
            prompt_text="This is a test prompt with {{variable}}"
        )
        
        # Check the result contains the error
        assert "error" in json.loads(result)
        assert "Test error" in json.loads(result)["error"]


class TestUpdatePromptTool:
    """Tests for the update_prompt_tool function."""
    
    @pytest.mark.asyncio
    async def test_update_prompt_minimal(self, mock_prompt_client):
        """Test updating a prompt with minimal parameters."""
        # Setup mock response
        mock_response = {
            "id": "prompt-12345",
            "name": "Updated Prompt"
        }
        mock_prompt_client.bedrock_client.update_prompt.return_value = mock_response
        
        # Call the function
        result = await update_prompt_tool(
            prompt_id="prompt-12345",
            name="Updated Prompt"
        )
        
        # Check the result
        assert json.loads(result) == mock_response
        
        # Check only the required parameters without validating optional ones
        mock_prompt_client.bedrock_client.update_prompt.assert_called_once()
        call_args = mock_prompt_client.bedrock_client.update_prompt.call_args[1]
        assert call_args["promptIdentifier"] == "prompt-12345"
        assert call_args["name"] == "Updated Prompt"
    
    @pytest.mark.asyncio
    async def test_update_prompt_full(self, mock_prompt_client):
        """Test updating a prompt with all parameters."""
        # Setup mock response
        mock_response = {"id": "prompt-12345"}
        mock_prompt_client.bedrock_client.update_prompt.return_value = mock_response
        
        # Call the function
        result = await update_prompt_tool(
            prompt_id="prompt-12345",
            name="Updated Prompt",
            description="Updated description",
            default_variant="custom",
            customer_encryption_key_arn="arn:aws:kms:us-west-2:123456789012:key/abcdef",
            model_id="anthropic.claude-3-haiku-20240307-v1:0",
            prompt_text="Updated prompt text with {{variable}}",
            temperature=0.5,
            top_p=0.8,
            max_tokens=500
        )
        
        # Check the result
        assert json.loads(result) == mock_response
        
        # Get the call arguments
        mock_prompt_client.bedrock_client.update_prompt.assert_called_once()
        call_args = mock_prompt_client.bedrock_client.update_prompt.call_args[1]
        
        # Check basic parameters
        assert call_args["promptIdentifier"] == "prompt-12345"
        assert call_args["name"] == "Updated Prompt"
        assert call_args["description"] == "Updated description"
        assert call_args["defaultVariant"] == "custom"
        assert call_args["customerEncryptionKeyArn"] == "arn:aws:kms:us-west-2:123456789012:key/abcdef"
        
        # Check variant configuration
        variant = call_args["variants"][0]
        assert variant["modelId"] == "anthropic.claude-3-haiku-20240307-v1:0"
        assert variant["templateType"] == "TEXT"
        assert variant["templateConfiguration"]["text"]["text"] == "Updated prompt text with {{variable}}"
        
        # Check inference configuration
        assert variant["inferenceConfiguration"]["temperature"] == 0.5
        assert variant["inferenceConfiguration"]["topP"] == 0.8
        assert variant["inferenceConfiguration"]["maxTokens"] == 500
    
    @pytest.mark.asyncio
    async def test_update_prompt_error(self, mock_prompt_client):
        """Test handling of errors when updating a prompt."""
        # Setup mock to raise an exception
        mock_prompt_client.bedrock_client.update_prompt.side_effect = Exception("Test error")
        
        # Call the function
        result = await update_prompt_tool(prompt_id="prompt-12345", name="Updated Prompt")
        
        # Check the result contains the error
        assert "error" in json.loads(result)
        assert "Test error" in json.loads(result)["error"]


class TestCreatePromptVersionTool:
    """Tests for the create_prompt_version_tool function."""
    
    @pytest.mark.asyncio
    async def test_create_prompt_version_minimal(self, mock_prompt_client):
        """Test creating a prompt version with minimal parameters."""
        # Setup mock response
        mock_response = {
            "id": "prompt-12345",
            "version": "1"
        }
        mock_prompt_client.bedrock_client.create_prompt_version.return_value = mock_response
        
        # Call the function
        result = await create_prompt_version_tool(prompt_id="prompt-12345")
        
        # Check the result
        assert json.loads(result) == mock_response
        
        # Check only that the function was called with the promptIdentifier
        mock_prompt_client.bedrock_client.create_prompt_version.assert_called_once()
        call_args = mock_prompt_client.bedrock_client.create_prompt_version.call_args[1]
        assert call_args["promptIdentifier"] == "prompt-12345"
    
    @pytest.mark.asyncio
    async def test_create_prompt_version_full(self, mock_prompt_client):
        """Test creating a prompt version with all parameters."""
        # Setup mock response
        mock_response = {
            "id": "prompt-12345",
            "version": "1"
        }
        mock_prompt_client.bedrock_client.create_prompt_version.return_value = mock_response
        
        # Call the function
        result = await create_prompt_version_tool(
            prompt_id="prompt-12345",
            description="Version 1",
            client_token="token123",
            tags={"version": "1.0"}
        )
        
        # Check the result
        assert json.loads(result) == mock_response
        
        # Check the function was called with the right parameters
        mock_prompt_client.bedrock_client.create_prompt_version.assert_called_once()
        call_args = mock_prompt_client.bedrock_client.create_prompt_version.call_args[1]
        assert call_args["promptIdentifier"] == "prompt-12345"
        assert call_args["description"] == "Version 1"
        assert call_args["clientToken"] == "token123"
        assert call_args["tags"] == {"version": "1.0"}
    
    @pytest.mark.asyncio
    async def test_create_prompt_version_error(self, mock_prompt_client):
        """Test handling of errors when creating a prompt version."""
        # Setup mock to raise an exception
        mock_prompt_client.bedrock_client.create_prompt_version.side_effect = Exception("Test error")
        
        # Call the function
        result = await create_prompt_version_tool(prompt_id="prompt-12345")
        
        # Check the result contains the error
        assert "error" in json.loads(result)
        assert "Test error" in json.loads(result)["error"]


class TestPromptsResource:
    """Tests for the prompts_resource function."""
    
    @pytest.mark.asyncio
    async def test_prompts_resource_success(self, mock_prompt_client):
        """Test successful retrieval of prompts resource."""
        # Setup mock response
        mock_prompt_summaries = [
            {
                "id": "prompt-12345",
                "name": "Test Prompt",
                "description": "A test prompt",
                "createdAt": "2023-01-01T00:00:00Z",
                "updatedAt": "2023-01-02T00:00:00Z",
                "version": "DRAFT"
            }
        ]
        mock_response = {"promptSummaries": mock_prompt_summaries}
        mock_prompt_client.bedrock_client.list_prompts.return_value = mock_response
        
        # Call the function
        result = await prompts_resource()
        result_dict = json.loads(result)
        
        # Check the result structure
        assert "prompt-12345" in result_dict
        assert result_dict["prompt-12345"]["name"] == "Test Prompt"
        assert result_dict["prompt-12345"]["description"] == "A test prompt"
        assert result_dict["prompt-12345"]["created_at"] == "2023-01-01T00:00:00Z"
        assert result_dict["prompt-12345"]["updated_at"] == "2023-01-02T00:00:00Z"
        assert result_dict["prompt-12345"]["id"] == "prompt-12345"
        assert result_dict["prompt-12345"]["version"] == "DRAFT"
    
    @pytest.mark.asyncio
    async def test_prompts_resource_error(self, mock_prompt_client):
        """Test handling of errors in prompts resource."""
        # Setup mock to raise an exception
        mock_prompt_client.bedrock_client.list_prompts.side_effect = Exception("Test error")
        
        # Call the function
        result = await prompts_resource()
        
        # Check the result contains the error
        assert "error" in json.loads(result)
        assert "Test error" in json.loads(result)["error"] 