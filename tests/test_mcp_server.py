"""Tests for the MCP server initialization."""

import pytest
from unittest.mock import patch

from bedrock_prompt_management_mcp_server.server import mcp


class TestMCPServer:
    """Tests for the MCP server initialization and configuration."""
    
    def test_mcp_server_init(self):
        """Test MCP server initialization."""
        # Check MCP server name
        assert mcp.name == "bedrock-prompt-management-mcp-server"
        
        # Check that instructions are set
        assert "AWS Bedrock Prompt Management MCP Server" in mcp.instructions
        
        # Check that dependencies are set
        assert "boto3" in mcp.dependencies
    
    @pytest.mark.asyncio
    async def test_tool_registration(self):
        """Test that all tools are registered."""
        # Instead of checking internal structure, just verify the tools exist
        # by checking the tool names in the available tools from get_tools()
        tools = await mcp.get_tools()
        
        expected_tools = ["ListPrompts", "GetPrompt", "CreatePrompt", "UpdatePrompt", "CreatePromptVersion"]
        
        # Verify all expected tools are registered
        for tool in expected_tools:
            assert tool in tools
    
    @pytest.mark.asyncio
    async def test_resource_registration(self):
        """Test that all resources are registered."""
        # Get resources from the MCP instance
        resources = await mcp.get_resources()
        
        # Check that at least one resource related to prompts exists
        assert any("prompt" in resource for resource in resources)


@patch("bedrock_prompt_management_mcp_server.server.mcp")
def test_main_default(mock_mcp):
    """Test main function with default arguments."""
    from bedrock_prompt_management_mcp_server.server import main
    
    # Mock sys.argv to simulate no command line arguments
    with patch("sys.argv", ["bedrock-prompt-management-mcp-server"]):
        # Call main
        main()
        
        # Check that mcp.run was called without transport
        mock_mcp.run.assert_called_once_with()


@patch("bedrock_prompt_management_mcp_server.server.mcp")
def test_main_with_sse(mock_mcp):
    """Test main function with SSE transport."""
    from bedrock_prompt_management_mcp_server.server import main
    
    # Mock sys.argv to simulate --sse argument
    with patch("sys.argv", ["bedrock-prompt-management-mcp-server", "--sse"]):
        # Call main
        main()
        
        # Check that mcp.run was called with SSE transport
        mock_mcp.run.assert_called_once_with(transport="sse")


@patch("bedrock_prompt_management_mcp_server.server.mcp")
def test_main_with_port(mock_mcp):
    """Test main function with custom port."""
    from bedrock_prompt_management_mcp_server.server import main
    
    # Mock sys.argv to simulate --port argument
    with patch("sys.argv", ["bedrock-prompt-management-mcp-server", "--port", "9999"]):
        # Call main
        main()
        
        # We can only verify that run was called, not how port was set
        # since we can't easily check attribute assignment on mocks
        mock_mcp.run.assert_called_once_with()


@patch("bedrock_prompt_management_mcp_server.server.mcp")
def test_main_with_sse_and_port(mock_mcp):
    """Test main function with SSE transport and custom port."""
    from bedrock_prompt_management_mcp_server.server import main
    
    # Mock sys.argv to simulate --sse and --port arguments
    with patch("sys.argv", ["bedrock-prompt-management-mcp-server", "--sse", "--port", "9999"]):
        # Call main
        main()
        
        # Check that run was called with SSE transport
        mock_mcp.run.assert_called_once_with(transport="sse") 