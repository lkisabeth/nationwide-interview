"""AWS Bedrock Prompt Management MCP Server."""

import argparse
import json
import os
import sys
from typing import Dict, List, Optional

from fastmcp import FastMCP
from loguru import logger
from pydantic import Field

from bedrock_prompt_management_mcp_server.clients import BedrockPromptManagementClient


# Configure logging
logger.remove(0)
logger.add(sys.stderr, level='INFO')


# Initialize Bedrock client
try:
    prompt_client = BedrockPromptManagementClient(
        region_name=os.getenv("AWS_REGION"),
        profile_name=os.getenv("AWS_PROFILE"),
    )
except Exception as e:
    logger.error(f"Error initializing Bedrock client: {e}")
    raise e


# Create MCP server
mcp = FastMCP(
    "bedrock-prompt-management-mcp-server",
    instructions="""
    The AWS Bedrock Prompt Management MCP Server provides access to Amazon Bedrock's Prompt Management capabilities, allowing you to create, discover, and manage prompts.

    ## Usage Workflow:
    1. Start by using the `ListPrompts` tool to discover available prompts
    2. Use `GetPrompt` to retrieve detailed information about a specific prompt
    3. Use `CreatePrompt` to create new prompts with customized templates
    4. Use `UpdatePrompt` to modify existing prompts (requires prompt ID)
    5. Use `CreatePromptVersion` to save versions of your prompts for deployment

    ## Important Notes:
    - Prompt variables must be specified in template text using {{variable_name}} syntax
    - When updating prompts, the name parameter is required by the API even if unchanged
    - CreatePromptVersion creates a permanent snapshot of a prompt that can be deployed
    """,
    dependencies=["boto3"],
)

@mcp.resource(uri="resource://prompts", name="BedrockPrompts", mime_type="application/json")
async def prompts_resource() -> str:
    """List all available prompts in the AWS Bedrock account.

    This resource returns a mapping of prompt IDs to their details, including:
    - name: The human-readable name of the prompt
    - description: A description of the prompt
    - created_at: When the prompt was created
    - updated_at: When the prompt was last updated
    - id: The unique identifier of the prompt
    - version: The version of the prompt

    ## Example response structure:
    ```json
    {
        "prompt-12345": {
            "name": "Customer Support Response",
            "description": "Generates customer support responses",
            "created_at": "2024-02-29T10:15:30Z",
            "updated_at": "2024-03-14T08:45:12Z",
            "id": "prompt-12345",
            "version": "DRAFT"
        }
    }
    ```
    """
    try:
        # Use list_prompts API
        response = prompt_client.bedrock_client.list_prompts()
        
        # Transform the response to the desired format
        result = {}
        for prompt in response.get("promptSummaries", []):
            prompt_id = prompt.get("id", "")
            result[prompt_id] = {
                "name": prompt.get("name", ""),
                "description": prompt.get("description", ""),
                "created_at": prompt.get("createdAt", ""),
                "updated_at": prompt.get("updatedAt", ""),
                "id": prompt_id,
                "version": prompt.get("version", "")
            }
            
        return json.dumps(result)
    except Exception as e:
        logger.error(f"Error retrieving prompts: {e}")
        return json.dumps({"error": str(e)})


@mcp.tool(name="ListPrompts")
async def list_prompts_tool(
    max_results: int = Field(
        100,
        description="Maximum number of prompts to retrieve",
    ),
) -> str:
    """List all prompts available in your AWS Bedrock account.

    This tool retrieves a list of all prompts stored in your AWS Bedrock Prompt Management,
    showing their IDs, names, descriptions and other basic information.

    ## Response format:
    The response is a JSON object containing prompt summaries, each with:
    - id: The unique identifier for the prompt
    - name: The name of the prompt
    - description: A description of what the prompt does
    - createdAt: When the prompt was created
    - updatedAt: When the prompt was last updated
    - version: The version of the prompt
    """
    try:
        response = prompt_client.bedrock_client.list_prompts(maxResults=max_results)
        return json.dumps(response, default=str)
    except Exception as e:
        logger.error(f"Error listing prompts: {e}")
        return json.dumps({"error": str(e)})


@mcp.tool(name="GetPrompt")
async def get_prompt_tool(
    prompt_id: str = Field(
        ...,
        description="The ID of the prompt to retrieve",
    ),
    version_number: Optional[str] = Field(
        None,
        description="Optional version number to retrieve (defaults to DRAFT)",
    ),
) -> str:
    """Get detailed information about a specific prompt.

    This tool retrieves comprehensive details about a prompt, including its
    template, variants, inference configuration, and metadata.

    ## Required parameters:
    - prompt_id: The unique identifier of the prompt to retrieve
    
    ## Optional parameters:
    - version_number: A specific version to retrieve (if not provided, the DRAFT version is returned)

    ## Response format:
    The response is a JSON object containing the complete prompt details including:
    - id: The unique identifier
    - name: The prompt name
    - description: A description of what the prompt does
    - variants: The prompt variants with template configuration
    - createdAt: Creation timestamp
    - updatedAt: Last update timestamp
    - version: The version number
    """
    try:
        kwargs = {"promptIdentifier": prompt_id}
        if version_number:
            kwargs["promptVersion"] = version_number
                
        response = prompt_client.bedrock_client.get_prompt(**kwargs)
        return json.dumps(response, default=str)
    except Exception as e:
        logger.error(f"Error getting prompt {prompt_id}: {e}")
        return json.dumps({"error": str(e)})


@mcp.tool(name="CreatePrompt")
async def create_prompt_tool(
    name: str = Field(
        ...,
        description="Name of the prompt",
    ),
    description: str = Field(
        ...,
        description="Description of the prompt",
    ),
    model_id: str = Field(
        ...,
        description="Model ID to use for the prompt (e.g., 'anthropic.claude-3-sonnet-20240229-v1:0')",
    ),
    prompt_text: str = Field(
        ...,
        description="Template text for the prompt",
    ),
    temperature: Optional[float] = Field(
        None,
        description="Temperature for model inference (0.0 to 1.0)",
    ),
    top_p: Optional[float] = Field(
        None,
        description="Top-p for model inference (0.0 to 1.0)",
    ),
    max_tokens: Optional[int] = Field(
        None,
        description="Maximum number of tokens to generate",
    ),
    client_token: Optional[str] = Field(
        None,
        description="A unique, case-sensitive identifier to ensure idempotency",
    ),
    customer_encryption_key_arn: Optional[str] = Field(
        None,
        description="The ARN of the KMS key to encrypt the prompt",
    ),
    default_variant: Optional[str] = Field(
        "default",
        description="The name of the default variant for the prompt",
    ),
    tags: Optional[Dict[str, str]] = Field(
        None,
        description="Any tags to attach to the prompt",
    ),
) -> str:
    """Create a new prompt in Amazon Bedrock Prompt Management.

    This tool creates a new prompt with the specified configuration, including
    text template, model settings, and inference parameters.

    ## Required parameters:
    - name: A name for the prompt
    - description: A description of what the prompt does
    - model_id: The foundation model ID to use
    - prompt_text: The template text for the prompt

    ## Optional parameters:
    - temperature: Model temperature (0.0 to 1.0)
    - top_p: Model top-p (0.0 to 1.0)
    - max_tokens: Maximum tokens to generate
    - client_token: A unique identifier to ensure idempotency
    - customer_encryption_key_arn: ARN of KMS key for encryption
    - default_variant: Name of the default variant (defaults to "default")
    - tags: Key-value pairs for resource tagging

    ## Response format:
    The response is a JSON object containing the details of the created prompt,
    including its generated ID and default version.
    """
    try:
        # Create prompt variant
        variant = {
            "name": default_variant,
            "templateType": "TEXT",
            "templateConfiguration": {
                "text": {
                    "text": prompt_text
                }
            },
            "modelId": model_id
        }
        
        # Add inference configuration if parameters provided
        if any(param is not None for param in [temperature, top_p, max_tokens]):
            inference_config = {}
            if temperature is not None:
                inference_config["temperature"] = temperature
            if top_p is not None:
                inference_config["topP"] = top_p
            if max_tokens is not None:
                inference_config["maxTokens"] = max_tokens
            
            if inference_config:
                variant["inferenceConfiguration"] = inference_config
        
        # Create request parameters
        kwargs = {
            "name": name,
            "description": description,
            "variants": [variant]
        }
        
        # Add optional parameters if provided
        if default_variant:
            kwargs["defaultVariant"] = default_variant
        if client_token:
            kwargs["clientToken"] = client_token
        if customer_encryption_key_arn:
            kwargs["customerEncryptionKeyArn"] = customer_encryption_key_arn
        if tags:
            kwargs["tags"] = tags
        
        # Create the prompt
        response = prompt_client.bedrock_client.create_prompt(**kwargs)
        
        return json.dumps(response, default=str)
    except Exception as e:
        logger.error(f"Error creating prompt: {e}")
        return json.dumps({"error": str(e)})


@mcp.tool(name="UpdatePrompt")
async def update_prompt_tool(
    prompt_id: str = Field(
        ...,
        description="ID of the prompt to update",
    ),
    name: str = Field(
        None,
        description="Name of the prompt (required by AWS API even if unchanged)",
    ),
    description: Optional[str] = Field(
        None,
        description="New description for the prompt",
    ),
    default_variant: Optional[str] = Field(
        None,
        description="New name of the default variant",
    ),
    customer_encryption_key_arn: Optional[str] = Field(
        None,
        description="The ARN of the KMS key to encrypt the prompt",
    ),
    model_id: Optional[str] = Field(
        None,
        description="New model ID to use for the prompt",
    ),
    prompt_text: Optional[str] = Field(
        None,
        description="New template text for the prompt with variables in {{variable_name}} format",
    ),
    temperature: Optional[float] = Field(
        None,
        description="New temperature for model inference",
    ),
    top_p: Optional[float] = Field(
        None,
        description="New top-p for model inference",
    ),
    max_tokens: Optional[int] = Field(
        None,
        description="New maximum number of tokens to generate",
    ),
) -> str:
    """Update an existing prompt in Amazon Bedrock Prompt Management.

    This tool updates a prompt with new configuration values. The AWS API requires
    that the name parameter be provided even if you're not changing it.

    ## Required parameters:
    - prompt_id: The ID of the prompt to update
    - name: Current name of the prompt (required by AWS API even if unchanged)

    ## Optional parameters (only specify what you want to change):
    - description: A new description
    - default_variant: New name of the default variant
    - customer_encryption_key_arn: New ARN of KMS key for encryption
    - model_id: A new foundation model ID
    - prompt_text: A new template text with variables in {{variable_name}} format
    - temperature: New model temperature
    - top_p: New model top-p
    - max_tokens: New maximum tokens to generate

    ## Response format:
    The response is a JSON object containing the updated prompt details.
    Remember that this creates a draft of the prompt - to make the changes
    permanent, you need to create a new version with CreatePromptVersion.

    ## Notes:
    When updating a prompt, you need to retrieve the current prompt details first
    using GetPrompt to ensure you pass the required name parameter correctly.
    """
    try:
        # Build the request payload
        request_body = {}

        # Add base parameters if provided
        if name is not None:
            request_body["name"] = name
        if description is not None:
            request_body["description"] = description
        if default_variant is not None:
            request_body["defaultVariant"] = default_variant
        if customer_encryption_key_arn is not None:
            request_body["customerEncryptionKeyArn"] = customer_encryption_key_arn
        
        # Create variant if any model-related parameters are provided
        if any(param is not None for param in [model_id, prompt_text, temperature, top_p, max_tokens]):
            variant = {"name": default_variant or "default"}
            
            # Only add fields that are provided
            if model_id is not None:
                variant["modelId"] = model_id
            
            # Handle template configuration if prompt_text is provided
            if prompt_text is not None:
                variant["templateType"] = "TEXT"
                variant["templateConfiguration"] = {
                    "text": {
                        "text": prompt_text
                    }
                }
            
            # Add inference configuration if parameters provided
            if any(param is not None for param in [temperature, top_p, max_tokens]):
                inference_config = {}
                if temperature is not None:
                    inference_config["temperature"] = temperature
                if top_p is not None:
                    inference_config["topP"] = top_p
                if max_tokens is not None:
                    inference_config["maxTokens"] = max_tokens
                
                if inference_config:
                    variant["inferenceConfiguration"] = inference_config
            
            # Add variant to request data
            request_body["variants"] = [variant]
        
        # Update the prompt - pass the promptIdentifier separately as a path parameter
        response = prompt_client.bedrock_client.update_prompt(
            promptIdentifier=prompt_id,
            **request_body
        )
        
        return json.dumps(response, default=str)
    except Exception as e:
        logger.error(f"Error updating prompt {prompt_id}: {e}")
        return json.dumps({"error": str(e)})


@mcp.tool(name="CreatePromptVersion")
async def create_prompt_version_tool(
    prompt_id: str = Field(
        ..., 
        description="The ID of the prompt to create a version for",
    ),
    description: Optional[str] = Field(
        None,
        description="Description of the new version",
    ),
    client_token: Optional[str] = Field(
        None,
        description="A unique, case-sensitive identifier to ensure idempotency",
    ),
    tags: Optional[Dict[str, str]] = Field(
        None,
        description="Any tags to attach to the prompt version",
    ),
) -> str:
    """Create a new version of an existing prompt.

    This tool creates a permanent snapshot of a prompt for deployment to production.
    It captures the current state of a prompt as a numbered version (starting from 1)
    that can be deployed and referenced with stability, unlike the DRAFT version
    which changes with each update.

    ## Required parameters:
    - prompt_id: The ID of the prompt to version

    ## Optional parameters:
    - description: A description of the new version explaining the changes
    - client_token: A unique identifier to ensure idempotency
    - tags: Key-value pairs for resource tagging

    ## Response format:
    The response is a JSON object containing details about the new version,
    including the version number and creation timestamp.

    ## When to use:
    Use this tool after finalizing changes to a prompt with UpdatePrompt to create
    a permanent version that can be safely referenced in production applications.
    """
    try:
        # Build the request body
        request_body = {}
        if description:
            request_body["description"] = description
        if client_token:
            request_body["clientToken"] = client_token
        if tags:
            request_body["tags"] = tags

        # Call the create_prompt_version API
        response = prompt_client.bedrock_client.create_prompt_version(
            promptIdentifier=prompt_id,
            **request_body
        )
        
        return json.dumps(response, default=str)
    except Exception as e:
        logger.error(f"Error creating version for prompt {prompt_id}: {e}")
        return json.dumps({"error": str(e)})


def main():
    """Run the MCP server with CLI argument support."""
    parser = argparse.ArgumentParser(description="A Model Context Protocol (MCP) server for AWS Bedrock Prompt Management")
    parser.add_argument("--sse", action="store_true", help="Use SSE transport")
    parser.add_argument("--port", type=int, default=8888, help="Port to run the server on")

    args = parser.parse_args()

    # Run server with appropriate transport
    if args.sse:
        mcp.settings.port = args.port
        mcp.run(transport="sse")
    else:
        mcp.run()


if __name__ == "__main__":
    main() 