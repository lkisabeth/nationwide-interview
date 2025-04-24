"""AWS Bedrock Prompt Management MCP Server."""

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Any

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

    ## Usage:
    1. Use the `ListPrompts` tool to discover available prompts
    2. Use `GetPrompt` to retrieve detailed information about a specific prompt
    3. Use `CreatePrompt` to create new prompts with customized templates
    4. Use `UpdatePrompt` to modify existing prompts (requires prompt ID)
    5. Use `CreatePromptVersion` to save versions of your prompts for deployment
    6. Use `DeletePrompt` to delete prompts or specific versions

    ## Important Notes:
    - Prompt variables must be specified in template text using {{variable_name}} syntax
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
        response = prompt_client.bedrock_client.list_prompts()

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
        paginator = prompt_client.bedrock_client.get_paginator("list_prompts")
        pages = paginator.paginate(maxResults=max_results)
        
        prompts = []
        for page in pages:
            prompts.extend(page.get("promptSummaries", []))
            
        return json.dumps({"promptSummaries": prompts}, default=str)
    except Exception as e:
        logger.error(f"Error listing prompts: {e}")
        return json.dumps({"error": str(e)})


@mcp.tool(name="GetPrompt")
async def get_prompt_tool(
    prompt_id: str = Field(
        ...,
        description="ID of the prompt to retrieve",
    ),
    prompt_version: Optional[str] = Field(
        None,
        description="Optional version of the prompt to retrieve. Omit for the DRAFT version.",
    ),
) -> str:
    """Retrieve detailed information about a specific prompt.
    
    This tool retrieves the details of a prompt by its ID, including its template configuration,
    variants, and metadata. You can optionally specify a version to retrieve.
    
    ## Required parameters:
    - prompt_id: The unique identifier of the prompt to retrieve
    
    ## Optional parameters:
    - prompt_version: The version of the prompt to retrieve (omit for the working DRAFT version)
    
    ## Response format:
    The response is a JSON object containing detailed information about the prompt,
    including its variants, template configuration, and metadata.
    """
    try:
        # Build request parameters with a dictionary comprehension that filters out None values
        params = {k: v for k, v in {
            "promptIdentifier": prompt_id,
            "promptVersion": prompt_version
        }.items() if v is not None}
        
        # Call the get_prompt API
        response = prompt_client.bedrock_client.get_prompt(**params)
        
        return json.dumps(response, default=str)
    except Exception as e:
        logger.error(f"Error retrieving prompt {prompt_id}: {e}")
        return json.dumps({"error": str(e)})


@mcp.tool(name="CreatePrompt")
async def create_prompt_tool(
    name: str = Field(
        ...,
        description="Name of the prompt to create",
    ),
    template_type: str = Field(
        ...,
        description="Type of prompt template: TEXT or CHAT",
    ),
    template_text: str = Field(
        ...,
        description="Content for the template - either TEXT content or CHAT system message",
    ),
    input_variable_names: List[str] = Field(
        ...,
        description="List of input variable names (without the {{ }} brackets)",
    ),
    default_variant: str = Field(
        "default",
        description="Name of the default variant",
    ),
    prompt_description: Optional[str] = Field(
        None,
        description="Description for the prompt",
    ),
    model_id: Optional[str] = Field(
        None,
        description="Model ID to use with the prompt",
    ),
    temperature: Optional[float] = Field(
        None,
        description="Controls randomness of the response. Lower is more deterministic (0-1).",
    ),
    max_tokens: Optional[int] = Field(
        None,
        description="Maximum number of tokens to generate",
    ),
    chat_messages: Optional[List[Dict[str, str]]] = Field(
        None,
        description="For CHAT templates only: List of message objects, each with 'role' (user/assistant) and 'content'",
    ),
    client_token: Optional[str] = Field(
        None,
        description="A unique, case-sensitive identifier to ensure idempotency",
    ),
    customer_encryption_key_arn: Optional[str] = Field(
        None,
        description="The ARN of the KMS key to encrypt the prompt",
    ),
    tags: Optional[Dict[str, str]] = Field(
        None,
        description="Key-value pairs for resource tagging",
    ),
) -> str:
    """Create a new prompt in Amazon Bedrock Prompt Management.
    
    This tool creates a new prompt in your AWS Bedrock Prompt Management library.
    You can create either a TEXT or CHAT prompt with customizable templates.
    
    ## Required parameters:
    - name: The name of the prompt
    - template_type: The type of prompt template (TEXT or CHAT)
    - template_text: For TEXT templates, the template content; for CHAT templates, the system message
    - input_variable_names: List of variable names to use in the template (without {{ }})
    
    ## Optional configuration:
    - default_variant: Name of the default variant (defaults to "default")
    - prompt_description: Description for the prompt
    - model_id: Foundation model ID to use 
    - temperature: Controls randomness (0-1)
    - max_tokens: Maximum tokens to generate
    - chat_messages: For CHAT templates only - list of messages with 'role' and 'content'
    - client_token: Unique identifier for idempotency
    - customer_encryption_key_arn: ARN of KMS key for encryption
    - tags: Key-value pairs for resource tagging
    
    ## Response format:
    The response is a JSON object containing details about the created prompt,
    including its ID, ARN, and variant information.
    """
    try:
        # Build the base request body with required and optional parameters
        request_body = {k: v for k, v in {
            "name": name,
            "defaultVariant": default_variant,
            "description": prompt_description,
            "clientToken": client_token,
            "customerEncryptionKeyArn": customer_encryption_key_arn,
            "tags": tags
        }.items() if v is not None}
        
        # Format input variables
        formatted_input_vars = [{"name": var_name} for var_name in input_variable_names]
        
        # Create inference configuration if temperature or max_tokens are provided
        inference_config = {k: v for k, v in {
            "temperature": temperature,
            "maxTokens": max_tokens
        }.items() if v is not None}
        
        # Create the variant object
        variant = {
            "name": default_variant,
            "templateType": template_type,
        }
        
        # Add model ID if provided
        if model_id is not None:
            variant["modelId"] = model_id
            
        # Add inference configuration if provided
        if inference_config:
            variant["inferenceConfiguration"] = {"text": inference_config}
            
        # Create template configuration based on type
        if template_type == "TEXT":
            variant["templateConfiguration"] = {
                "text": {
                    "text": template_text,
                    "inputVariables": formatted_input_vars
                }
            }
                
        elif template_type == "CHAT":
            chat_config = {
                "system": [{"text": template_text}],
                "inputVariables": formatted_input_vars
            }
            
            # Add chat messages if provided
            if chat_messages and len(chat_messages) > 0:
                formatted_messages = []
                for msg in chat_messages:
                    formatted_messages.append({
                        "role": msg["role"],
                        "content": [{"text": msg["content"]}]
                    })
                chat_config["messages"] = formatted_messages
            else:
                # Default messages if none provided
                chat_config["messages"] = [
                    {
                        "role": "user", 
                        "content": [{"text": "Hello"}]
                    },
                    {
                        "role": "assistant",
                        "content": [{"text": "How can I help you today?"}]
                    }
                ]
                
            variant["templateConfiguration"] = {"chat": chat_config}
        
        # Add the variant to the request body
        request_body["variants"] = [variant]

        response = prompt_client.bedrock_client.create_prompt(**request_body)
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
        ...,
        description="Name of the prompt (required even if unchanged)",
    ),
    default_variant: Optional[str] = Field(
        None,
        description="Name of the default variant",
    ),
    prompt_description: Optional[str] = Field(
        None,
        description="Description for the prompt",
    ),
    customer_encryption_key_arn: Optional[str] = Field(
        None,
        description="The ARN of the KMS key to encrypt the prompt",
    ),
    variants: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="List of prompt variants to update",
    ),
) -> str:
    """Update an existing prompt in Amazon Bedrock Prompt Management.
    
    This tool updates an existing prompt in your AWS Bedrock Prompt Management library.
    You must include both fields that you want to keep and fields that you want to replace.
    
    ## Required parameters:
    - prompt_id: The ID of the prompt to update
    - name: The name of the prompt (required even if unchanged)
    
    ## Optional parameters:
    - default_variant: Name of the default variant
    - prompt_description: Description for the prompt
    - customer_encryption_key_arn: ARN of KMS key for encryption
    - variants: List of prompt variants to update (see CreatePrompt for structure)
    
    ## Response format:
    The response is a JSON object containing details about the updated prompt,
    including its ID, ARN, and variant information.
    
    ## Important notes:
    - The name parameter is required by the API even if you don't want to change it
    - When updating variants, you must provide all configuration you want to keep
    - Only the DRAFT version can be updated (use CreatePromptVersion for versioning)
    """
    try:
        # Build the request body with required parameters and filter out None values from optional ones
        request_body = {k: v for k, v in {
            "promptIdentifier": prompt_id,
            "name": name,
            "defaultVariant": default_variant,
            "description": prompt_description,
            "customerEncryptionKeyArn": customer_encryption_key_arn,
            "variants": variants
        }.items() if v is not None}
        
        # Update the prompt
        response = prompt_client.bedrock_client.update_prompt(**request_body)
        
        return json.dumps(response, default=str)
    except Exception as e:
        logger.error(f"Error updating prompt {prompt_id}: {e}")
        return json.dumps({"error": str(e)})


@mcp.tool(name="CreatePromptVersion")
async def create_prompt_version_tool(
    prompt_id: str = Field(
        ...,
        description="ID of the prompt to create a version of",
    ),
    description: Optional[str] = Field(
        None,
        description="Description for the prompt version",
    ),
    client_token: Optional[str] = Field(
        None,
        description="A unique, case-sensitive identifier to ensure idempotency",
    ),
    tags: Optional[Dict[str, str]] = Field(
        None,
        description="Key-value pairs for resource tagging",
    ),
) -> str:
    """Create a new version of an existing prompt.
    
    This tool creates a static snapshot of a prompt that can be deployed to production.
    It creates a permanent, versioned copy of the prompt's DRAFT version.
    
    ## Required parameters:
    - prompt_id: The ID of the prompt to create a version of
    
    ## Optional parameters:
    - description: Description for the prompt version
    - client_token: Unique identifier for idempotency
    - tags: Key-value pairs for resource tagging
    
    ## Response format:
    The response is a JSON object containing details about the created prompt version,
    including its ID, ARN, and variant information.
    
    ## Important notes:
    - Versions are numbered incrementally starting from 1
    - Versions are immutable snapshots that can be deployed to production
    - You cannot edit a version after it's created (you can only create new versions)
    """
    try:
        # Build request body with required parameters and filter out None values from optional ones
        request_body = {k: v for k, v in {
            "promptIdentifier": prompt_id,
            "description": description,
            "clientToken": client_token,
            "tags": tags
        }.items() if v is not None}
        
        # Create the prompt version
        response = prompt_client.bedrock_client.create_prompt_version(**request_body)
        
        return json.dumps(response, default=str)
    except Exception as e:
        logger.error(f"Error creating prompt version for {prompt_id}: {e}")
        return json.dumps({"error": str(e)})


@mcp.tool(name="DeletePrompt")
async def delete_prompt_tool(
    prompt_id: str = Field(
        ...,
        description="ID of the prompt to delete",
    ),
    prompt_version: Optional[str] = Field(
        None,
        description="Optional version of the prompt to delete. Omit to delete the entire prompt.",
    ),
) -> str:
    """Delete a prompt or a specific version of a prompt.
    
    This tool deletes either an entire prompt or a specific version of a prompt.
    
    ## Required parameters:
    - prompt_id: The ID of the prompt to delete
    
    ## Optional parameters:
    - prompt_version: The version of the prompt to delete. Omit to delete the entire prompt.
    
    ## Response format:
    The response is a JSON object containing the ID and version of the deleted prompt.
    """
    try:
        # Build request parameters and filter out None values
        params = {k: v for k, v in {
            "promptIdentifier": prompt_id,
            "promptVersion": prompt_version
        }.items() if v is not None}
        
        # Call the delete_prompt API
        response = prompt_client.bedrock_client.delete_prompt(**params)
        
        return json.dumps(response, default=str)
    except Exception as e:
        logger.error(f"Error deleting prompt {prompt_id}: {e}")
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