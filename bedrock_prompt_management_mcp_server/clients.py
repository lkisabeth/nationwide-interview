"""Client interfaces for AWS Bedrock Prompt Management API."""

import os
from typing import Dict, List, Optional, Any

import boto3
from loguru import logger


def get_bedrock_client(region_name: Optional[str] = None, profile_name: Optional[str] = None):
    """Get a Bedrock client using the provided configuration.
    
    Args:
        region_name: AWS region name, defaults to AWS_REGION env var
        profile_name: AWS profile name, defaults to AWS_PROFILE env var
    
    Returns:
        A boto3 Bedrock client
    """
    region = region_name or os.getenv("AWS_REGION")
    profile = profile_name or os.getenv("AWS_PROFILE")
    
    session = boto3.Session(profile_name=profile, region_name=region)
    
    try:
        return session.client(service_name="bedrock-agent")
    except Exception as e:
        logger.error(f"Error creating Bedrock client: {e}")
        raise e


class BedrockPromptManagementClient:
    """Client for interacting with Amazon Bedrock Prompt Management API."""
    
    def __init__(
        self, 
        bedrock_client=None, 
        region_name: Optional[str] = None,
        profile_name: Optional[str] = None
    ):
        """Initialize the Bedrock Prompt Management client.
        
        Args:
            bedrock_client: Optional pre-configured Bedrock client
            region_name: AWS region name, defaults to AWS_REGION env var
            profile_name: AWS profile name, defaults to AWS_PROFILE env var
        """
        self.bedrock_client = bedrock_client or get_bedrock_client(
            region_name=region_name, profile_name=profile_name
        )
    
    def list_prompts(self, max_results: int = 100) -> List[Dict[str, Any]]:
        """List all prompts in the Bedrock account.
        
        Args:
            max_results: Maximum number of results to return
            
        Returns:
            List of prompt summaries
        """
        try:
            paginator = self.bedrock_client.get_paginator("list_prompts")
            pages = paginator.paginate(maxResults=max_results)
            
            prompts = []
            for page in pages:
                prompts.extend(page.get("promptSummaries", []))
                
            return prompts
        except Exception as e:
            logger.error(f"Error listing prompts: {e}")
            raise e
    
    def get_prompt(self, prompt_id: str, version_number: Optional[int] = None) -> Dict[str, Any]:
        """Get a specific prompt by ID.
        
        Args:
            prompt_id: The ID of the prompt to retrieve
            version_number: Optional version number to retrieve
            
        Returns:
            Prompt details
        """
        try:
            kwargs = {"promptIdentifier": prompt_id}
            if version_number:
                kwargs["promptVersion"] = version_number
                
            response = self.bedrock_client.get_prompt(**kwargs)
            return response
        except Exception as e:
            logger.error(f"Error getting prompt {prompt_id}: {e}")
            raise e
    
    def create_prompt(
        self,
        name: str,
        description: str,
        model_id: str,
        template: str,
        variables: Optional[List[Dict[str, Any]]] = None,
        system_instruction: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        inference_config: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Create a new prompt.
        
        Args:
            name: Name of the prompt
            description: Description of the prompt
            model_id: Model ID to use for the prompt
            template: Template text for the prompt
            variables: Optional list of variables for the prompt
            system_instruction: Optional system instruction for the prompt
            messages: Optional list of messages for the prompt
            inference_config: Optional inference configuration
            tools: Optional tools configuration
            metadata: Optional metadata for the prompt
            
        Returns:
            The created prompt details
        """
        try:
            kwargs = {
                "name": name,
                "description": description,
                "modelId": model_id,
            }
            
            # Add prompt configuration based on provided parameters
            if messages:
                prompt_config = {
                    "promptType": "STRUCTURED",
                    "structuredPromptConfig": {
                        "messages": messages
                    }
                }
                
                if system_instruction:
                    prompt_config["structuredPromptConfig"]["systemInstruction"] = system_instruction
                
                if tools:
                    prompt_config["structuredPromptConfig"]["tools"] = tools
                    
                kwargs["promptConfiguration"] = prompt_config
            else:
                kwargs["promptConfiguration"] = {
                    "promptType": "TEXT",
                    "textPromptConfig": {
                        "text": template
                    }
                }
            
            # Add variables
            if variables:
                kwargs["variableConfiguration"] = {
                    "variables": variables
                }
            
            # Add inference configuration
            if inference_config:
                kwargs["inferenceConfiguration"] = inference_config
                
            # Add metadata
            if metadata:
                kwargs["metadata"] = metadata
                
            response = self.bedrock_client.create_prompt(**kwargs)
            return response
        except Exception as e:
            logger.error(f"Error creating prompt: {e}")
            raise e
    
    def update_prompt(
        self,
        prompt_id: str,
        description: Optional[str] = None,
        model_id: Optional[str] = None,
        template: Optional[str] = None,
        variables: Optional[List[Dict[str, Any]]] = None,
        system_instruction: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        inference_config: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Update an existing prompt.
        
        Args:
            prompt_id: ID of the prompt to update
            description: Optional new description
            model_id: Optional new model ID
            template: Optional new template text
            variables: Optional new variables
            system_instruction: Optional new system instruction
            messages: Optional new messages
            inference_config: Optional new inference configuration
            tools: Optional new tools configuration
            metadata: Optional new metadata
            
        Returns:
            The updated prompt details
        """
        try:
            kwargs = {
                "promptIdentifier": prompt_id,
            }
            
            if description:
                kwargs["description"] = description
                
            if model_id:
                kwargs["modelId"] = model_id
                
            # Update prompt configuration based on provided parameters
            if messages or template or system_instruction or tools:
                prompt_config = {}
                
                if messages:
                    prompt_config["promptType"] = "STRUCTURED"
                    prompt_config["structuredPromptConfig"] = {
                        "messages": messages
                    }
                    
                    if system_instruction:
                        prompt_config["structuredPromptConfig"]["systemInstruction"] = system_instruction
                    
                    if tools:
                        prompt_config["structuredPromptConfig"]["tools"] = tools
                elif template:
                    prompt_config["promptType"] = "TEXT"
                    prompt_config["textPromptConfig"] = {
                        "text": template
                    }
                
                if prompt_config:
                    kwargs["promptConfiguration"] = prompt_config
            
            # Update variables
            if variables:
                kwargs["variableConfiguration"] = {
                    "variables": variables
                }
            
            # Update inference configuration
            if inference_config:
                kwargs["inferenceConfiguration"] = inference_config
                
            # Update metadata
            if metadata:
                kwargs["metadata"] = metadata
                
            response = self.bedrock_client.update_prompt(**kwargs)
            return response
        except Exception as e:
            logger.error(f"Error updating prompt {prompt_id}: {e}")
            raise e
    
    def create_prompt_version(
        self,
        prompt_id: str,
        description: str
    ) -> Dict[str, Any]:
        """Create a new version of an existing prompt.
        
        Args:
            prompt_id: ID of the prompt to version
            description: Description of the new version
            
        Returns:
            The new version details
        """
        try:
            response = self.bedrock_client.create_prompt_version(
                promptIdentifier=prompt_id,
                description=description
            )
            return response
        except Exception as e:
            logger.error(f"Error creating version for prompt {prompt_id}: {e}")
            raise e