"""Client interfaces for AWS Bedrock Prompt Management API."""

import os
from typing import Optional

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