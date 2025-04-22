#!/bin/bash

# Run the AWS Bedrock Prompt Management MCP Server
# This script ensures proper environment setup and runs the server

set -e

# Check if Python 3.10+ is installed
python_version=$(python --version 2>&1 | awk '{print $2}')
major_version=$(echo $python_version | cut -d. -f1)
minor_version=$(echo $python_version | cut -d. -f2)

if [ "$major_version" -lt 3 ] || ([ "$major_version" -eq 3 ] && [ "$minor_version" -lt 10 ]); then
    echo "Error: Python 3.10 or higher is required"
    echo "Current version: $python_version"
    exit 1
fi

# Check if AWS credentials are configured
if [ -z "$AWS_ACCESS_KEY_ID" ] || [ -z "$AWS_SECRET_ACCESS_KEY" ]; then
    if [ -z "$AWS_PROFILE" ]; then
        echo "Warning: No AWS credentials found in environment variables"
        echo "Make sure you have configured AWS credentials through environment variables or AWS_PROFILE"
    else
        echo "Using AWS profile: $AWS_PROFILE"
    fi
fi

# Check if AWS region is set
if [ -z "$AWS_REGION" ]; then
    echo "Warning: AWS_REGION is not set, defaulting to us-east-1"
    export AWS_REGION="us-east-1"
else
    echo "Using AWS region: $AWS_REGION"
fi

# Install the package in development mode if not already installed
if ! pip show bedrock-prompt-management-mcp-server &> /dev/null; then
    echo "Installing the MCP server package..."
    pip install -e .
fi

# Parse command line arguments
PORT="8888"
TRANSPORT="stdio"

while [[ $# -gt 0 ]]; do
    case $1 in
        --port)
            PORT="$2"
            shift 2
            ;;
        --sse)
            TRANSPORT="sse"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--port PORT] [--sse]"
            exit 1
            ;;
    esac
done

# Run the server
echo "Starting AWS Bedrock Prompt Management MCP Server..."
if [ "$TRANSPORT" == "sse" ]; then
    echo "Using SSE transport on port $PORT"
    export FASTMCP_TRANSPORT="sse"
    python3 -m bedrock_prompt_management_mcp_server.server --sse --port "$PORT"
else
    echo "Using stdio transport"
    export FASTMCP_TRANSPORT="stdio"
    python3 -m bedrock_prompt_management_mcp_server.server
fi 