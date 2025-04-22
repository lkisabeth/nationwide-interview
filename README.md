# AWS Bedrock Prompt Management MCP Server

This MCP (Model Context Protocol) server provides an interface to AWS Bedrock's Prompt Management features, allowing generative AI assistants to retrieve, create, and update prompts stored in your AWS Bedrock account.

## Features

- **Discover Prompts**: Find and explore all available prompts in your AWS Bedrock account
- **Create Prompts**: Create new prompts with variables, system instructions, and tool configurations
- **Update Prompts**: Modify existing prompts and create new versions
- **Execute Prompts**: Run prompts directly using the Amazon Bedrock runtime
- **Prompt Variants**: Create and test prompt variants to compare different approaches

### AWS Requirements

1. **AWS CLI Configuration**: You must have the AWS CLI configured with credentials and an AWS_PROFILE that has access to Amazon Bedrock and Prompt Management
2. **Amazon Bedrock Model Access**: You must have enabled model access for the foundation models you intend to use
3. **IAM Permissions**: Your IAM role/user must have appropriate permissions to:
   - List, create, and update prompts in Bedrock Prompt Management
   - Invoke Bedrock models through the runtime API
   - Create and access prompt versions

## Installation

You can add this MCP server to your MCP configuration file (e.g. for Amazon Q Developer CLI MCP, `~/.aws/amazonq/mcp.json`):
(I tested with Cursor and SSE)

```json
{
  "mcpServers": {
    "bedrock_prompt_management_mcp_server": {
      "url": "http://127.0.0.1:8888/sse",
      "env": {
        "AWS_PROFILE": "your-profile-name",
        "AWS_REGION": "us-east-1"
      },
      "disabled": false,
      "autoApprove": []
    }
  }
}
```

## Usage

This MCP server exposes the following tools:

1. `ListPrompts` - Lists all the prompts in your Bedrock account
2. `GetPrompt` - Gets a specific prompt by ID
3. `CreatePrompt` - Creates a new prompt with specified configuration
4. `UpdatePrompt` - Updates an existing prompt
5. `CreatePromptVersion` - Creates a new version of an existing prompt

## Example

```
# List all prompts
Tool use: ListPrompts

# Get a specific prompt
Tool use: GetPrompt with these inputs: promptId["my-prompt-id"]

# Create a new prompt
Tool use: CreatePrompt with these inputs: name["Product description generator"] description["Generates product descriptions based on key features"] modelId["anthropic.claude-3-sonnet-20240229-v1:0"] promptText["You are a product description writer. Write a compelling description for a product with these features: {{features}}"] variables[["features"]]

# Create a new version of a prompt
Tool use: CreatePromptVersion with these inputs: promptId["my-prompt-id"] description["Updated version with better system instructions"]
```

## Development

To develop this MCP server:

1. Clone the repository
2. Create a virtual environment: `python venv venv`
                            `source venv/bin/activate`
3. Install dependencies: `pip install -e .`
4. Run the server with SSE: `./run.sh --sse