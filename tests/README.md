# AWS Bedrock Prompt Management MCP Server Tests

This directory contains tests for the AWS Bedrock Prompt Management MCP Server.

## Running Tests

You can run the tests using pytest:

```bash
# Run all tests
python -m pytest

# Run a specific test file
python -m pytest tests/test_server.py

# Run a specific test
python -m pytest tests/test_server.py::TestListPromptsTool::test_list_prompts_success
```

## Test Structure

- `test_server.py`: Tests for the MCP server tools
- `test_clients.py`: Tests for the Bedrock client interfaces
- `test_mcp_server.py`: Tests for MCP server initialization and configuration

## Test Strategy

These tests use mocking extensively to avoid making real AWS API calls during testing.
The main AWS Bedrock client is mocked, and each test verifies:

1. Correct handling of successful API responses
2. Proper error handling
3. Correct parameter processing
4. Appropriate formatting of responses

## Adding New Tests

When adding a new tool or functionality:

1. Create unit tests for the new function/class
2. Mock all external dependencies
3. Test both success and error cases
4. Ensure coverage of edge cases

## Test Dependencies

The test suite requires:
- pytest
- pytest-cov
- pytest-asyncio

These are included in the project's dev dependencies. 