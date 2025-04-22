"""Pytest configuration file."""

import pytest


# Mark all tests as asyncio
pytest_plugins = ["pytest_asyncio"]


# Used by pytest-asyncio
def pytest_addoption(parser):
    """Add pytest command line options."""
    parser.addini("asyncio_mode", default="strict", help="default asyncio mode")
    

# Configure asyncio
def pytest_configure(config):
    """Configure pytest."""
    # This setting is needed for pytest-asyncio
    config.inicfg["asyncio_mode"] = "strict" 