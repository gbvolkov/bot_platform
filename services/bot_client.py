"""Compatibility import; implementation lives in platform_client.http."""
from platform_sdk.compatibility import alias_module
alias_module(__name__, globals(), "platform_client.http")
