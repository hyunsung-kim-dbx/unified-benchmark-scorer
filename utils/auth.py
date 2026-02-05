"""
Authentication utilities for Databricks Apps.

Handles on-behalf-of-users authorization pattern.
"""

import os
from typing import Optional, Dict

import streamlit as st
from databricks import sdk
from databricks.sdk import WorkspaceClient


def get_user_context() -> Dict[str, Optional[str]]:
    """
    Get user context from Databricks Apps forwarded headers.

    Returns:
        {
            "email": User's email address,
            "access_token": User's OAuth token for on-behalf-of queries,
            "user": Username
        }

    Note: Returns None values when not running in Databricks Apps.
    """
    try:
        headers = st.context.headers
        return {
            "email": headers.get("X-Forwarded-Email"),
            "access_token": headers.get("X-Forwarded-Access-Token"),
            "user": headers.get("X-Forwarded-User"),
        }
    except Exception:
        # Not running in Streamlit or no headers available
        return {
            "email": None,
            "access_token": None,
            "user": None,
        }


def get_service_principal_config() -> sdk.config.Config:
    """
    Get Service Principal configuration for app-level operations.

    Used for:
    - Storing benchmark results
    - Accessing shared resources
    - Operations that don't need user context
    """
    return sdk.config.Config()


def get_user_workspace_client(
    host: Optional[str] = None
) -> WorkspaceClient:
    """
    Create a WorkspaceClient authenticated as the current user.

    Uses the forwarded access token for on-behalf-of authorization.
    All operations will respect the user's permissions.

    Args:
        host: Optional Databricks host. Uses SP config if not provided.

    Returns:
        WorkspaceClient authenticated as the user

    Raises:
        ValueError: If no access token is available
    """
    user_ctx = get_user_context()

    if not user_ctx["access_token"]:
        raise ValueError(
            "No access token found. "
            "This function requires running in Databricks Apps context."
        )

    sp_config = get_service_principal_config()

    return WorkspaceClient(
        host=host or sp_config.host,
        token=user_ctx["access_token"]
    )


def get_fallback_token() -> Optional[str]:
    """
    Get token with fallback for local development.

    Priority:
    1. User's forwarded token (Databricks Apps)
    2. Environment variable DATABRICKS_TOKEN (local dev)
    3. None
    """
    user_ctx = get_user_context()

    if user_ctx["access_token"]:
        return user_ctx["access_token"]

    return os.getenv("DATABRICKS_TOKEN")


def require_auth() -> Dict[str, str]:
    """
    Require authentication, stopping the app if not authenticated.

    Use this at the top of pages that require user context.

    Returns:
        User context dict with email, access_token, user

    Raises:
        st.stop(): Stops Streamlit execution if not authenticated
    """
    user_ctx = get_user_context()

    if not user_ctx["access_token"]:
        st.error("⚠️ Authentication required. Please access this app through Databricks.")
        st.stop()

    return user_ctx
