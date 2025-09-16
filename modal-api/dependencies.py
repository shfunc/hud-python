"""
HUD Gym API - Authentication dependencies for FastAPI.

This module provides authentication-related dependencies for API endpoints,
including API key validation and user-team membership verification.
"""
from __future__ import annotations

import logging

from fastapi import Depends, Header, HTTPException, Request, status
from pydantic import BaseModel

from .supabase import get_supabase_client

LOGGER = logging.getLogger(__name__)


async def verify_api_key(authorization: str = Header(...)) -> str:
    """
    Verify the API key from the Authorization header.

    First checks Redis cache, then falls back to database query if not found.
    Caches successful validations in Redis with a 24-hour TTL.

    Args:
        authorization: The Authorization header value

    Returns:
        str: The user_id of the API key

    Raises:
        HTTPException: If the API key is invalid
    """
    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization header format",
        )

    api_key = authorization.replace("Bearer ", "")

    # Fall back to database validation
    supabase = await get_supabase_client()

    # Determine query strategy based on API key format
    result = None
    try:
        result = (
            await supabase.table("api_keys")
            .select("user_team_membership_id")
            .eq("api_key", api_key)
            .eq("revoked", False)
            .execute()
        )
    except Exception as e:
        LOGGER.warning("Error querying database for API key: %s", str(e))

    # If still not found, the API key is invalid
    if not result or not result.data:
        masked_api_key = api_key[:4] + "*" * (len(api_key) - 8) + api_key[-4:]
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid or revoked API key: {masked_api_key}",
        )

    user_team_membership_id = result.data[0]["user_team_membership_id"]

    if not user_team_membership_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key not associated with any team membership",
        )

    LOGGER.info("Verified API key for membership_id: %s", user_team_membership_id)

    return user_team_membership_id


async def get_membership_from_id(membership_id: str) -> dict:
    """
    Get the user-team membership details from the membership ID.

    Args:
        membership_id: The user-team membership ID

    Returns:
        dict: The membership record

    Raises:
        HTTPException: If the membership is not found or deleted
    """
    supabase = await get_supabase_client()

    membership_query = await (
        supabase.table("user_team_memberships")
        .select("*")
        .eq("id", membership_id)
        .eq("deleted", False)
        .limit(1)
        .execute()
    )

    if not membership_query.data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Membership not found or has been deleted",
        )

    return membership_query.data[0]


async def get_user_from_id(user_id: str) -> dict:
    """
    Get user details from user ID.

    Args:
        user_id: The user ID

    Returns:
        dict: The user record

    Raises:
        HTTPException: If the user is not found or deleted
    """
    supabase = await get_supabase_client()

    user_query = await (
        supabase.table("users")
        .select("*")
        .eq("id", user_id)
        .eq("deleted", False)
        .limit(1)
        .execute()
    )

    if not user_query.data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found or has been deleted",
        )

    return user_query.data[0]


async def get_team_from_id(team_id: str) -> dict:
    """
    Get team details from team ID.

    Args:
        team_id: The team ID

    Returns:
        dict: The team record

    Raises:
        HTTPException: If the team is not found or deleted
    """
    supabase = await get_supabase_client()

    team_query = await (
        supabase.table("teams")
        .select("*")
        .eq("id", team_id)
        .eq("deleted", False)
        .limit(1)
        .execute()
    )

    if not team_query.data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Team not found or has been deleted",
        )

    return team_query.data[0]


class AuthContext(BaseModel):
    """Container for authentication context information."""

    user: dict
    team: dict
    membership: dict
    api_key: str


async def get_membership(
    request: Request,
    validated_membership_id: str = Depends(verify_api_key),
) -> dict:
    """
    Get the user-team membership from the API key.

    Args:
        request: The FastAPI request object
        validated_membership_id: The membership ID from API key verification

    Returns:
        dict: The membership record
    """
    # Not cached, fetch from DB
    membership_data = await get_membership_from_id(validated_membership_id)

    return membership_data


async def get_auth_context(
    request: Request,
    membership: dict = Depends(get_membership),
    authorization: str = Header(...),
) -> AuthContext:
    """
    Get the complete authentication context from the membership.

    This provides a complete context containing the user, team, and membership details
    that can be injected into API routes.

    Args:
        request: The FastAPI request object
        membership: The membership record from get_membership

    Returns:
        AuthContext: Object containing user, team and membership details
    """
    user_data = await get_user_from_id(membership["user_id"])
    team_data = await get_team_from_id(membership["team_id"])
    
    # Extract API key from Bearer token
    api_key = authorization.replace("Bearer ", "")

    return AuthContext(user=user_data, team=team_data, membership=membership, api_key=api_key)
