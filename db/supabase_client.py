"""
db/supabase_client.py — single shared Supabase client instance.

Reads credentials from the environment (loaded from .env locally, or from
Railway's environment variables in production):

    SUPABASE_URL   e.g. https://yourproject.supabase.co
    SUPABASE_KEY   the project anon (public) key

Import the client anywhere with:

    from db.supabase_client import supabase
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from supabase import create_client, Client

# Load .env if present (no-op in production where vars are already set).
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError(
        "SUPABASE_URL and SUPABASE_KEY must be set. "
        "Copy .env.example to .env and fill in your credentials, "
        "or configure them in the Railway dashboard."
    )

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
