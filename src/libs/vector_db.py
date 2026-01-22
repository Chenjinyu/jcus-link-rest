from typing import Any, Dict, List
import functools
import asyncpg
from asyncpg.pool import PoolConnectionProxy
from supabase import Client, create_client

from src.config import settings

class VectorDB:
    """Vector Database Implementation"""
    
    def __init__(
            self,
            supabase_url: str | None = None,
            supabase_key: str | None = None,
            postgres_url: str | None = None
            ) -> None:
        
        self.supabase_url = supabase_url or settings.supabase_url
        if self.supabase_url is None:
            raise ValueError("Supabase URL must be provided")
        self.supabase_key = supabase_key or settings.supabase_key
        if self.supabase_key is None:
            raise ValueError("Supabase Key must be provided")
        self.postgres_url = postgres_url or settings.supabase_postgres_url
        if self.postgres_url is None:
            raise ValueError("Postgres URL must be provided")

        self.supabase_db: Client = create_client(self.supabase_url, self.supabase_key)
        self.pg_pool: asyncpg.pool.Pool | None = None
    
    def get_supabase_db_client(self):
        return self.supabase_db
    
    async def init_pool(self):
        """Initialize asyncpg connection pool"""
        if self.pg_pool:
            return self.pg_pool
        
        return await asyncpg.create_pool(
            self.postgres_url,
            min_size=2,
            max_size=10,
            command_timeout=60,
            statement_cache_size=0,  # Disable prepared statement caching.
        )
        
    def with_db_transcation(self, func):
        """Decorator to run function within a database transaction"""
        @functools.wraps(func)
        async def wrapper(self, *args, **kwargs):
            if not self.pg_pool:
                self.pg_pool = await self.init_pool()
            async with self.pg_pool.acquire() as conn:
                async with conn.transaction():
                    kwargs['conn'] = conn
                    return await func(self, *args, **kwargs)
        return wrapper
        
    async def get_connection(self) -> PoolConnectionProxy:
        """Get a connection from the pool"""
        if not self.pg_pool:
            self.pg_pool = await self.init_pool()
        return await self.pg_pool.acquire()
    
    async def release_connection(self, conn: PoolConnectionProxy) -> None:
        """Release a connection back to the pool"""
        if self.pg_pool:
            await self.pg_pool.release(conn)
            
    async def close_pool(self) -> None:
        """Close the connection pool"""
        if self.pg_pool:
            await self.pg_pool.close()
            
    async def execute_rpc_function(self, function_name: str, *args) -> List[Dict[str, Any]]:
        # Execute a PostgreSQL RPC function
        conn = await self.get_connection()
        try:
            result = await conn.fetch(f"SELECT * FROM {function_name}({','.join(['$' + str(i) for i in range(1, len(args) + 1)])})", *args)
            return [dict(row) for row in result]
        finally:
            await self.release_connection(conn)
            
            