import json
import re
import ast
from typing import Any, Union
from datetime import date, datetime

def flatten_dict_to_text( data: dict[str, Any]) -> str:
        """Convert dictionary to searchable text"""
        parts = []
        for key, value in data.items():
            if isinstance(value, (str, int, float, bool)):
                parts.append(f"{key}: {value}")
            elif isinstance(value, list):
                parts.append(f"{key}: {', '.join(map(str, value))}")
            elif isinstance(value, dict):
                parts.append(f"{key}: {flatten_dict_to_text(value)}")
        return ". ".join(parts)
    
def parse_date(date_value: Union[str, date] | None) -> date | None:
        """
        Convert string date to Python date object for asyncpg.

        Args:
            date_value: String in 'YYYY-MM-DD' format or date object or None

        Returns:
            date object or None
        """
        if date_value is None:
            return None
        if isinstance(date_value, date):
            return date_value
        if isinstance(date_value, str):
            # Parse 'YYYY-MM-DD' format
            try:
                return datetime.strptime(date_value, "%Y-%m-%d").date()
            except ValueError:
                # Try other common formats
                try:
                    return datetime.strptime(date_value, "%Y/%m/%d").date()
                except ValueError:
                    raise ValueError(
                        f"Invalid date format: {date_value}. Expected 'YYYY-MM-DD' or 'YYYY/MM/DD'"
                    )
                    
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split text into overlapping word chunks."""
    words = text.split()
    if len(words) <= chunk_size:
        return [text]
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks


def format_exception_message(exc: Exception) -> str:
    message = str(exc).strip()
    if message:
        return message
    return f"{type(exc).__name__}: {exc!r}"
