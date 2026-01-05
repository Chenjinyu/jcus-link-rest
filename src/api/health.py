from fastapi import APIRouter

from src.config.settings import settings


router = APIRouter(prefix="/health", tags=["health"])


@router.get("")
async def health_check() -> dict[str, str]:
    return {
        "status": "200",
        "message": "Healthy",
        "version": settings.app_version
    }
