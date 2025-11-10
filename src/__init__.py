from .clients.postgres_client import *
from .data_models.health_news import Base, HealthNews

__all__ = ["HealthNews", "Base", "insert_or_update"]
