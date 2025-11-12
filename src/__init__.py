from .clients.postgres_client import *
from .data_models.health_news import Base, HealthNews
from .processor.data_processor import *

__all__ = ["HealthNews", "Base", "insert_or_update"]
