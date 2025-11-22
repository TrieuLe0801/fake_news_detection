from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, Integer, String, Text
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class HealthNews(Base):
    __tablename__ = "health_news"

    id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String(500), nullable=False)
    url = Column(String(1000), unique=True, nullable=False)
    source = Column(String(200), nullable=True)
    summary = Column(Text, nullable=True)
    content = Column(Text, nullable=True)
    published_at = Column(DateTime, nullable=True)
    crawled_at = Column(DateTime, default=datetime.utcnow)
    is_fake = Column(Boolean, default=None)  # Can be updated after classifying
    normalized_content = Column(Text, nullable=True)

    def __repr__(self):
        return f"<HealthNews(title={self.title[:50]}, source={self.source})>"
