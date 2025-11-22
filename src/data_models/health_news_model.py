from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
from sqlalchemy import func
from sqlalchemy.orm import sessionmaker

from src.clients.postgres_client import get_all_data, insert_or_update
from src.data_models.health_news import HealthNews


class HealthNewsModel:

    def __init__(self, engine):
        """
        Args:
            engine: SQLAlchemy engine
        """
        self.engine = engine
        self.SessionLocal = sessionmaker(bind=engine)
        self._df = None

    # ==================== Data Access Methods ====================

    def get_all_as_objects(self) -> List[HealthNews]:
        """Get all health news as SQLAlchemy objects using existing function"""
        return get_all_data(self.engine)

    def get_all_as_dataframe(self, filters: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """Query all health news as DataFrame with optional filters"""
        with self.SessionLocal() as session:
            query = session.query(HealthNews)

            if filters:
                if "start_date" in filters and filters["start_date"]:
                    query = query.filter(HealthNews.crawled_at >= filters["start_date"])

                if "end_date" in filters and filters["end_date"]:
                    query = query.filter(HealthNews.crawled_at <= filters["end_date"])

                if "sources" in filters and filters["sources"]:
                    query = query.filter(HealthNews.source.in_(filters["sources"]))

                if "is_fake" in filters and filters["is_fake"] is not None:
                    query = query.filter(HealthNews.is_fake == filters["is_fake"])

            # Convert to DataFrame directly
            df = pd.read_sql(query.statement, session.bind)
            return df

    def get_sources(self) -> List[str]:
        """Get unique sources"""
        with self.SessionLocal() as session:
            result = session.query(HealthNews.source).distinct().all()
            return sorted([r[0] for r in result if r[0] is not None])

    def get_date_range(self) -> tuple:
        """Get min and max crawled dates"""
        with self.SessionLocal() as session:
            result = session.query(
                func.min(HealthNews.crawled_at), func.max(HealthNews.crawled_at)
            ).first()
            return result

    def get_summary_stats(self) -> Dict[str, int]:
        """Get summary statistics from database"""
        with self.SessionLocal() as session:
            total = session.query(func.count(HealthNews.id)).scalar()
            fake = (
                session.query(func.count(HealthNews.id))
                .filter(HealthNews.is_fake == True)
                .scalar()
            )
            real = (
                session.query(func.count(HealthNews.id))
                .filter(HealthNews.is_fake == False)
                .scalar()
            )
            sources = session.query(func.count(func.distinct(HealthNews.source))).scalar()

            return {
                "total_articles": total,
                "fake_news": fake or 0,
                "real_news": real or 0,
                "num_sources": sources or 0,
            }

    def bulk_insert_or_update(self, df: pd.DataFrame, mode: str = "upsert"):
        """
        Bulk insert/update using existing postgres client function

        Args:
            df: DataFrame to insert
            mode: "upsert" or "ignore"
        """
        insert_or_update(df, self.engine, mode=mode)

    # ==================== Business Logic Methods ====================

    def load_data(self, filters: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """Load data from database as DataFrame"""
        self._df = self.get_all_as_dataframe(filters=filters)
        return self._df

    def get_dataframe(self) -> pd.DataFrame:
        """Get cached dataframe"""
        if self._df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        return self._df.copy()

    def prepare_for_analysis(self, text_col: str = "content") -> pd.DataFrame:
        """Prepare data for analysis by adding computed columns"""
        df = self.get_dataframe()

        # Add text length features
        df["text_length"] = df[text_col].astype(str).str.len()
        df["word_count"] = df[text_col].astype(str).apply(lambda x: len(x.split()))

        # Add date features
        if "crawled_at" in df.columns:
            df["crawled_at"] = pd.to_datetime(df["crawled_at"])
            df["year"] = df["crawled_at"].dt.year
            df["month"] = df["crawled_at"].dt.month
            df["day"] = df["crawled_at"].dt.day
            df["day_of_week"] = df["crawled_at"].dt.day_name()

        # Handle missing values
        df["is_fake"] = df["is_fake"].fillna(-1).astype(int)  # -1 for unlabeled

        self._df = df
        return df

    def get_text_by_label(self, text_col: str, label_col: str, label_value: int) -> str:
        """Get concatenated text for a specific label"""
        df = self.get_dataframe()
        texts = df[df[label_col] == label_value][text_col].astype(str).tolist()
        return " ".join(texts)

    def get_all_text(self, text_col: str) -> str:
        """Get all concatenated text"""
        df = self.get_dataframe()
        return " ".join(df[text_col].astype(str).tolist())

    def get_metrics(self) -> Dict[str, int]:
        """Get summary metrics"""
        return self.get_summary_stats()

    def get_available_sources(self) -> List[str]:
        """Get list of available sources"""
        return self.get_sources()

    def get_available_date_range(self) -> tuple:
        """Get date range of data"""
        return self.get_date_range()
