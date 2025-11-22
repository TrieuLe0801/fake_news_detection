import io

import pandas as pd
from sqlalchemy.orm import sessionmaker

from src.data_models.health_news import Base, HealthNews


def insert_or_update(df: pd.DataFrame, engine: object, mode: str = "upsert"):
    """
    Bulk insert a pandas DataFrame into PostgreSQL using COPY with ON CONFLICT handling.

    Args:
        df (pd.DataFrame): Data to insert.
        engine (object): SQLAlchemy engine connected to the PostgreSQL database.
        mode (str): "upsert" (insert or update existing) or "ignore" (skip duplicates).
    """
    # Ensure only valid columns from model are included
    # Ensure table exists (create it if missing)
    Base.metadata.create_all(engine)
    valid_columns = [
        col.name
        for col in HealthNews.__table__.columns
        if not col.primary_key and col.name in df.columns
    ]
    df = df[valid_columns]

    table_name = HealthNews.__tablename__

    with sessionmaker(bind=engine)() as session:
        raw_conn = session.connection().connection
        with raw_conn.cursor() as cursor:
            # Step 1: Create temporary table
            cursor.execute(
                f"CREATE TEMP TABLE temp_{table_name} AS SELECT * FROM {table_name} LIMIT 0;"
            )

            # Step 2: COPY into temporary table
            with io.StringIO() as buffer:
                df.to_csv(buffer, index=False, header=False)
                buffer.seek(0)
                cursor.copy_expert(
                    f"""
                    COPY temp_{table_name} ({','.join(df.columns)})
                    FROM STDIN WITH (FORMAT CSV, DELIMITER ',', NULL '', QUOTE '"')
                    """,
                    buffer,
                )

            # Step 3: Merge data with conflict handling
            if mode == "ignore":
                conflict_action = "DO NOTHING"
            elif mode == "upsert":
                update_clause = ", ".join(
                    [f"{col}=EXCLUDED.{col}" for col in df.columns if col != "url"]
                )
                conflict_action = f"DO UPDATE SET {update_clause}"
            else:
                raise ValueError("Invalid mode. Use 'upsert' or 'ignore'.")

            cursor.execute(
                f"""
                INSERT INTO {table_name} ({','.join(df.columns)})
                SELECT {','.join(df.columns)} FROM temp_{table_name}
                ON CONFLICT (url) {conflict_action};
            """
            )
            # Remove temporary table after insert or upsert
            cursor.execute(f"DROP TABLE IF EXISTS temp_{table_name};")

        raw_conn.commit()


def get_all_data(engine: object):
    with sessionmaker(bind=engine)() as session:
        health_news = session.query(HealthNews).order_by(HealthNews.id).all()
        return health_news


# def get_data_at_df(engine: object):
#     with sessionmaker(bind=engine)() as session:
#         health_news = session.query(HealthNews).order_by(HealthNews.id).all()
