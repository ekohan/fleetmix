import argparse
import logging
import sqlite3
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_data_dir() -> Path:
    """Get the data directory path."""
    return Path(__file__).resolve().parent


def import_data_to_unified_sales(db_path: Path):
    """Import all sales data into the unified sales table."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # Drop existing table if it exists
        logger.info("Dropping existing sales table...")
        cursor.execute("DROP TABLE IF EXISTS sales")

        # Create unified table
        logger.info("Creating new sales table...")
        schema_path = get_data_dir() / "queries" / "create_sales_table.sql"
        with open(schema_path, encoding="utf-8") as f:
            cursor.executescript(f.read())

        # Import 2023 data
        logger.info("Importing 2023 data...")
        data_2023_path = get_data_dir() / "raw" / "sales_2023_transformed.sql"
        with open(data_2023_path, encoding="utf-8") as f:
            cursor.executescript(f"BEGIN TRANSACTION;\n{f.read()}\nCOMMIT;")

        # Import 2024 data
        logger.info("Importing 2024 data...")
        data_2024_path = get_data_dir() / "raw" / "sales_2024_transformed.sql"
        with open(data_2024_path, encoding="utf-8") as f:
            cursor.executescript(f"BEGIN TRANSACTION;\n{f.read()}\nCOMMIT;")

        # Verify row counts
        cursor.execute("SELECT SourceYear, COUNT(*) FROM sales GROUP BY SourceYear")
        counts = cursor.fetchall()
        for year, count in counts:
            logger.info(f"Imported {count} rows for year {year}")

        conn.commit()
        logger.info("Import completed successfully")

    except sqlite3.Error as e:
        logger.error(f"An error occurred: {e}")
        conn.rollback()
    finally:
        conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Import sales data into unified table."
    )
    parser.add_argument("--db", default="opperar.db", help="Path to SQLite database.")
    args = parser.parse_args()

    # Get the script's directory relative to where it's being invoked
    db_path = get_data_dir() / args.db
    import_data_to_unified_sales(db_path)
