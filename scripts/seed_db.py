import argparse
import psycopg2
import litellm
from evaluator.config import config
from evaluator.utils.mock_embedding import is_mock_key, generate_mock_embedding
from evaluator.logging_config import get_logger, setup_logging

setup_logging()
logger = get_logger("DBSeeder")

SAMPLE_CORPUS = [
    "Clinical protocol payload: Patient eligibility relies on strict physiological boundaries including standard biomarker baselines.",
    "Database Context: Maximum allowable budget ceiling for campaign cluster Alpha is set to exactly $50,000 for Q3 operations.",
    "System Constraint: Direct neural generations must bypass unvetted transactional state commits to prevent data corruption."
]

INSERT_SQL = """
INSERT INTO document_chunks (content, embedding)
VALUES (%s, %s)
ON CONFLICT (content) DO NOTHING;
"""


def _ensure_schema(force_reset: bool = False):
    import subprocess
    import sys

    if force_reset:
        logger.info("Running alembic downgrade to remove existing schema...")
        subprocess.run([sys.executable, "-m", "alembic", "downgrade", "base"], check=False)
        logger.info("Running alembic upgrade to recreate schema...")
        subprocess.run([sys.executable, "-m", "alembic", "upgrade", "head"], check=True)
    else:
        subprocess.run([sys.executable, "-m", "alembic", "upgrade", "head"], check=True)
    logger.info("Database schema ensured via Alembic.")


def seed_database(force_reset: bool = False):
    mode = "force reset" if force_reset else "safe"
    logger.info(f"Running seed_db in {mode} mode.")
    logger.info("Connecting to Postgres database instance...")
    _ensure_schema(force_reset=force_reset)

    with psycopg2.connect(config.DATABASE_URL) as conn:
        with conn.cursor() as cur:
            for chunk in SAMPLE_CORPUS:
                if is_mock_key(config.OPENAI_API_KEY):
                    vector = generate_mock_embedding(chunk)
                    logger.info("Using mock embedding for offline mode.")
                else:
                    embed_resp = litellm.embedding(model=config.EMBEDDING_MODEL, input=[chunk])
                    vector = embed_resp["data"][0]["embedding"]

                cur.execute(INSERT_SQL, (chunk, vector))
            conn.commit()
    logger.info("Successfully seeded database with core semantic vectors.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed the RAG database with sample corpus.")
    parser.add_argument(
        "--force-reset",
        action="store_true",
        help="Reset schema via Alembic before seeding. Destructive! Use for clean development resets.",
    )
    args = parser.parse_args()
    seed_database(force_reset=args.force_reset)