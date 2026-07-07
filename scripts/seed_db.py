import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import logging
import psycopg2
import litellm
from evaluator.config import config
from evaluator.utils.mock_embedding import is_mock_key, generate_mock_embedding

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DBSeeder")

SAMPLE_CORPUS = [
    "Clinical protocol payload: Patient eligibility relies on strict physiological boundaries including standard biomarker baselines.",
    "Database Context: Maximum allowable budget ceiling for campaign cluster Alpha is set to exactly $50,000 for Q3 operations.",
    "System Constraint: Direct neural generations must bypass unvetted transactional state commits to prevent data corruption."
]

def seed_database():
    logger.info("Connecting to Postgres database instance...")
    with psycopg2.connect(config.DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

            cur.execute("DROP TABLE IF EXISTS document_chunks;")
            cur.execute("""
                CREATE TABLE document_chunks (
                    id SERIAL PRIMARY KEY,
                    content TEXT NOT NULL,
                    embedding vector(1536)
                );
            """)
            conn.commit()
            logger.info("Database schema and pgvector structures created successfully.")

            for chunk in SAMPLE_CORPUS:
                if is_mock_key(config.OPENAI_API_KEY):
                    vector = generate_mock_embedding(chunk)
                    logger.info("Using mock embedding for offline mode.")
                else:
                    embed_resp = litellm.embedding(model=config.EMBEDDING_MODEL, input=[chunk])
                    vector = embed_resp['data'][0]['embedding']

                cur.execute(
                    "INSERT INTO document_chunks (content, embedding) VALUES (%s, %s);",
                    (chunk, vector)
                )
            conn.commit()
    logger.info("Successfully seeded database with core semantic vectors.")

if __name__ == "__main__":
    seed_database()
