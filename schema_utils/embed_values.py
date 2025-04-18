import pickle
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass
class ColumnVectorIndex:
    index: faiss.IndexFlatL2
    original_strings: list[str]

    def get_similar_strings(self, emb_model: SentenceTransformer, query_string: str, k: int = 3) -> list[str]:
        query_embedding = emb_model.encode([query_string])
        results = self.index.search(query_embedding, k=min(k, len(self.original_strings)))
        return [self.original_strings[i] for i in results[1][0]]


def embed_values_in_db(bench: str, db_base_path: str, db_id: str, embed_model: SentenceTransformer):
    """Embed the values of the TEXT-like columns of the tables in the database.
    Store the embeddings in a FAISS index and save it in the end.
    Vector Index (FAISS) design:
        - each column --> a set of vectors (embeddings)
        - query embeddings --> retrieve some TEXT values in this column
        - which means every (table, TEXT-like column) has a vector index
        - we have a dict[table_name][column_name] = FAISS index
    """
    print(f"Start embedding {bench}: {db_id}")
    TEXT_COLUMN_TYPES = ["TEXT", "VARCHAR", "CHAR", "DATE", "DATETIME"]
    all_indexes: dict[tuple[str, str], faiss.IndexFlatL2] = {}

    # Connect to the database
    conn = sqlite3.connect(db_base_path / db_id / f"{db_id}.sqlite")
    cursor = conn.cursor()

    # Get all the tables in the database, we will embed the values of each table
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    table_names = [table[0] for table in tables if table[0] != "sqlite_sequence"]

    for table_name in table_names:
        # Get the columns of the table
        cursor.execute(f'PRAGMA table_info("{table_name}")')
        columns = cursor.fetchall()

        for column in columns:
            col_name = column[1]
            col_type = column[2]
            if any(TEXT_COLUMN_TYPE in col_type.upper() for TEXT_COLUMN_TYPE in TEXT_COLUMN_TYPES):
                cursor.execute(f'SELECT DISTINCT "{col_name}" FROM "{table_name}" LIMIT 5000')
                values = []
                while True:
                    try:
                        row = cursor.fetchone()
                        if row is None:
                            break
                        value = row[0]
                        if isinstance(value, str) and (value is not None) and str(value).strip():
                            values.append(value)
                    except (UnicodeDecodeError, sqlite3.OperationalError):
                        continue

                if len(values) == 0:
                    continue

                # Embed the values with FAISS
                try:
                    embeddings: np.ndarray = embed_model.encode(values)
                except Exception as e:
                    print(f"Warning: Could not embed values in {table_name}.{col_name}: {e}")
                    print(f"Type = {type(values[0])}, TEXT_COLUMN_TYPE = {col_type.upper()}")
                    exit(0)

                index = faiss.IndexFlatL2(embeddings.shape[1])
                index.add(embeddings)
                all_indexes[(table_name, col_name)] = ColumnVectorIndex(index, values)

    OUTPUT_DIR = Path("/ssd/yizhe/lin/indexes") / bench
    if not OUTPUT_DIR.exists():
        OUTPUT_DIR.mkdir(parents=True)

    with open(OUTPUT_DIR / f"{db_id}.pkl", "wb") as f:
        pickle.dump(all_indexes, f)


# if __name__ == "__main__":
#     DEFAULT_MODEL = "Alibaba-NLP/gte-large-en-v1.5"
#     emb_model = SentenceTransformer(DEFAULT_MODEL, device="cuda:3", trust_remote_code=True)
#     # bench = ["spider-train", "spider-dev"]  #
#     bench = ["bird-train", "bird-dev"]
#     for b in bench:
#         match b:
#             case "spider-train":
#                 DB_BASE = Path().cwd().parent / "dataset" / "spider" / "database"
#             case "spider-dev":
#                 DB_BASE = Path().cwd().parent / "dataset" / "spider" / "database"
#             case "bird-train":
#                 DB_BASE = Path("/ssd/yizhe/lin/bird/bird-train/train_databases")
#             case "bird-dev":
#                 DB_BASE = Path("/ssd/yizhe/lin/bird/bird-dev/dev_databases")
#             case _:
#                 raise ValueError(f"Unknown benchmark: {b}")

#         # each database is in a folder with the name of the database
#         for db_path in DB_BASE.iterdir():
#             if not db_path.is_dir():
#                 continue

#             db_id = db_path.stem
#             embed_values_in_db(b, DB_BASE, db_id, emb_model)

#     print("Done!")
