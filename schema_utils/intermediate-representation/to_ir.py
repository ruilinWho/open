import json
import os
import sqlite3
from pathlib import Path

import chardet
import pandas as pd
from langchain.utilities.sql_database import SQLDatabase


def string_equivalent(str1: str, str2: str) -> bool:
    """
    Compare two strings for "equality".
    Complete equality or differences only in spaces and underscores are considered equal.

    :param str1: The first string
    :param str2: The second string
    :return: Return True if the strings are considered equal, otherwise return False
    """
    if str1 == str2:
        return True

    normalized_str1 = "".join(char.lower() for char in str1 if char not in " _`")
    normalized_str2 = "".join(char.lower() for char in str2 if char not in " _`")
    return normalized_str1 == normalized_str2


class Schema2IR:
    """
    Input information of a database schema, convert it to a dict representation of the schema.
    Thie intermediate representation (IR) can then be used for schema linking.

    IR format:
    - db_id
    - db_dir
    - db_json
    - bench
    - tables
        - table_id
        - table_name
    """

    ir: dict[str, any]

    def __init__(self, db_id: str, db_dir: Path, db_json: dict, bench: str):
        self.db_id = db_id
        self.db_dir = db_dir
        self.db_json = db_json
        self.bench = bench

        # assert self.bench in ["spider", "bird_train", "bird_dev"]
        self.ir = {"db_id": db_id, "bench": bench}

        self._parse_tables()

    def _parse_tables(self):
        self.ir["tables"] = []
        for table_idx, table_name in enumerate(self.db_json["table_names"]):
            original_table_name = self.db_json["table_names_original"][table_idx]

            table = {
                "table_idx": table_idx,
                "table_name": original_table_name,
                "table_comment": f" -- {table_name}" if not string_equivalent(table_name, original_table_name) else "",
            }

            columns, primary_keys = self._parse_columns(table_idx, original_table_name)
            table["columns"] = columns
            table["primary_keys"] = primary_keys
            table["foreign_keys"] = self._parse_foreign_keys(table_idx, original_table_name)
            table["value_examples"] = self._parse_value_examples(original_table_name)
            self.ir["tables"].append(table)

    def _parse_columns(self, table_idx: int, table_name: str) -> list[dict[str, any]]:
        # connect to db
        db_path = self.db_dir / f"{self.db_id}.sqlite"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        column_types = {}
        column_null_able = {}
        column_is_primary_key = {}
        column_contain_null = {}
        cursor.execute(f'PRAGMA table_info("{table_name}")')

        # get (type | nullable | primary key)
        for col in cursor.fetchall():
            column_name = col[1]
            column_type = col[2]
            column_types[column_name] = column_type
            column_null_able[column_name] = col[3]
            if col[5] >= 1:
                column_is_primary_key[column_name] = True

            # Check if the column contains NULL
            sql = f'SELECT COUNT(*) FROM "{table_name}" WHERE "{column_name}" IS NULL'
            cursor.execute(sql)
            column_contain_null[column_name] = cursor.fetchone()[0] > 0

        # get UNIQUE columns
        column_unique = {}
        cursor.execute(f'PRAGMA index_list("{table_name}")')
        for index in cursor.fetchall():
            if index[2] == 1:
                for column in cursor.execute(f'PRAGMA index_info("{index[1]}")').fetchall():
                    column_name = column[2]
                    column_unique[column_name] = True
        conn.close()

        # Prepare description for BIRD
        if self.bench == "bird_train" or self.bench == "bird_dev":
            desc_path = self.db_dir / "database_description" / f"{table_name}.csv"
            if not desc_path.exists():
                raise ValueError(f"Description file {desc_path} does not exist.")
            with open(desc_path, "rb") as f:
                result = chardet.detect(f.read())
            desc_df = pd.read_csv(desc_path, encoding=result["encoding"])

        # make column list and primary key list
        columns = []
        primary_keys = []

        for col_idx, (col_table_idx, col_name) in enumerate(self.db_json["column_names"]):
            if col_table_idx == table_idx:
                original_col_name = self.db_json["column_names_original"][col_idx][1]

                # Type
                col_type = column_types[original_col_name].upper()
                if "INT" in col_type:
                    col_type = "INT"

                # Nullable and UNIQUE
                column_def = f'    "{original_col_name}" {col_type}'
                if (column_null_able[original_col_name]) and (original_col_name not in column_is_primary_key):
                    column_def += " NOT NULL"
                if (original_col_name in column_unique) and (original_col_name not in column_is_primary_key):
                    column_def += " UNIQUE"

                not_primary_and_nullable = bool(
                    (original_col_name not in column_is_primary_key) and (not column_null_able[original_col_name])
                )

                # Column comment and description
                if self.bench == "spider":
                    if not string_equivalent(col_name, original_col_name):
                        column_def += f" -- {col_name}"
                else:
                    cmt = " -- "
                    if not string_equivalent(col_name, original_col_name):
                        cmt += col_name

                    # Add description
                    if self.bench != "bird_train" and self.bench != "bird_dev":
                        col_desc = None
                    else:
                        desc_line = desc_df[desc_df["original_column_name"].str.strip() == original_col_name]
                        if not desc_line.empty:
                            col_desc = desc_line["column_description"].values[0]

                    if (
                        col_desc
                        and not pd.isna(col_desc)
                        and not string_equivalent(col_name, col_desc)
                        and not string_equivalent(original_col_name, col_desc)
                    ):
                        cmt += f"({col_desc})"

                    if cmt != " -- ":
                        column_def += cmt

                if original_col_name in column_is_primary_key:
                    primary_keys.append(len(columns))

                columns.append(
                    {
                        "col_idx": len(columns),
                        "col_name": original_col_name,
                        "col_defination": column_def,
                        "not_primary_and_nullable": not_primary_and_nullable,
                        "contain_null": column_contain_null[original_col_name],
                    }
                )

        return columns, primary_keys

    def _parse_foreign_keys(self, table_idx: int, table_name: str) -> list[dict[str, any]]:
        """Get foreign key information for a specific table."""
        foreign_keys = []
        db = self.db_json
        for fk in db["foreign_keys"]:
            if db["column_names"][fk[0]][0] == table_idx:
                foreign_keys.append(
                    {
                        "table": f'"{table_name}"',
                        "column": f'"{db["column_names_original"][fk[0]][1]}"',
                        "referenced_table": db["table_names_original"][db["column_names"][fk[1]][0]],
                        "referenced_column": f'"{db["column_names_original"][fk[1]][1]}"',
                    }
                )
        return foreign_keys

    def _parse_value_examples(self, table_name: str) -> dict[str, any]:
        db = SQLDatabase.from_uri(f"sqlite:///{os.path.join(self.db_dir, self.db_id + '.sqlite')}")

        # Column names
        column_info = db._execute(f"SELECT name FROM pragma_table_info('{table_name}');")
        column_names = [col["name"] for col in column_info]

        # For every column, execute a query to get a sample of data
        column_data = {}
        for col_name in column_names:
            result = db._execute(f'SELECT DISTINCT "{col_name}" FROM "{table_name}" LIMIT 3')
            if not result:
                continue

            column_values = []
            for row in result:
                column_values.append(row[col_name])

            # Not too long
            column_values = [value for value in column_values if len(str(value)) < 80 and value]
            if not column_values:
                continue
            column_data[col_name] = column_values

        return column_data

    def to_dict(self):
        return self.ir


if __name__ == "__main__":
    SPIDER_BASE = Path().cwd().parent / "dataset" / "spider"
    BIRD_TRAIN_BASE = Path().cwd().parent / "dataset" / "bird-train"
    BIRD_DEV_BASE = Path().cwd().parent / "dataset" / "bird-dev"

    # benchs = ["bird_train"]
    # benchs = ["bird_dev"]
    # benchs = ["spider"]

    benchs = [
        # "DB_schema_synonym",
        # "DB_schema_abbreviation",
        "DB_DBcontent_equivalence",
    ]

    for bench in benchs:
        # if bench == "spider":
        #     DB_BASE = SPIDER_BASE / "database"
        #     TABLE_PATH = SPIDER_BASE / "tables.json"
        # elif bench == "bird_train":
        #     DB_BASE = BIRD_TRAIN_BASE / "train_databases"
        #     TABLE_PATH = BIRD_TRAIN_BASE / "train_tables.json"
        # elif bench == "bird_dev":
        #     DB_BASE = BIRD_DEV_BASE / "dev_databases"
        #     TABLE_PATH = BIRD_DEV_BASE / "dev_tables.json"

        if bench == "DB_schema_synonym":
            DB_BASE = Path("/Users/gobegobe/diagnostic-robustness-text-to-sql/data/DB_schema_synonym/database_post_perturbation")
            TABLE_PATH = Path("/Users/gobegobe/diagnostic-robustness-text-to-sql/data/DB_schema_synonym/tables_post_perturbation.json")
        elif bench == "DB_schema_abbreviation":
            DB_BASE = Path("/Users/gobegobe/diagnostic-robustness-text-to-sql/data/DB_schema_abbreviation/database_post_perturbation")
            TABLE_PATH = Path("/Users/gobegobe/diagnostic-robustness-text-to-sql/data/DB_schema_abbreviation/tables_post_perturbation.json")
        elif bench == "DB_DBcontent_equivalence":
            DB_BASE = Path("/Users/gobegobe/diagnostic-robustness-text-to-sql/data/DB_DBcontent_equivalence/database_post_perturbation")
            TABLE_PATH = Path(
                "/Users/gobegobe/diagnostic-robustness-text-to-sql/data/DB_DBcontent_equivalence/tables_post_perturbation.json"
            )

        with open(TABLE_PATH) as f:
            tables = json.load(f)

        db_ids = sorted([table["db_id"] for table in tables])

        # if bench == "spider":
        #     db_ids = [db_id for db_id in db_ids if db_id != "scholar" and db_id != "formula_1"]
        #     db_ids = ["scholar"]
        ir_set = []

        for db_id in db_ids:
            db_dir = DB_BASE / db_id
            db_json = [table for table in tables if table["db_id"] == db_id][0]
            print(f"### {db_id}")
            ir = Schema2IR(db_id, db_dir, db_json, bench).to_dict()

            ir_set.append(ir)

        with open(f"./ir_2025/{bench}_ir.json", "w") as f:
            json.dump(ir_set, f, indent=2)
