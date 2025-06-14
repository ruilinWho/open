#######################################################################
# This script is used to build the DPO data for the schema linking task.
# Note: Some paths have been anonymized to avoid identity disclosure.
#######################################################################
import copy
import json
import random
from pathlib import Path


def linearize(schema_links: list[dict[str, any]], column_choice: str) -> str:
    result = {}

    if column_choice == "loss_columns":
        # This check ensures we don't produce a loss example that is identical to the win example.
        if all([len(t["loss_columns"]) == 0 for t in schema_links]):
            return None

    for link in schema_links:
        table = link["table_name"]
        columns = link[column_choice]
        if len(columns) == 0:
            continue
        result[table] = columns

    # If all tables ended up with no columns, return None
    if not result:
        return None

    return "```json\n" + json.dumps(result) + "\n```"


benches = ["spider", "bird"]


def synthesize_dpo_data(ir: dict[str, any], datapoint: dict[str, any]) -> dict[str, any]:
    """
    To build the DPO data, we use each schema-link ground truth.
    For each datapoint, we generate preference data (chosen/rejected pairs).

    - "yes examples" (chosen): Use the ground-truth tables and columns.
    - "no examples" (rejected): Are generated by following a specific corruption strategy:
        1. Remove 1-2 columns (50% chance for 1, 50% for 2).
        2. Remove 1-2 tables (70% chance for 1, 30% for 2).
    This entire process is repeated twice for each datapoint to generate multiple examples.
    """
    schema_link: list = datapoint["schema_link"]

    base_tuned_schema_link = []

    deletable_columns = []
    # Record all foreign key relationships to handle cascading deletes.
    foreign_key_records: dict[str, list[tuple[str, str]]] = {}

    for table in schema_link:
        table_name = table["table_name"]
        related_columns = table["related_columns"]

        # Find the full table schema from the Intermediate Representation (IR).
        table_ir = next((t for t in ir["tables"] if t["table_name"] == table_name), None)
        if table_ir is None:
            continue  # Should not happen in practice

        pk_indexes: list[int] = table_ir.get("primary_keys", [])
        fks: list[dict] = table_ir.get("foreign_keys", [])

        # Record foreign keys originating from this table.
        for fk in fks:
            from_table = table_name
            from_col = fk["column"].strip('"')
            to_table = fk["referenced_table"].strip('"')
            foreign_key_records.setdefault(to_table, []).append((from_table, from_col))

        # Identify columns that are safe to remove (part of the ground truth but not a PK).
        for column in table_ir["columns"]:
            col_idx = column["col_idx"]
            col_name = column["col_name"]

            # A column is deletable if it's in the ground truth and not a primary key.
            # (Note: Foreign keys can be deleted, their relationships are handled later).
            if (col_idx not in pk_indexes) and (col_name in related_columns):
                deletable_columns.append({"table_name": table_name, "column_name": col_name})

        # Build the base structure with ground truth for 'win' and a copy for 'loss'.
        base_tuned_schema_link.append(
            {
                "table_name": table_name,
                "related_columns": related_columns,  # Ground truth (for reference)
                "win_columns": copy.deepcopy(related_columns),  # The "yes" example
                "loss_columns": copy.deepcopy(related_columns),  # The "no" example (to be corrupted)
            }
        )

    messed_results = []

    # Per the strategy, repeat the synthesis process twice for each datapoint.
    for _ in range(2):
        # --- Strategy 1: Create a "no example" by removing columns ---
        if deletable_columns:
            temp_schema_link_col_loss = copy.deepcopy(base_tuned_schema_link)

            # Determine how many columns to remove: 1 (50% prob) or 2 (50% prob).
            num_cols_to_remove = 2 if random.random() < 0.5 and len(deletable_columns) >= 2 else 1

            cols_to_delete = random.sample(deletable_columns, num_cols_to_remove)

            for col_to_delete in cols_to_delete:
                for table in temp_schema_link_col_loss:
                    if table["table_name"] == col_to_delete["table_name"] and col_to_delete["column_name"] in table["loss_columns"]:
                        table["loss_columns"].remove(col_to_delete["column_name"])
                        break

            messed_results.append(
                {
                    "mess_type": "column",
                    "db_id": datapoint["db_id"],
                    "question": datapoint["question"],
                    "evidence": datapoint["evidence"],
                    "schema": datapoint["schema"],
                    "schema_link": temp_schema_link_col_loss,
                }
            )

        if len(base_tuned_schema_link) > 1:
            temp_schema_link_tbl_loss = copy.deepcopy(base_tuned_schema_link)

            # Determine how many tables to remove: 1 (70% prob) or 2 (30% prob).
            num_tables_to_remove = 2 if random.random() < 0.3 and len(base_tuned_schema_link) >= 2 else 1

            tables_to_delete = random.sample(base_tuned_schema_link, num_tables_to_remove)

            for table_to_delete in tables_to_delete:
                table_name_to_delete = table_to_delete["table_name"]

                # Main removal: Set the table's own loss_columns to empty.
                for table in temp_schema_link_tbl_loss:
                    if table["table_name"] == table_name_to_delete:
                        table["loss_columns"] = []
                        break

                # Handle FKs: Remove columns from other tables that reference the deleted table.
                if table_name_to_delete in foreign_key_records:
                    for from_table, from_col in foreign_key_records[table_name_to_delete]:
                        for table in temp_schema_link_tbl_loss:
                            if table["table_name"] == from_table and from_col in table["loss_columns"]:
                                table["loss_columns"].remove(from_col)

            messed_results.append(
                {
                    "mess_type": "table",
                    "db_id": datapoint["db_id"],
                    "question": datapoint["question"],
                    "evidence": datapoint["evidence"],
                    "schema": datapoint["schema"],
                    "schema_link": temp_schema_link_tbl_loss,
                }
            )

    return messed_results


def build_dpo_data(bench: str):
    assert bench in benches

    if bench == "bird":
        ir_path = "<Anonymous>"
        schema_link_path = "<Anonymous>"
        DATASET = json.load(open("<Anonymous>"))
    else:
        ir_path = "<Anonymous>"
        schema_link_path = "<Anonymous>"
        DATASET = json.load(open("<Anonymous>"))

    ir_set = json.load(open(ir_path))
    schema_links = json.load(open(schema_link_path))
    print("IR length: ", len(ir_set))
    print("Schema length: ", len(schema_links))

    ########## Choose db_ids to build the DPO data ##########
    all_db_ids = set([sl["db_id"] for sl in schema_links])

    chosen_db_ids = []

    ########## Build DPO data on specific db_id ##########

    dpo_data = []

    db_to_ir_map = {ir["db_id"]: ir for ir in ir_set}

    for datapoint in schema_links:
        db_id = datapoint["db_id"]
        # Skip if this db_id is not in the chosen set (if chosen_db_ids is used)
        if chosen_db_ids and db_id not in chosen_db_ids:
            continue

        ir = db_to_ir_map.get(db_id)
        if not ir:
            print(f"Problem Occurred: IR not found for db_id {db_id}")
            continue

        # Synthesize DPO examples for the current data point.
        data = synthesize_dpo_data(ir, datapoint)

        if data:
            for datum in data:
                datum["bench"] = bench
                datum["train_type"] = "DPO"
                dpo_data.append(datum)

    print(f"Bench = {bench} | Total dpo_data generated: {len(dpo_data)}")

    ########## Format the data ##########

    train_data = []
    for datum in dpo_data:
        win_sl = linearize(datum["schema_link"], "win_columns")
        loss_sl = linearize(datum["schema_link"], "loss_columns")

        # If the loss example is empty or identical to the win example, skip it.
        if loss_sl is None or win_sl == loss_sl:
            continue

        train_data.append(
            {
                "bench": bench,
                "train_type": "DPO",
                "mess_type": datum["mess_type"],
                "db_id": datum["db_id"],
                "schema": datum["schema"],
                "question": datum["question"],
                "evidence": datum["evidence"],
                "win_sl": win_sl,
                "loss_sl": loss_sl,
            }
        )

    ########## SFT data ##########
    sft_data = []
    sft_db_ids = all_db_ids - set(chosen_db_ids)

    for sl in schema_links:
        if sl["db_id"] not in sft_db_ids:
            continue

        sft_data.append(
            {
                "bench": bench,
                "train_type": "SFT",
                "db_id": sl["db_id"],
                "schema": sl["schema"],
                "question": sl["question"],
                "evidence": sl["evidence"],
                "standard_sl": linearize(sl["schema_link"], "related_columns"),
            }
        )

    mixed_data = train_data + sft_data

    # Changed the output directory name to reflect the content.
    OUTPUT_DIR = "./mixed_DPO_SFT_data"
    if not Path(OUTPUT_DIR).exists():
        Path(OUTPUT_DIR).mkdir(parents=True)

    with open(f"{OUTPUT_DIR}/{bench}_mixed.json", "w") as f:
        json.dump(mixed_data, f, indent=2)

    return mixed_data


spider_mix = build_dpo_data("spider")
bird_mix = build_dpo_data("bird")


all_mix = spider_mix + bird_mix
with open("./all_mixed.json", "w") as f:
    json.dump(all_mix, f, indent=2)
    print("\n--- Final Summary ---")
    print(f"Total mixed data length: {len(all_mix)}")
    print(f"DPO data number = {len([d for d in all_mix if d['train_type'] == 'DPO'])}")
    print(f"SFT data number = {len([d for d in all_mix if d['train_type'] == 'SFT'])}")
