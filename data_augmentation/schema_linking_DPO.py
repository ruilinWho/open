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
        if all([len(t["loss_columns"]) == 0 for t in schema_links]):
            return None

    for link in schema_links:
        table = link["table_name"]
        columns = link[column_choice]
        if len(columns) == 0:
            continue
        result[table] = columns

    return "```json\n" + json.dumps(result) + "\n```"


benches = ["spider", "bird"]


def synthesize_dpo_data(ir: dict[str, any], datapoint: dict[str, any]) -> dict[str, any]:
    """
    To build the DPO data, we use each schema-link gronud truth.
    For each datapoint, we need to generate preference data: {base, result_loss}
        - base: link["table_name"] and link["related_columns"]
        - result_loss:
            - base - random columns(tables are perfect)
            - base - random tables
    """

    schema_link: list = datapoint["schema_link"]
    tuned_schema_link = []

    deletable_columns = []
    # record all foreign key relationship, if we delete a table, we need to delete all the foreign key pointing to it
    foreign_key_records: dict[str, list[tuple[str, str]]] = {}

    for table in schema_link:
        table_name = table["table_name"]
        related_columns = table["related_columns"]

        # Get all columns of this table (related and unrelated)
        # NOTE: what we "add" to synthetic data should not be Primary key or Foreign key, so check it
        table_ir = [t for t in ir["tables"] if t["table_name"] == table_name][0]
        columns: list[dict] = [col for col in table_ir["columns"]]
        pk_indexes: list[int] = table_ir["primary_keys"]
        fks: list[dict] = table_ir["foreign_keys"]

        for fk in fks:
            from_table = table_name
            from_col = fk["column"].strip('"')
            to_table = fk["referenced_table"].strip('"')
            foreign_key_records.setdefault(to_table, []).append((from_table, from_col))

        # Remove columns that are primary keys or foreign keys
        for column in columns:
            col_idx = column["col_idx"]
            col_name = column["col_name"]

            if (col_idx not in pk_indexes) and any([col_name == related_name for related_name in related_columns]):
                deletable_columns.append({"table_name": table_name, "column_name": column["col_name"]})

        # Now we have the added columns, build the win data
        tuned_schema_link.append(
            {
                "table_name": table_name,
                "related_columns": related_columns,
                "win_columns": copy.deepcopy(related_columns),
                "loss_columns": copy.deepcopy(related_columns),  # copy here, delete later
            }
        )

    # Now for every datapoint, we remove 1~2 columns to build the loss data
    # 50% remove one column; 50% remove two columns

    messed_results = []
    tuned_schema_link_backup = copy.deepcopy(tuned_schema_link)

    # Add K columns bad case
    for _ in range(3):
        tuned_schema_link = copy.deepcopy(tuned_schema_link_backup)

        if len(deletable_columns) < 1:
            break
        elif len(deletable_columns) == 1:
            to_deletes = [random.choice(deletable_columns)]
        else:
            if random.random() < 0.7:
                to_deletes = [random.choice(deletable_columns)]
            else:
                to_deletes = random.sample(deletable_columns, 2)

        for deleting in to_deletes:
            for table in tuned_schema_link:
                if table["table_name"] == deleting["table_name"]:
                    table["loss_columns"].remove(deleting["column_name"])
                    break

        messed_results.append(
            {
                "mess_type": "column",
                "db_id": datapoint["db_id"],
                "question": datapoint["question"],
                "evidence": datapoint["evidence"],
                "schema": datapoint["schema"],
                "schema_link": tuned_schema_link,
            }
        )

        # Remove current to_deletes
        for deleting in to_deletes:
            deletable_columns.remove(deleting)

    if len(tuned_schema_link) == 1:
        return messed_results

    # (Table Lost DPO data)
    for to_delete_table in tuned_schema_link_backup:
        tuned_schema_link = copy.deepcopy(tuned_schema_link_backup)

        for table in tuned_schema_link:
            if table["table_name"] == to_delete_table["table_name"]:
                # Set loss_columns to []
                table["loss_columns"] = []
                break

        to_delete_table_name = to_delete_table["table_name"]
        if to_delete_table_name in foreign_key_records:
            to_delete_fks = foreign_key_records[to_delete_table_name]
            for fk in to_delete_fks:
                for table in tuned_schema_link:
                    if table["table_name"] == fk[0] and fk[1] in table["loss_columns"]:
                        table["loss_columns"].remove(fk[1])

        # Now we have the final data
        messed_results.append(
            {
                "mess_type": "table",
                "db_id": datapoint["db_id"],
                "question": datapoint["question"],
                "evidence": datapoint["evidence"],
                "schema": datapoint["schema"],
                "schema_link": tuned_schema_link,
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

    # while True:
    #     tmp_db_ids = [db_id for db_id in all_db_ids if (db_id != "formula_1" and db_id != "scholar")]

    #     chosen_db_ids = random.sample(tmp_db_ids, int(len(all_db_ids) * 0.3))

    #     chosen_question_count = 0
    #     for db_id in chosen_db_ids:
    #         chosen_question_count += len([q for q in DATASET if q["db_id"] == db_id])
    #     question_percent = chosen_question_count / len(schema_links)
    #     if 0.24 < question_percent < 0.26:
    #         print("Total question count: ", chosen_question_count)
    #         print("DPO percent: ", question_percent, "| SFT percent: ", 1 - question_percent, "\n")
    #         break
    #     else:
    #         pass
    #         # print("Question percent: ", question_percent)
    chosen_db_ids = []
    ########## Build DPO data on specific db_id ##########

    dpo_data = []
    for db_id in chosen_db_ids:
        try:
            ir = [ir for ir in ir_set if ir["db_id"] == db_id][0]
        except:
            print("Problem Occured: ", db_id)
        datapoints = [sl for sl in schema_links if sl["db_id"] == db_id]

        for datapoint in datapoints:
            # XXX: Considering this is a probabilistic algorithm, we do it multiple times
            data = synthesize_dpo_data(ir, datapoint)
            assert type(data) is list or data is None
            if data is not None:
                for datum in data:
                    datum["bench"] = bench
                    datum["train_type"] = "DPO"
                    dpo_data.append(datum)

    print("Bench = ", bench, "| Total dpo_data length: ", len(dpo_data))

    ########## Format the data ##########

    train_data = []
    for datum in dpo_data:
        win_sl = linearize(datum["schema_link"], "win_columns")
        loss_sl = linearize(datum["schema_link"], "loss_columns")

        # If every table is empty, we skip this datapoint
        if loss_sl is None:
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
    # del datum #FIXME
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

    OUTPUT_DIR = "./only_SFT"
    if not Path(OUTPUT_DIR).exists():
        Path(OUTPUT_DIR).mkdir()

    with open(f"{OUTPUT_DIR}/{bench}_mixed.json", "w") as f:
        json.dump(mixed_data, f, indent=2)

    return mixed_data


spider_mix = build_dpo_data("spider")
bird_mix = build_dpo_data("bird")


all_mix = spider_mix + bird_mix
with open("./all_mixed.json", "w") as f:
    json.dump(all_mix, f, indent=2)
    print("Mixed data length: ", len(all_mix))
    print("DPO data number = ", len([d for d in all_mix if d["train_type"] == "DPO"]))
    print("SFT data number = ", len([d for d in all_mix if d["train_type"] == "SFT"]))
