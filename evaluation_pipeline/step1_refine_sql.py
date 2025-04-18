import copy
import json
import pickle
import random
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from accelerate import PartialState
from accelerate.utils import gather_object
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import ModelConfig, SFTConfig, TrlParser
from utils.embed_values import ColumnVectorIndex
from utils.ir_to_schema import IR2Schema

LOCAL_CLASSIFICATION_TEMPLATE = """Given a database table, a question, and a column in the table, your task is to determine whether the column is useful to generate a SQL query for answering the question.
Note: Some example values of the column are shown to you, if any example values match the question, the column is likely to be useful.

[Table schema]
{table_schema}
[Column to check]
column name: {column_name}
{column_value_examples}
[Question]
{question}

-- Return one word: True or False.
"""

LOCAL_RESPONSE_TEMPLATE = "[Judgement]\n"
DEFAULT_MODEL = "Alibaba-NLP/gte-large-en-v1.5"
INDEX_DIR = "/home/ubuntu/indexes/bird-dev"


def model_generate_single(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_tokens: int = 16,
    response_template: str = LOCAL_RESPONSE_TEMPLATE,
) -> str:
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response_template},
    ]
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, continue_final_message=True)
    input_ids = tokenizer.encode(formatted_prompt, return_tensors="pt", padding=False).to(model.device)

    output = model.generate(
        input_ids,
        max_new_tokens=max_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    response = tokenizer.decode(output[0][input_ids.shape[1] :], skip_special_tokens=True)
    return response


##########################################################################################


@dataclass
class CustomConfig:
    refine_model_path: str = field()
    persistent_input_path: str = field()


parser = TrlParser((CustomConfig, ModelConfig))
(custom_config, model_config) = parser.parse_args_and_config()
custom_config: CustomConfig
model_config: ModelConfig

# Load Validation Data path
persistent_input = json.load(open(custom_config.persistent_input_path, "r"))

for i, dp in enumerate(persistent_input):
    dp["number"] = i

print(f"Validation Data Loaded, length = {len(persistent_input)}")
random.seed(42)
random.shuffle(persistent_input)

# Make storage path
storage_path = Path.cwd() / "pipeline"
storage_path.mkdir(exist_ok=True)

# Load Models: local(refine) model & embedding model
local_model = AutoModelForCausalLM.from_pretrained(
    custom_config.refine_model_path,
    attn_implementation=model_config.attn_implementation,
    torch_dtype=model_config.torch_dtype,
)

CACHE_FOLDER = "/home/ubuntu/.cache/huggingface/hub"
emb_model = SentenceTransformer(DEFAULT_MODEL, trust_remote_code=True, cache_folder=CACHE_FOLDER, local_files_only=True)

tokenizer = AutoTokenizer.from_pretrained(custom_config.refine_model_path)
tokenizer.padding_side = "right"

distributed_state = PartialState()
local_model.to(distributed_state.device)
emb_model.to(distributed_state.device)


with distributed_state.split_between_processes(persistent_input) as persistent_data:
    print(f"device = {distributed_state.device}. Starting to refine!")

    # Load Index
    index_dict = {}
    db_ids = [db_id.stem for db_id in Path(INDEX_DIR).glob("*.pkl")]
    for db_id in db_ids:
        index_path = Path(INDEX_DIR) / f"{db_id}.pkl"
        index = pickle.load(open(index_path, "rb"))
        index_dict[db_id] = index

    # Load IR-Set
    ir_set = json.load(open("./utils/bird_dev_ir.json", "r"))

    subnode_results = []
    pbar = tqdm(persistent_data, total=len(persistent_data), position=distributed_state.process_index)
    for datapoint in pbar:
        db_id = datapoint["db_id"]
        index = index_dict[db_id]
        ir = [ir for ir in ir_set if ir["db_id"] == db_id][0]

        global_link = datapoint["global_link"]
        if global_link == None:
            global_link = []

        full_link = copy.deepcopy(global_link)
        question = datapoint["question"]
        evidence = datapoint["evidence"]
        if len(evidence.strip()) >= 5:
           question = question + "\n" + "hint: " + evidence

        try:
            converter_local = IR2Schema(ir, None, index, question, emb_model, implicit_linking=False)

            for table_name in global_link:
                if not any(table_name == t["table_name"] for t in ir["tables"]):
                    continue

                table_ir = [t for t in ir["tables"] if t["table_name"] == table_name][0]
                all_columns = [col["col_name"] for col in table_ir["columns"] if col["col_idx"] not in table_ir["primary_keys"]]
                for foreign_key in table_ir["foreign_keys"]:
                    fk_table_name = foreign_key["table"].strip('"')
                    fk_column_name = foreign_key["column"].strip('"')

                    if (fk_table_name == table_name) and (fk_column_name in all_columns):
                        all_columns.remove(fk_column_name)

                for column_name in all_columns:
                    if column_name in global_link[table_name]:
                        continue

                    table_schema, column_value_examples = converter_local.get_specific_schema(table_name, column_name)
                    prompt = LOCAL_CLASSIFICATION_TEMPLATE.format(
                        table_schema=table_schema,
                        column_name=column_name,
                        column_value_examples=column_value_examples,
                        question=question,
                    )
                    res = model_generate_single(local_model, tokenizer, prompt)
                    if "True" in res:
                        full_link[table_name].append(column_name)

            if global_link == []:
                converter = IR2Schema(ir, None, index, question, emb_model, implicit_linking=False)
            else:
                converter = IR2Schema(ir, full_link, index, question, emb_model, implicit_linking=False)
            refined_schema, _ = converter.to_schema()

            result = copy.deepcopy(datapoint)
            result["full_link"] = full_link
            result["refined_schema"] = refined_schema
            subnode_results.append(result)

        except Exception as e:
            print(f"db_id = {db_id}, table name = {table_name}")
            raise e
            print("Exception Caught(This should not happen in theory):", e)

distributed_state.wait_for_everyone()
gathered_results = gather_object(subnode_results)

if distributed_state.is_main_process:
    # Save Results
    with open(f"{storage_path}/1_refined_input.json", "w") as f:
        json.dump(gathered_results, f, indent=2)
    print("### Len of results = ", len(gathered_results))