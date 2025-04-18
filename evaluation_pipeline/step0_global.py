import copy
import json
import pickle
import random
import re
import os
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

SCHEMA_LINK_TEMPLATE = """Given the following SQL tables, your job is to determine the column names and table that the question is referring to. \
The selected tables and columns should be sufficient to construct a SQL query that answers the question. \
Output in json format to indicate which table uses which column.
{schema}{hint}
### Question: {question}
"""

SCHEMA_LINK_RESPONSE_TEMPLATE = "### Chosen tables and columns:\n"

def extract_json(s: str) -> str:
    # extract from ```json and ```
    pattern = r"```json\s*(.*?)\s*```"
    match = re.search(pattern, s, re.DOTALL)
    return match.group(1) if match else s

DEFAULT_MODEL = "Alibaba-NLP/gte-large-en-v1.5"

def model_generate(prompt: str, model: AutoModelForCausalLM, tokenizer: AutoTokenizer) -> str:
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": SCHEMA_LINK_RESPONSE_TEMPLATE + "```json\n"},
    ]
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, continue_final_message=True)
    device = model.device
    input_ids = tokenizer.encode(formatted_prompt, return_tensors="pt", padding=False).to(device)

    output = model.generate(
        input_ids,
        max_new_tokens=512,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    response = tokenizer.decode(output[0][input_ids.shape[1] :], skip_special_tokens=True)
    return response

##########################################################################################


@dataclass
class CustomConfig:
    global_model_path: str = field()

parser = TrlParser((CustomConfig, ModelConfig))
(custom_config, model_config) = parser.parse_args_and_config()
custom_config: CustomConfig
model_config: ModelConfig

# Index Directory
INDEX_DIR = "/home/ubuntu/indexes/bird-dev"
IR_PATH = Path("/home/ubuntu/BIRD/Llama3.1-pipeline/utils/bird_dev_ir.json")

question_dir = Path("/home/ubuntu/dataset/bird_dev_dynamic_link.json")
input_questions = json.load(open(question_dir, "r"))

for i, dp in enumerate(input_questions):
    dp["number"] = i

print(f"Dr-Spider Data Loaded, length = {len(input_questions)}")
random.seed(42)
random.shuffle(input_questions)

# Make storage path
storage_path = Path.cwd() / "pipeline"
storage_path.mkdir(exist_ok=True)

# Load Models: global model & embedding model
global_model = AutoModelForCausalLM.from_pretrained(
    custom_config.global_model_path,
    attn_implementation=model_config.attn_implementation,
    torch_dtype=model_config.torch_dtype,
)


tokenizer = AutoTokenizer.from_pretrained(custom_config.global_model_path)
tokenizer.padding_side = "right"

distributed_state = PartialState()
global_model.to(distributed_state.device)

CACHE_FOLDER = "/home/ubuntu/.cache/huggingface/hub"
emb_model = SentenceTransformer(DEFAULT_MODEL, trust_remote_code=True, cache_folder=CACHE_FOLDER, local_files_only=True)
emb_model.to(distributed_state.device)


with distributed_state.split_between_processes(input_questions) as input_data:
    print(f"device = {distributed_state.device}. Starting to Global Schema Linking!")

    # Load Index
    index_dict = {}
    db_ids = [db_id.stem for db_id in Path(INDEX_DIR).glob("*.pkl")]
    for db_id in db_ids:
        index_path = Path(INDEX_DIR) / f"{db_id}.pkl"
        index = pickle.load(open(index_path, "rb"))
        index_dict[db_id] = index

    # Load IR-Set
    ir_set = json.load(open(IR_PATH, "r"))

    subnode_results = []
    pbar = tqdm(input_data, total=len(input_data), position=distributed_state.process_index)
    for datapoint in pbar:
        db_id = datapoint["db_id"]
        question = datapoint["question"]
        evidence = datapoint["evidence"]
        if not datapoint["evidence"].strip():
            hint = ""
        else:
            hint = f'\n### Hint: {datapoint["evidence"]}'
        # if len(evidence.strip()) >= 5:
        #    question = question + "\n" + "hint: " + evidence


        index = index_dict[db_id]
        ir = [ir for ir in ir_set if ir["db_id"] == db_id][0]
        converter = IR2Schema(ir, None, index, question, emb_model, False)
        dynamic_schema = converter.to_schema()

        prompt = SCHEMA_LINK_TEMPLATE.format(schema=dynamic_schema[0], question=datapoint["question"], hint=hint)
        link_result = "```json\n" + model_generate(prompt, global_model, tokenizer)

        subnode_results.append(
            {
                "number": datapoint["number"],
                "db_id": datapoint["db_id"],
                "question": datapoint["question"],
                "evidence": datapoint["evidence"],
                "query": datapoint["query"],
                "link_pred": link_result,
            }
        )

distributed_state.wait_for_everyone()
gathered_results = gather_object(subnode_results)

if distributed_state.is_main_process:

    persistent_results = []

    for datapoint in tqdm(gathered_results, total=len(gathered_results)):
        db_id = datapoint["db_id"]
        question = datapoint["question"]
        evidence = datapoint["evidence"]
        index = index_dict[db_id]
        ir = [ir for ir in ir_set if ir["db_id"] == db_id][0]

        try:
            link_json = extract_json(datapoint["link_pred"])
            link_pred = json.loads(link_json)

            # dynamic schema, usable for later model
            converter = IR2Schema(ir, link_pred, index, question, emb_model, False)
            dynamic_schema = converter.to_schema()
            persistent_results.append(
                {
                    "number": datapoint["number"],
                    "db_id": db_id,
                    "question": question,
                    "evidence": evidence,
                    "query": datapoint["query"],
                    "schema": dynamic_schema[0],
                    "global_link": link_pred,
                }
            )
        except Exception as e:
            print("Exception: ", e, link_pred)
            converter = IR2Schema(ir, None, index, question, emb_model, False)
            dynamic_schema = converter.to_schema()
            persistent_results.append(
                {
                    "number": datapoint["number"],
                    "db_id": db_id,
                    "question": question,
                    "evidence": evidence,
                    "query": datapoint["query"],
                    "schema": dynamic_schema[0],
                    "global_link": None,
                }
            )

    os.makedirs("pipeline", exist_ok=True)

    write_path = "pipeline/0_global_link.json"
    with open(write_path, "w") as f:
        json.dump(persistent_results, f, indent=2)