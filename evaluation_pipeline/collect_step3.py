import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from tqdm import tqdm
from trl import ModelConfig, TrlParser

if __name__ == "__main__":
    storage_path = Path.cwd() / "pipeline"

    # Load all the json file in storage_path with "GPU" in filename
    file_names = [filename for filename in storage_path.iterdir() if "res_" in filename.name]
    all_results = []
    for file_name in file_names:
        with open(file_name, "r") as f:
            results = json.load(f)
            all_results += results

    all_results.sort(key=lambda x: x["number"])

    with open("Pred_selected.txt", "w") as f:
        for item in all_results:
            f.write(item["selected_sql"] + "\n")

    correct_sum = sum([r["selected_correct"] for r in all_results])
    print(f"EX accuracy = {correct_sum / len(all_results)}")

    storage_path = Path.cwd() / "pipeline"
    with open(storage_path / "4_selected_output.json", "w") as f:
        json.dump(all_results, f, indent=2)

    ############################################################################################
    # Use Test-suite
    template = "python3 /home/ubuntu/Dr_Spider/dr-spider-test/evaluation.py --gold {gold_path} --pred {pred_path} --db {db_path} --etype exec --keep_distinct --table {table_path}"

    gold_path = "/home/ubuntu/database_files/spider/dev_gold.sql"
    db_path = "/home/ubuntu/database_files/spider/database"
    table_path = "/home/ubuntu/database_files/spider/tables.json"

    ## First Use Pred.txt
    name = "Pred_selected"
    pred_path = f"{name}.txt"
    output_file = f"{name}-test-suite.txt"

    # 格式化命令
    command = template.format(gold_path=gold_path, pred_path=pred_path, db_path=db_path, table_path=table_path)

    # 打开输出文件并执行命令
    with open(output_file, "w") as f:
        process = subprocess.run(command, shell=True, stdout=f, stderr=subprocess.STDOUT)
