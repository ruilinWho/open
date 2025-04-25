# OpenSQL

This repository contains source code of `OpenSQL` , specifically:

1. training scripts (Folder: training_scripts) 

2. data augmentation scripts (Folder: data_augmentation)

3. augmented data for training (Folder: data_for_training)

4. other utilize scripts for pre-processing (Folder: schema_utils)

`./requirements.txt` contains the necessary python module configurations.

Below we introduce the files in each folder and respective uses and meanings.

**Note: We added corresponding descriptions in each file.**

## 1. Training scripts

The training scripts and launch scripts are contained in `./training_scripts`.

1. `./training_scripts/schema-linking`  contains the training scripts for the global-local schema linking module
2. `./training_scripts/SQL-generation` contains the training scripts for the diversified SQL generator.
3. `./training_scripts/SQL-selection` contains the training scripts for the stepwise SQL selector.

**Note: For every scripts, we use `sh launch.sh`**to start the training. 

## 2. Data augmentation scripts

The scripts for our data augmentation are contained in `./data_augmentation`.

1. schema linking
   1. `./data_augmentation/schema_linking_SFT.py` : from raw SQL queries, extract referred tables and columns to construct data for global schema linking (SFT).
   2. `./data_augmentation/schema_linking_local.py`: from raw SQL queries and databases, construct binary classification training data for local schema linking.
   3. `./data_augmentation/schema_linking_DPO.py`: from global schema linking training data, randomly remove tables and columns to construct training data for schema-aware preference learning.
   4. `./data_augmentation/diversified_SQL_generation.py` : incorporated an rephrase LLM to generate multiple equivalent queries.
   5. `./data_augmentation/stepwise_selection.ipynb`: incorporate and annotation LLM to generate stepwise reasoning paths for SQL pairwise comparison.

## 3. Augmented Data

We provide augmented data for training three components of `OpenSQL`

1. `schema_linking_all.zip` contains the data for training schema linking models (including SFT training data and DPO training data)
2. `SQL_generation.zip` contains the data for training SQL generators. For each datapoint, we augment multiple equivalent SQL queries for training.
3. `SQL_selection.zip` contains the data for training the pairwise SQL selector. For each datapoint, we attached a detailed stepwise rationale into it to support training the stepwise selector.

## 4. Pre-processing scripts

### 4.1 IR (intermediate-representation)

We use IR to generate linearized schema representation. Each IR contains table descriptions, column descriptions and foreign-key relationships.

The IR files in our experiments are in `./schema-utils/intermediate-representation`.

We use two files to generate IR and decode IR:

- `./schema-utils/intermediate-representation/to_ir.py`: Given a database, produce an IR in a file. (Then this IR can be decoded to produce sub-schema representations)
- `./schema-utils/intermediate-representation/ir_to_Schema.py`: Given an IR and a python dictionary, decode an IR to output a linearized schema representation.

### 4.2 Embed Databases

We use `./schema_utils/embed/embed_values.py` to embed databases with `FAISS` and extract similar values in run-time.

class `ColumnVectorIndex` contains the code about extracting similar values.

function `embed_values_in_db` is used to generate vector indexes of the database.



### 5. evalutation pipeline

We formulate OpenSQL as sequential steps to generate SQL:

- (step0) global schema linking
- (step1) local schema linking (refine_sql)
- (step2) generate SQL candidates
- (step3) perform pairwise selection to choose the final output.

`./evaluation_pipeline/start_pipeline.sh` is used to start the evaluation pipeline.
