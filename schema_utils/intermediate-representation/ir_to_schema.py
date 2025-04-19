import json
from typing import Optional

from sentence_transformers import SentenceTransformer

from .embed_values import ColumnVectorIndex


class IR2Schema:
    def __init__(
        self,
        ir: dict,
        chosen: Optional[dict[str, list[str]]],
        tindex: Optional[dict[tuple[str, str], ColumnVectorIndex]],
        question: Optional[str],
        emb_model: Optional[SentenceTransformer] = None,
        implicit_linking: bool = False,
    ):
        self.ir = ir
        self.chosen = chosen
        self.tindex = tindex
        self.question = question
        self.emb_model = emb_model

        ### Implicit Linking
        if implicit_linking and (self.chosen is not None):
            # We add those columns that are similar to the question as chosen
            count = 0
            for table_name in self.chosen:
                try:
                    table = [t for t in self.ir["tables"] if t["table_name"] == table_name][0]
                except:
                    continue

                # Find Primary Keys and Foreign Keys, we won't implicit link these
                primary_key_ids = table["primary_keys"]
                foreign_key_names = [fk["column"].strip('"') for fk in table["foreign_keys"]]

                column_names_with_sim = []
                for column in table["columns"]:
                    if column["col_idx"] in primary_key_ids:
                        continue
                    if column["col_name"].strip('"') in foreign_key_names:
                        continue
                    similarity = self.emb_model.encode(column["col_name"]).dot(self.emb_model.encode(self.question))
                    column_names_with_sim.append((column["col_name"], similarity))

                column_names_with_sim.sort(key=lambda x: x[1], reverse=True)
                # Top 3
                column_names_with_sim = [name for name, _ in column_names_with_sim[:1]]
                for name in column_names_with_sim:
                    if name not in self.chosen[table_name]:
                        self.chosen[table_name].append(name)
                        count += 1
            print(f"Implicit Linking: {count} columns added")

    def is_table_chosen(self, table_name: str) -> bool:
        if not self.chosen:
            return True
        chosen_tabla_names = [name.lower() for name in self.chosen]
        return table_name.lower() in chosen_tabla_names

    def is_column_chosen(self, random_table_name: str, column_name: str) -> bool:
        if not self.chosen:
            return True
        if not self.is_table_chosen(random_table_name):
            return False

        for table_name in self.chosen:
            if table_name.lower() == random_table_name.lower():
                chosen_column_names = [name.lower() for name in self.chosen[table_name]]
                return column_name.lower() in chosen_column_names

    def to_schema(self) -> str:
        if self.chosen is not None:
            self.pred_link = self.chosen.copy()
        else:
            self.pred_link = None

        schema = f"-- Database name: {self.ir['db_id']}\n"
        schema += "-- Database schema:\n"
        for table in self.ir["tables"]:
            table_name = table["table_name"]
            if not self.is_table_chosen(table_name):
                continue
            schema += self._table_statement(table)
            schema += self._value_example(table)
        return schema, self.pred_link

    def get_specific_schema(self, table_name: str, column_name: str) -> tuple[str, str]:
        table = [t for t in self.ir["tables"] if t["table_name"] == table_name][0]
        return self._table_statement(table), self._value_example(table, column_name)

    def _table_statement(self, table: dict[str, any]) -> str:
        # Do the latter
        have_primary_keys = False
        have_foreign_keys = False
        statement_latter = ""

        # DO primary keys
        primary_keys = table["primary_keys"]
        if primary_keys:
            primary_names = ['"' + table["columns"][idx]["col_name"] + '"' for idx in primary_keys]
            primary_statement = "    PRIMARY KEY (" + ", ".join(primary_names) + ")"
            statement_latter += primary_statement
            have_primary_keys = True

        # DO foreign keys
        extra_foreign_keys = set()
        for i, fk in enumerate(table["foreign_keys"]):
            from_column = fk["column"]
            to_table = fk["referenced_table"]
            to_column = fk["referenced_column"]

            if (not self.is_table_chosen(to_table)) and (not self.is_column_chosen(table["table_name"], from_column)):
                continue
            extra_foreign_keys.add(from_column.strip('"'))

            if (have_primary_keys) or (have_foreign_keys):
                statement_latter += ",\n"
            statement_latter += f"    FOREIGN KEY ({from_column}) REFERENCES {to_table}({to_column})"
            have_foreign_keys = True
        statement_latter += "\n);\n"

        # Do the former
        former_statement = f'Table {table["table_name"]} ({table["table_comment"]}\n'
        chosen_columns = []

        for column in table["columns"]:
            # Schema linking Interface
            if self.chosen:
                valid = column["col_idx"] in primary_keys
                valid = valid or self.is_column_chosen(table["table_name"], column["col_name"])
                valid = valid or (column["col_name"] in extra_foreign_keys)

                if not valid:
                    continue
                elif self.pred_link is not None:
                    if column["col_name"] not in self.pred_link[table["table_name"]]:
                        self.pred_link[table["table_name"]].append(column["col_name"])
            chosen_columns.append(column)

        column_str = ""
        for i, column in enumerate(chosen_columns):
            col_desc = column["col_defination"]
            # remove \n in col_desc
            col_desc = col_desc.replace("\n", " ")
            if i < len(chosen_columns) - 1:
                if " --" in col_desc:
                    idx = col_desc.index(" --")
                    column_str += col_desc[:idx] + "," + col_desc[idx:] + "\n"
                else:
                    column_str += col_desc + ",\n"
            else:
                if have_primary_keys or have_foreign_keys:
                    if " --" in col_desc:
                        idx = col_desc.index(" --")
                        column_str += col_desc[:idx] + "," + col_desc[idx:] + "\n"
                    else:
                        column_str += col_desc + ",\n"
                else:
                    column_str += col_desc + "\n"
        former_statement += column_str

        return former_statement + statement_latter

    def _value_example(self, table: dict[str, any], specific_column: Optional[str] = None) -> str:
        if not specific_column:
            statement = "/* Value examples for each column:\n"
        else:
            statement = f"/* Value examples for column {specific_column}:\n"

        primary_keys = table["primary_keys"]
        have_example = False

        if not specific_column:
            column_choice = table["columns"]
        else:
            column_choice = [col for col in table["columns"] if col["col_name"] == specific_column]

        for column in column_choice:
            col_name = column["col_name"]
            col_idx = column["col_idx"]

            if self.chosen:
                valid = col_idx in primary_keys
                valid = valid or self.is_column_chosen(table["table_name"], col_name)
                if not valid:
                    continue

            # Add value example
            if col_name not in table["value_examples"]:
                continue

            # If this column is TETX-like and not chosen and empty, we can use the vector index to find similar values
            key = (table["table_name"], col_name)
            if (self.tindex is not None) and (key in self.tindex):
                column_vector_index = self.tindex[key]
                # similar_values = column_vector_index.get_similar_strings(col_name)
                value_example = column_vector_index.get_similar_strings(self.emb_model, self.question)
                value_example = [val for val in value_example if len(val) < 100]
                for val in table["value_examples"][col_name]:
                    if val not in value_example and len(value_example) < 4:
                        value_example.append(val)
            else:
                value_example = table["value_examples"][col_name]

            if type(value_example[0]) is str:
                value_example = ["'" + str(val).strip() + "'" for val in value_example]
            else:
                value_example = [str(val) for val in value_example]
            statement += f'"{col_name}": ' + ", ".join(value_example) + "\n"
            have_example = True

        if not have_example:
            return ""
        return statement + " */\n"


if __name__ == "__main__":
    with open("./schema_irs/spider_ir.json", "r") as f:
        ir_set = json.load(f)

    for ir in ir_set:
        chosen = None
        ir2schema = IR2Schema(ir, chosen)
        schema = ir2schema.to_schema()
        print(schema)
