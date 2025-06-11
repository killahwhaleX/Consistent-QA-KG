import os
from pathlib import Path
import json
from tqdm import tqdm
import re
import json
from typing import List, Dict
import argparse
import logging
from datetime import datetime

from neo4j import GraphDatabase
from neo4j_graphrag.retrievers import Text2CypherRetriever
from neo4j_graphrag.llm import LLMInterface, LLMResponse

import ollama

from utils import read_jsonl_file, read_json_file, load_prompts 

# Ollama wrapper for use with Text2CypherRetriever

class OllamaLLM(LLMInterface):

    def invoke(self, input: str) -> LLMResponse:
        response = ollama.chat(model=self.model_name, messages=[
          {
            'role': 'user',
            'content': input,
          },
        ])

        return LLMResponse(
            content=response["message"]["content"]
        )

    async def ainvoke(self, input: str) -> LLMResponse:
        return self.invoke(input)  # TODO: implement async with ollama.AsyncClient

def get_logger(session_dir):

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    log_path = os.path.join(session_dir, "log")
    path = Path(log_path)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)

    if path.is_dir():
        path = path /  f"log-{datetime.now().strftime('%Y_%m_%d-%H_%M_%S')}.txt"

    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

class ResultFile():
    def __init__(self, result_dir, relation):
        self.relation = relation

        path = Path(result_dir)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)

        if path.is_dir():
            path = path /  f"{self.relation}-result.jsonl"

        self.path = path
        
    def write_jsonl(self, json_obj):

        with open(self.path, "a") as f:
            json.dump(json_obj, f)
            f.writelines("\n")

class Neo4jConnection():
    def __init__(self, uri, auth):
        self.uri = uri
        self.auth = auth

    def is_local_connection(self):
        return "localhost" in self.uri.lower() or "127.0.0.1" in self.uri

    def __enter__(self):
        if self.is_local_connection():
            self.driver = GraphDatabase.driver(self.uri, auth=self.auth, encrypted=False, notifications_min_severity="OFF")
        else:
            self.driver = GraphDatabase.driver(self.uri, auth=self.auth, notifications_min_severity="OFF")
        return self.driver
    
    def __exit__(self, type, value, traceback):
        self.driver.close()

def load_template(prompt_template_file):
    with open(prompt_template_file, 'r') as f:
        template = f.read()

    #validate
    if template.find("%question%") == -1:
        raise ValueError("Template must contain %question%")

    return template

def load_examples(examples_file, use_examples_per_relations_format):
    examples = read_json_file(examples_file)
    
    # Validate
    if use_examples_per_relations_format:
        if not isinstance(examples, dict):
            raise ValueError("Examples must be a dict when using examples_per_relations_format")
    else:
        if not isinstance(examples, list):
            raise ValueError("Examples must be a list when not using examples_per_relations_format")
    
    return examples

def get_relation_id(string):
    match = re.search(r'P\d+', string)
    if match:
        return match.group()
    return None

def get_task_queue(pararel_task_dir, relations):
    task_queue = {}
    for relation in relations:
        # add suport for files with additional charactes, e.g. P36_subset.jsonl
        relation_id = get_relation_id(relation)
        task_queue[relation_id] = read_jsonl_file(os.path.join(pararel_task_dir, f"{relation}.jsonl"))
    return task_queue

def process_queue(neo4j_uri, neo4j_user, neo4j_password, ollama_model_name, task_queue, logger, template, examples, use_examples_per_relations_format, results_dir):
    
    # check that the model is available 
    try: 
        if ollama_model_name not in [model.model.split(":")[0] for model in ollama.list().models] + [model.model for model in ollama.list().models]:
            raise ValueError(f"Model {ollama_model_name} not available")
    except Exception as X:    
        raise ValueError(f"ollama not available")

    llm = OllamaLLM(ollama_model_name)

    # Connect to Neo4j database
    with Neo4jConnection(neo4j_uri, auth=(neo4j_user, neo4j_password)) as driver:
        driver.verify_connectivity()

        #get schema
        result = driver.execute_query("CALL apoc.meta.schema() yield value")
        neo4j_schema = str(result.records[0].value())

        total_num_of_tasks = sum(len(lst) for lst in task_queue.values())

        with tqdm(total=total_num_of_tasks) as pbar:

            for relation in task_queue.keys():

                result_file = ResultFile(results_dir, relation)

                if use_examples_per_relations_format:
                    retriever_examples = examples[relation]
                else:
                    retriever_examples = examples

                if len(retriever_examples) > 0:
                    retriever = Text2CypherRetriever(
                        driver=driver,
                        llm=llm,  
                        neo4j_schema=neo4j_schema,
                        examples=retriever_examples,
                    )
                else:
                    retriever = Text2CypherRetriever(
                        driver=driver,
                        llm=llm,  
                        neo4j_schema=neo4j_schema,
                    )

                logger.info("")
                logger.info(f"Processing relation {relation}")

                task_list = task_queue[relation]
                for task in task_list:
                
                    task["relation"] = relation

                    query_text = template.replace("%question%", task["prompt"])
                    task["query_text"] = query_text

                    try:
                        result = retriever.search(query_text=query_text)
                    except Exception as X:
                        task["result"] = ""
                        task["exception"] = str(X)

                        result_file.write_jsonl(task)
                        pbar.update(1)
                        continue

                    task["cypher"] = result.metadata["cypher"]

                    if len(result.items) == 0:
                        task["result"] = ""

                        result_file.write_jsonl(task)
                        pbar.update(1)
                        continue

                    if len(result.items) > 1:
                        logger.warning(f"More than one result for task {task}")
                        task["results_count"] = len(result.items)
                        task["multi_result"] = str([item.content for item in result.items]) 

                    task["result"] = result.items[0].content

                    result_file.write_jsonl(task)
                    pbar.update(1)


def main():
    parser = argparse.ArgumentParser("")
    
    # neo4j connection
    parser.add_argument("--neo4j_uri", type=str, help="", default="bolt://localhost:7687")
    parser.add_argument("--neo4j_user", type=str, help="", default="neo4j")
    parser.add_argument("--neo4j_password", type=str, help="", default="neo4j")

    # Process
    parser.add_argument("--ollama_model_name", type=str, help="model", default="tomasonjo/llama3-text2cypher-demo")

    parser.add_argument("--examples_file", type=str, help="model", default="")
    parser.add_argument("--use_examples_per_relations_format", action="store_true", help="")
    parser.add_argument("--prompt_template_file", type=str, help="model", default="prompt_template.txt")

    parser.add_argument("--pararel_task_dir", type=str, help="pararel data directory", default="./data/pararel")
    parser.add_argument("--relations", "-r", type=str, default="P937,P1412,P127,P103,P276,P159,P140,P136,P495,P17,P361,P36,P740,P264,P407,P138,P30,P131,P176,P449,P279,P19,P101,P364,P106,P1376,P178,P413,P27,P20", help="what relations")

    parser.add_argument("--run_dir", type=str, help="", default="./run")

    # parser.add_argument("--results_dir", type=str, help="", default="./results")
    # parser.add_argument("--log_dir", type=str, help="", default="./logs")

    args = parser.parse_args()

    session_dir = os.path.join(args.run_dir, datetime.now().strftime('%Y_%m_%d-%H_%M_%S'))

    results_dir = os.path.join(session_dir, "results")
    path = Path(results_dir)

    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)

    logger = get_logger(session_dir)

    relations = args.relations.split(",")

    task_queue = get_task_queue(args.pararel_task_dir, relations)

    template = load_template(args.prompt_template_file)
    if len(args.examples_file) > 0:
        examples = load_examples(args.examples_file, args.use_examples_per_relations_format)
    else:
        examples = ""

    # Add all configurations to the log file
    logger.info(f"Settings used")
    logger.info(f"relations: {relations}") 
    logger.info(f"neo4j uri: {args.neo4j_uri}, user: {args.neo4j_user}")
    logger.info(f"ollama model: {args.ollama_model_name}")
    logger.info(f"prompt template: {template}")
    logger.info(f"separate examples per relation: {args.use_examples_per_relations_format}")
    logger.info(f"examples: {examples}")
    logger.info(f"pararel task dir: {args.pararel_task_dir}")
    logger.info(f"session dir: {session_dir}")     

    process_queue(args.neo4j_uri, args.neo4j_user, args.neo4j_password, args.ollama_model_name, task_queue, logger, template, examples, args.use_examples_per_relations_format, results_dir)

    print("Done")

if __name__ == "__main__":
    main()