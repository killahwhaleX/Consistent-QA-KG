import os
from pathlib import Path
import json
from tqdm import tqdm
import re
import json
from typing import List, Dict, Optional
import argparse
import logging
from datetime import datetime
from collections import Counter
import pickle
import csv
from statistics import mean, stdev
from collections import Counter


from utils import read_jsonl_file, write_jsonl_file 


def get_logger(session_dir:str ) -> logging.Logger:

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


def collect_data(results_dir: str, relations: List[str], logger: Optional[logging.Logger]) -> Dict[str, Dict[str, Dict[str, List]]]:
    data = {}
    for relation in relations:
        logger.info(f"Collecting relation {relation}")
        data[relation] = {}
        file_path = os.path.join(results_dir, f"{relation}-result.jsonl")
        if not os.path.exists(file_path):
            if logger:
                logger.warning(f"File not found: {file_path}")
            continue
        raw_data = read_jsonl_file(file_path)
        for entry in raw_data:
            rel_ix = entry.get("rel_ix")
            obj_label = entry.get("obj_label")
            if rel_ix is not None:
                if rel_ix not in data[relation]:
                    data[relation][rel_ix] = {
                        "results": [],
                        "correct": obj_label
                    }
                data[relation][rel_ix]["results"].append(entry)

    return data

def extract_result(raw_result: str) -> str:
    match = re.search(r"'([^']+)'", raw_result)
    cleaned = match.group(1) if match else raw_result
    return cleaned.strip().lower()


def get_relation_patterns(graph_path: str, relations: List[str]) -> Dict[str, List[str]]:
    return_dict = {}
    for relation in relations:
        pattern_path = os.path.join(graph_path, relation+".jsonl")
        patterns = read_jsonl_file(pattern_path)
        return_dict[relation] = [pattern_entry["pattern"] for pattern_entry in  patterns]
    return return_dict


def clean_data(data: Dict[str, Dict[str, Dict[str, List]]], logger: Optional[logging.Logger] = None) -> dict:
    for relation, rel_data in data.items():
        if logger:
            logger.info(f"Cleaning relation {relation}")
        for rel_ix, entry in rel_data.items():
            #print(entry)
            entry["correct"] = entry["correct"].strip().lower()  
            for result in entry["results"]:
                result["correct"] = entry["correct"].strip().lower() 
                result["extracted_result"] = extract_result(result["result"])

            entry["extracted_results"] = [item["extracted_result"] for item in entry["results"]]

    return data

def get_relation_name_from_cypher(cypher: str) -> str:
    relation_pattern = r"\[:(.*?)\]"  # Extracts text inside square brackets after '-[:'

    match = re.search(relation_pattern, cypher)
    if match:
        return (match.group(1)) 
    else:
        return ""
    
def split_relation_names(relation_str: str) -> List[str]:
    rels = relation_str.split("|")
    rels = [rel.replace(":", "").strip() for rel in rels]
    return rels

def clean_string(value:str) -> str:
    cleaned_value =  value.replace(".", "_").replace(",", "_").replace("–", "_").replace("-", "_").replace("'", "_").replace("(", "_").replace(")", "_").replace("/", "_").replace("&", "_").replace("ü", "_").replace("æ", "_").replace("é", "_").replace('\u200b', '')    
    if not cleaned_value[0].isalpha():
        cleaned_value = "o" + cleaned_value
    return cleaned_value

def format_label_name(value:str) -> str:
    return clean_string("".join(value.replace("_", " ").title().split()))
   
def format_relationship_name(value:str) -> str:
    return clean_string("_".join(value.lower().split()).upper())

class RelationLabelsItem():
    rel_label = ""
    subj_label = ""
    obj_label = ""

    def __init__(self, rel_label: str, subj_label: str, obj_label: str) -> None:
        self.rel_label = format_relationship_name(rel_label)
        self.subj_label = format_label_name(subj_label)
        self.obj_label = format_label_name(obj_label)

    def to_dict(self):
        return {
            "rel_label": self.rel_label,
            "subj_label": self.subj_label,
            "obj_label": self.obj_label
        }

class RelationLabels():
    relation_labels: Dict[str, RelationLabelsItem] = {}

    def __init__(self, relations_label_path : str = "") -> None:
        if len(relations_label_path) > 0:
            with open( relations_label_path, 'r' ) as relations_label_file:
                reader = csv.DictReader(relations_label_file)
                for line in reader:
                    self.relation_labels[line["relation"]] = RelationLabelsItem(line["rel_label"], line["subj_label"], line["obj_label"])

    def __setitem__(self, key: str, item: RelationLabelsItem) -> None:
        self.relation_labels[key] = item

    def __getitem__(self, key:str) -> RelationLabelsItem:
        return self.relation_labels[key]

    def __len__(self) -> int:
        return len(self.relation_labels)

    def keys(self) -> List[str]:
        return list(self.relation_labels.keys())    
    
def evaluate(data: Dict[str, Dict[str, Dict[str, list]]], logger: Optional[logging.Logger] = None) -> dict:
    relation_metrics = {}
    all_relation_consistency_scores = []
    
    all_relation_accuracy_scores = []
    all_relation_all_task_accuracy_scores = []

    all_relation_consistent_and_accurate_scores = []
    all_relation_all_task_consistent_and_accurate_scores = [] 

    for relation, rel_data in data.items():
        if logger:
            logger.info(f"Evaluating relation {relation}")
        
        rel_ix_consistency_scores = []

        rel_ix_accuracy_scores = {}
        rel_ix_task_accuracy_scores = {}

        rel_ix_consistent_and_accurate_scores = [] 

        rel_ix_metrics = {}  # Store per-tuple metrics

        # Consistency: Compute metrics for each tuple (rel_ix)
        for rel_ix, entry in rel_data.items():
            results = entry.get("extracted_results", [])
            correct_answer = entry.get("correct")  # Ground truth answer
            
            # Pairwise consistency calculation
            if len(results) > 1:
                total_pairs = 0
                matching_pairs = 0
                for i in range(len(results)):
                    for j in range(i + 1, len(results)):
                        total_pairs += 1

                        # a blank answer is never consistent
                        if results[i] == "" or results[j] == "":
                            continue

                        if results[i] == results[j]:
                            matching_pairs += 1

                consistency_score = matching_pairs / total_pairs if total_pairs > 0 else 0.0
            else:
                consistency_score = 1.0  # If only one result, it's trivially consistent
            
            rel_ix_consistency_scores.append(consistency_score)

            # Accuracy calculation (only use first pattern)
            accuracy_score = 1.0 if results and results[0] == correct_answer else 0.0
            rel_ix_accuracy_scores[rel_ix] = accuracy_score
            
            correct_count = results.count(correct_answer)
            task_accuracy_score = correct_count / len(results) if results else 0.0
            rel_ix_task_accuracy_scores[rel_ix] = task_accuracy_score

            # Consistent & Accurate metric based on existing consistency and accuracy scores
            consistent_and_accurate_score = 1.0 if (consistency_score == 1.0 and accuracy_score == 1.0) else 0.0
            rel_ix_consistent_and_accurate_scores.append(consistent_and_accurate_score)

            # Store tuple-level metrics
            rel_ix_metrics[rel_ix] = {
                "consistency": consistency_score,
                "accuracy": accuracy_score,
                "task_accuracy": task_accuracy_score,
                "consistent_and_accurate": consistent_and_accurate_score
            }

        # Compute relation-level consistency (average over tuples)
        relation_consistency_score = (
            sum(rel_ix_consistency_scores) / len(rel_ix_consistency_scores)
            if rel_ix_consistency_scores else 0.0
        )
        all_relation_consistency_scores.append(relation_consistency_score)

        relation_accuracy = (
            sum(rel_ix_accuracy_scores.values()) / len(rel_ix_accuracy_scores)
            if rel_ix_accuracy_scores else 0.0
        )
        all_relation_accuracy_scores.append(relation_accuracy)

        # Compute relation-level accuracy (micro-average over tuples)
        task_relation_accuracy = (
            sum(rel_ix_task_accuracy_scores.values()) / len(rel_ix_task_accuracy_scores)
            if rel_ix_task_accuracy_scores else 0.0
        )
        all_relation_all_task_accuracy_scores.append(task_relation_accuracy)

        # Compute relation-level Consistent & Accurate (average over tuples)
        relation_consistent_and_accurate_score = (
            sum(rel_ix_consistent_and_accurate_scores) / len(rel_ix_consistent_and_accurate_scores)
            if rel_ix_consistent_and_accurate_scores else 0.0
        )
        all_relation_consistent_and_accurate_scores.append(relation_consistent_and_accurate_score)        
        

        # Store relation-level metrics
        relation_metrics[relation] = {
            "relation_consistency": relation_consistency_score,
            "relation_accuracy": relation_accuracy,
            "task_relation_accuracy": task_relation_accuracy,
            "relation_consistent_and_accurate": relation_consistent_and_accurate_score, 
            "details": rel_ix_metrics  # Store per-tuple details
        }

    # Compute overall consistency (macro-average over relations)
    overall_consistency = mean(all_relation_consistency_scores) if all_relation_consistency_scores else 0.0
    overall_consistency_stdev = stdev(all_relation_consistency_scores) if all_relation_consistency_scores else 0.0

    overall_accuracy = mean(all_relation_accuracy_scores) if all_relation_accuracy_scores else 0.0
    overall_accuracy_stdev = stdev(all_relation_accuracy_scores) if all_relation_accuracy_scores else 0.0

    overall_task_accuracy = mean(all_relation_all_task_accuracy_scores) if all_relation_all_task_accuracy_scores else 0.0
    overall_task_accuracy_stdev = stdev(all_relation_all_task_accuracy_scores) if all_relation_all_task_accuracy_scores else 0.0


    # Compute overall Consistent & Accurate (macro-average over relations)
    overall_consistent_and_accurate = mean(all_relation_consistent_and_accurate_scores) if all_relation_consistent_and_accurate_scores else 0.0
    overall_consistent_and_accurate_stdev = stdev(all_relation_consistent_and_accurate_scores) if all_relation_consistent_and_accurate_scores else 0.0



    result = {
        "overall_consistency": overall_consistency,
        "overall_consistency_stdev": overall_consistency_stdev,
        "overall_accuracy": overall_accuracy,
        "overall_accuracy_stdev": overall_accuracy_stdev,
        "overall_task_accuracy": overall_task_accuracy,
        "overall_task_accuracy_stdev": overall_task_accuracy_stdev,
        "overall_consistent_and_accurate": overall_consistent_and_accurate,
        "overall_consistent_and_accurate_stdev": overall_consistent_and_accurate_stdev,
        "relations": relation_metrics
    }
    
    return result

# analyze(cleaned_data, relation_patterns, relation_labels, logger)
def analyze(data: Dict[str, Dict[str, Dict[str, list]]], relation_patterns: Dict[str, List[str]], relation_labels: RelationLabels, logger: Optional[logging.Logger] = None) -> dict:

    analysis_results = {}
    incorrect_entries = {}
    correct_entries = {}

    for relation, rel_data in data.items():
        incorrect_entries[relation] = []
        correct_entries[relation] = []


        analysis_results[relation] = {
            "relation_labels": relation_labels[relation].to_dict(),
            "total_number_of_rel_ix": len(rel_data),
            "all_correct_rel_ix_count": 0,
            "incorrect_per_rel_ix_count": {},
            "total_number_of_pattern_ix": 0,
            "incorrect_per_pattern_ix_count": {},
            "result_lengths_matching": True,
            "results_matching_pararel_star_patterns": True, 
            "relation_names_in_cypher_correct": {},
            "relation_names_in_cypher_incorrect": {},
            "per_pattern_relation_name_distribution": {},
            "found_right_relation": 0,
            "cypher_errors": 0,
            "blank_results": 0
        }

        if logger:
            logger.info(f"Evaluating relation {relation}")
        
        # Valudate that all entries has the same number of results
        result_lengths = [len(entry["results"]) for entry in rel_data.values()]
        if len(set(result_lengths)) != 1:
            print(f"Mismatch in number of results for relation {relation}.")
            analysis_results[relation]["result_lengths_matching"] = False
            
        elif result_lengths[0] != len(relation_patterns[relation]):
            print("Mismatch with pattern counts for relation: ", relation)
            analysis_results[relation]["results_matching_pararel_star_patterns"] = False

        else:
            analysis_results[relation]["total_number_of_pattern_ix"] = len(relation_patterns[relation])

        analysis_results[relation]["patterns"] = relation_patterns[relation]
        temp_rel_name_per_pattern = {i: [] for i in range(len(relation_patterns[relation]))}      
           
        right_relation_name_found = 0   
        right_relation_name_not_found = 0   
        for rel_ix, entry in rel_data.items():

            for result in entry["results"]:

                relation_name = get_relation_name_from_cypher(result.get("cypher", ""))
                if relation_name == "":
                    relation_name = "<blank>"

                if "exception" in result:  
                    analysis_results[relation]["cypher_errors"] += 1

                if result["result"].strip() == "":
                    analysis_results[relation]["blank_results"] += 1
                   

                if analysis_results[relation]["relation_labels"]["rel_label"] in split_relation_names(relation_name):
                    right_relation_name_found += 1
                else:
                    right_relation_name_not_found += 1

                if entry["correct"] != result["extracted_result"]:
                    incorrect_entries[relation].append(result)

                    analysis_results[relation]["incorrect_per_rel_ix_count"][result["rel_ix"]] = analysis_results[relation]["incorrect_per_rel_ix_count"].get(result["rel_ix"], 0) + 1
                    analysis_results[relation]["incorrect_per_pattern_ix_count"][result["pattern_ix"]] = analysis_results[relation]["incorrect_per_pattern_ix_count"].get(result["pattern_ix"], 0) + 1
                

                    analysis_results[relation]["relation_names_in_cypher_incorrect"][relation_name] = analysis_results[relation]["relation_names_in_cypher_incorrect"].get(relation_name, 0) + 1
                else:
                    correct_entries[relation].append(result)

                    analysis_results[relation]["relation_names_in_cypher_correct"][relation_name] = analysis_results[relation]["relation_names_in_cypher_correct"].get(relation_name, 0) + 1

                temp_rel_name_per_pattern[result["pattern_ix"]].append(relation_name)

            if analysis_results[relation]["incorrect_per_rel_ix_count"].get(result["rel_ix"], 0) == 0:
                analysis_results[relation]["all_correct_rel_ix_count"] += 1


        analysis_results[relation]["found_right_relation"] = (right_relation_name_found / (right_relation_name_found + right_relation_name_not_found)) if (right_relation_name_found + right_relation_name_not_found) > 0 else 0.0
        analysis_results[relation]["per_pattern_relation_name_distribution"]  = {
            idx: dict(Counter(rel_list)) 
            for idx, rel_list in temp_rel_name_per_pattern.items()
        }

        # Sort the incorrect counts for pattern_ix
        analysis_results[relation]["incorrect_per_pattern_ix_count"] = {key:value for key, value in sorted(analysis_results[relation]["incorrect_per_pattern_ix_count"].items(), key=lambda item: int(item[0]))}
        pass
        
    

    return analysis_results, incorrect_entries, correct_entries


def writeResults(consistency_data: Dict[str, Dict[str, float]], file_dir: str, logger: Optional[logging.Logger] = None) -> None:
    file_path = os.path.join(file_dir, "results.json")
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(consistency_data, file, indent=4)
    if logger:
        logger.info(f"Data written to file: {file_path}")

def append_evaluation_results(analysis_results, evaluation_results):
    for relation in analysis_results.keys():
        if relation in evaluation_results["relations"]:
            analysis_results[relation]["evaluation"] = {
                "relation_consistency": evaluation_results["relations"][relation]["relation_consistency"],
                "relation_accuracy": evaluation_results["relations"][relation]["relation_accuracy"],
                "task_relation_accuracy": evaluation_results["relations"][relation]["task_relation_accuracy"],
                "relation_consistent_and_accurate": evaluation_results["relations"][relation]["relation_consistent_and_accurate"]
            }
    return analysis_results


def main():
    parser = argparse.ArgumentParser("")

    parser.add_argument("--work-dir", type=str, help="")
    parser.add_argument("--graph-path", type=str, help="")
    parser.add_argument("--relations", "-r", type=str, default="P17,P19,P20,P27,P30,P36,P101,P103,P106,P127,P131,P136,P138,P140,P159,P176,P178,P264,P276,P279,P361,P364,P407,P413,P449,P495,P740,P937,P1376,P1412", help="what relations")
    parser.add_argument("--relations_label_path", type=str, default="artificial_labels.csv", help="what relation labels")
    parser.add_argument("--relations_patterns_dir", type=str, default="graphs_json", help="")

    args = parser.parse_args()

    inputs_dir = os.path.join(args.work_dir, "inputs")

    session_dir = os.path.join(args.work_dir, "outputs")
    results_dir = os.path.join(session_dir, "results")

    path = Path(results_dir)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    
    logger = get_logger(session_dir)

    relations = args.relations.split(",")
    collected_data = collect_data(results_dir=inputs_dir, relations=relations, logger=logger)
    cleaned_data = clean_data(data=collected_data, logger=logger)

    evaluation_results = evaluate(data=cleaned_data,logger=logger)


    relation_patterns = get_relation_patterns(args.graph_path, relations)
    relation_labels = RelationLabels(args.relations_label_path) 
    analysis_results, incorrect_entries, correct_entries = analyze(cleaned_data, relation_patterns, relation_labels, logger)

    analysis_results = append_evaluation_results(analysis_results, evaluation_results)


    # write results
    with open(os.path.join(results_dir, "evaluation_results.json"), 'w', encoding='utf-8') as f:
        json.dump(evaluation_results, f, indent=4)    

    with open(os.path.join(results_dir, "analysis_results.json"), 'w', encoding='utf-8') as f:
        json.dump(analysis_results, f, indent=4)    

    incorrect_results_dir = path / "incorrect_entries"

    if not incorrect_results_dir.exists():
        incorrect_results_dir.mkdir(parents=True, exist_ok=True)

    for key in incorrect_entries.keys():
        write_jsonl_file(incorrect_entries[key], os.path.join(incorrect_results_dir, key + ".jsonl"))

    correct_results_dir = path / "correct_entries"

    if not correct_results_dir.exists():
        correct_results_dir.mkdir(parents=True, exist_ok=True)

    for key in correct_entries.keys():
        write_jsonl_file(correct_entries[key], os.path.join(correct_results_dir, key + ".jsonl"))


if __name__ == "__main__":
    main()
