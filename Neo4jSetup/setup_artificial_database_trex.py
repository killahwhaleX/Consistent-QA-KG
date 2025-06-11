from neo4j import GraphDatabase
from pathlib import Path
import argparse
import logging
from datetime import datetime
import json
from tqdm import tqdm
import csv

scripts_path = Path().absolute() / "scripts"

# Helper functions

def read_file(file_path) -> str:
    with open(file_path, "r") as file:
        return file.read()  

def run_cypher_file(driver, cypherfile_path, message=""):
    content = read_file(scripts_path / cypherfile_path)

    for statement in content.split(";"):
        if statement.strip() != "":
            summary = driver.execute_query(content)
            if message:     
                print(message)

def get_logger(log_path):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    path = Path(log_path)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)

    if path.is_dir():
        path = path /  f"log{datetime.now().strftime('%Y_%m_%d-%H_%M_%S')}.txt"

    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

# Neo4j Helper functions

uri_template = "http://www.wikidata.org/entity/%entity_id%"
property_uri_template = "https://www.wikidata.org/wiki/Property:%property_id%"

# create_template = "MERGE (wi:WikiItem {uri: '%uri%'})"
# literal_property_template = "MERGE (wi:WikiItem {uri: '%uri%'}) SET wi.%prop_name% = '%prop_value%'"
# literal_properties_template = "MERGE (wi:WikiItem {uri: '%uri%'}) SET %properties%"

labels_template = "MERGE (wi {uri: '%uri%'}) SET wi %labels%"
literal_property_template_2_props = "MERGE (wi {uri: '%uri%'}) SET wi.%prop_name% = '%prop_value%', wi.%prop_name2% = '%prop_value2%'"
relation_template = "MATCH (wi {uri: '%subj_uri%'}) MATCH (wi2 {uri: '%obj_uri%'}) MERGE (wi)-[:%rel_label% {uri: '%rel_uri%', id: '%rel_id%'}]->(wi2)" 

def clean_string(value):
    cleaned_value =  value.replace(".", "_").replace(",", "_").replace("–", "_").replace("-", "_").replace("'", "_").replace("(", "_").replace(")", "_").replace("/", "_").replace("&", "_").replace("ü", "_").replace("æ", "_").replace("é", "_").replace('\u200b', '')    
    if not cleaned_value[0].isalpha():
        cleaned_value = "o" + cleaned_value
    return cleaned_value


def format_label_name(value):
    return clean_string("".join(value.replace("_", " ").title().split()))
   
# def format_property_name(value):
#     content = "".join(value.title().split())
#     content =  content[0].lower() + content[1:]
#     return clean_string(content)

def format_relationship_name(value):
    return clean_string("_".join(value.lower().split()).upper())

def format_value(value):
    return value.replace("'", "\\'").replace('"', '\\"').replace("`", "\\`")

def create_entity_with_id(driver, id, uri, name):
    create_query = literal_property_template_2_props.replace("%uri%", uri).replace("%prop_name%", "name").replace("%prop_value%", format_value(name)).replace("%prop_name2%", "id").replace("%prop_value2%", id)
    summary = driver.execute_query(create_query)
    return summary

def create_relation_with_id(driver, subj_uri, obj_uri, rel_id, rel_uri, rel_label):
    rel_query = relation_template.replace('%subj_uri%', subj_uri) .replace('%obj_uri%', obj_uri).replace('%rel_label%', format_relationship_name(rel_label)).replace('%rel_uri%', rel_uri).replace('%rel_id%', rel_id)                      
    summary = driver.execute_query(rel_query)
    return summary

def sort_by_word_count(string_list):
    return sorted(string_list, key=lambda s: len(s.split()))

def set_labels(driver, uri, labels, max_number_of_labels=-1):

    if max_number_of_labels >= 0:
        labels = sort_by_word_count(labels)[:max_number_of_labels]

    if len(labels) > 0:    
        labels_text = ''.join(':' + format_label_name(label) for label in labels if label)
        label_query = labels_template.replace("%uri%", uri).replace("%labels%", labels_text)
        summary = driver.execute_query(label_query)
        return summary
    else:
        return ""


class Neo4jConnection():
    def __init__(self, uri, auth):
        self.uri = uri
        self.auth = auth

    def is_local_connection(self):
        return "localhost" in self.uri.lower() or "127.0.0.1" in self.uri

    def __enter__(self):
        if self.is_local_connection():
            self.driver = GraphDatabase.driver(self.uri, auth=self.auth, encrypted=False)
        else:
            self.driver = GraphDatabase.driver(self.uri, auth=self.auth)
        return self.driver
    
    def __exit__(self, type, value, traceback):
        self.driver.close()

# Clean database    

def clean_database(uri, username, password):
    auth = (username, password)

    with Neo4jConnection(uri, auth=auth) as driver:
        driver.verify_connectivity()

        # Drop previous data
        run_cypher_file(driver, "drop_relations.cypher" )   
        run_cypher_file(driver, "drop_entities.cypher")

    print("Cleaned database")   

class RelationLabelsItem():
    rel_label = ""
    subj_label = ""
    obj_label = ""

    def __init__(self, rel_label, subj_label, obj_label):
        self.rel_label = rel_label
        self.subj_label = subj_label
        self.obj_label = obj_label

class RelationLabels():
    relation_labels = {}

    def __init__(self, relations_label_path = ""):
        if len(relations_label_path) > 0:
            with open( relations_label_path, 'r' ) as relations_label_file:
                reader = csv.DictReader(relations_label_file)
                for line in reader:
                    self.relation_labels[line["relation"]] = RelationLabelsItem(line["rel_label"], line["subj_label"], line["obj_label"])

    def __setitem__(self, key, item):
        self.relation_labels[key] = item

    def __getitem__(self, key):
        return self.relation_labels[key]

    def __len__(self):
        return len(self.relation_labels)

    def keys(self):
        return self.relation_labels.keys()

def read_trex_jsonl_file(filename: str):
    dataset = []
    with open(filename) as f:
        for line in f:
            loaded_example = json.loads(line)
            loaded_example.pop('evidences', None)
            dataset.append(loaded_example)

    return dataset

class TrexProcessingQueue:      
    queue = []
    def __init__(self, trex_data_folder, relations):

        trex_data_folder_path = Path(trex_data_folder)
        trex_data_folder_path = trex_data_folder_path.resolve()

        for relation in relations:
            json_file = trex_data_folder_path / f"{relation}.jsonl"
            entries = read_trex_jsonl_file(json_file)

            self.queue.extend(entries)

    def get_next(self):
        return self.queue.pop(0)
    
    def __len__(self):
        return len(self.queue)
    
    def is_empty(self):
        return not self.queue

class Handled:      

    handled = []
    handled_file_name = ""

    def __init__(self, handled_file_name = "handled.txt", load_handled = False):
        self.handled_file_name = handled_file_name
        
        if load_handled:
            with open(handled_file_name, 'r') as content_file:
                content = content_file.read()
                self.handled=content.splitlines()
        else:
            self.handled = []

    def append(self, item):
        self.handled.append(item)  
        with open(self.handled_file_name, "a") as content_file:
            content_file.write(item + "\n")    

    def is_empty(self):
        return len(self.queue) == 0
    
    def is_handled(self, item):
        return item in self.handled

def sort_vals(vals):
    # remove items with an end_date



    return vals

def process_queue(uri, username, password, queue, relations, relation_labels, common_label, logger, load_handled = False):
    
    auth = (username, password)
    handled = Handled(load_handled = load_handled)

    with Neo4jConnection(uri, auth=auth) as driver:
        driver.verify_connectivity()

        with tqdm(total=len(queue)) as pbar:
            while not queue.is_empty():
                q_item = queue.get_next()
                
                # if handled.is_handled(q_id):
                #     pbar.update(1)
                #     continue

                rel_id =  q_item["predicate_id"]   
                rel_uri = property_uri_template.replace("%property_id%", rel_id)
                rel_label = relation_labels[rel_id].rel_label

                subj_id = q_item["sub_uri"]
                subj_uri = uri_template.replace("%entity_id%", subj_id)
                subj_name = q_item["sub_label"]
                subj_labels = [relation_labels[rel_id].subj_label]   
                if common_label:
                    subj_labels.append(common_label)


                obj_id = q_item["obj_uri"]
                obj_uri = uri_template.replace("%entity_id%", obj_id)
                obj_name = q_item["obj_label"]   
                obj_labels = [relation_labels[rel_id].obj_label] 
                if common_label:
                    obj_labels.append(common_label)  

                # Create entry 
                create_entity_with_id(driver, subj_id, subj_uri, subj_name)
                set_labels(driver, subj_uri, subj_labels)
                logger.info(f"Created subject: {subj_name} ({subj_uri})")

                create_entity_with_id(driver, obj_id, obj_uri, obj_name)
                set_labels(driver, obj_uri, obj_labels)
                logger.info(f"Created object: {obj_name} ({obj_uri})")

                create_relation_with_id(driver, subj_uri, obj_uri, rel_id, rel_uri, rel_label)
                pbar.update(1)



def main():
    parser = argparse.ArgumentParser("")
    
    # neo4j connection
    parser.add_argument("--neo4j_uri", type=str, help="", default="bolt://localhost:7687")
    parser.add_argument("--neo4j_user", type=str, help="", default="neo4j")
    parser.add_argument("--neo4j_password", type=str, help="", default="neo4j")

    # keep or clean database
    parser.add_argument("--keep-database", action="store_true", help="")
    parser.add_argument("--load_handled", action="store_true", help="")

    # import
    parser.add_argument("--trex_data_folder", type=str, help="trex data directory", default="../data/trex/data/TREx")
    parser.add_argument("--relations", "-r", type=str, default="P937,P1412,P127,P103,P276,P159,P140,P136,P495,P17,P361,P36,P740,P264,P407,P138,P30,P131,P176,P449,P279,P19,P101,P364,P106,P1376,P178,P413,P27,P20", help="what relations")

    parser.add_argument("--common_label", type=str, default="", help="articifial relation labels")
    parser.add_argument("--relations_label_path", type=str, default="artificial_labels.csv", help="what relation labels")
    
    parser.add_argument("--log_path", type=str, help="", default="./logs")

    # parser.add_argument("--debug", action="store_true", help="")
    # parser.add_argument("--debug_max_items", type=int, help="", default="20")
    
    args = parser.parse_args()

    logger = get_logger(args.log_path)

    if not args.keep_database:
        clean_database(args.neo4j_uri, args.neo4j_user, args.neo4j_password)
    else:
        print("Keeping database")

    relations = args.relations.split(",")
    relation_labels = RelationLabels(args.relations_label_path) 

    if(len(relations) != len(relation_labels.keys())):
        #TODO: log "not all relataions has labels" 
        pass   

    queue = TrexProcessingQueue(args.trex_data_folder, relations)

    process_queue(args.neo4j_uri, args.neo4j_user, args.neo4j_password, queue, relations, relation_labels,  args.common_label, logger, args.load_handled)

if __name__ == "__main__":
    main()