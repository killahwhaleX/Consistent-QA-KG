# Consistent-KG-QA

This repo contains the code for the 2025 Masters Thesis ***Consistent Question Answering via Knowledge Graph Querying***. It describes how to create the runs that the paper's ananlysis is based on. 

## Setup

This file describes the setup command using Windows command line. They should be very similar in PowerShell or on linux.

Create project directory and cd into it

```
mkdir project
cd project
```

Clone this repository into the project folder

```
git clone https://github.com/killahwhaleX/Consistent-QA-KG.git
```

Clone the pararel-star repository into the project folder and apply an update patch

```
git clone https://github.com/dsaynova/pararel.git
git apply --whitespace=nowarn Consistent-QA-KG/set-up_pararel_repo/generate_data_py.patch
```

Create virtual environment and activate

```
python -m venv CQAKG
"./CQAKG/Scripts/activate"
```

or using CONDA

```
conda create --name CQAKG
conda activate CQAKG
```

install requirements

```
pip install -r Consistent-QA-KG\requirements.txt
```

## Generate ParaRel* data

Here we generate the task files that are used during the runs that contains the ParaRel* dataset

```
cd pararel/pararel/consistency
python generate_data.py --folder_name ../../pararel_taskfiles --data_path ../../data
cd ../../..
```

## Set-up Neo4j database

To create the artificial Neo4j database based on T-REx as described in the thesis, perform the following steps:

### Fetch T-REx data

```
curl --output data.zip https://dl.fbaipublicfiles.com/LAMA/data.zip
```

or for PowerShell

```
curl -Uri https://dl.fbaipublicfiles.com/LAMA/data.zip -OutFile "data.zip" )
```

followed by 

```
mkdir trex_data
tar -xf data.zip -C trex_data 
del data.zip
```

### Import data into Neo4j 

This require a local installation of Neo4j with apoc enabled. See 
https://neo4j.com/docs/operations-manual/current/installation/ and https://neo4j.com/docs/apoc/current/installation/ 

```
cd Consistent-QA-KG/Neo4jSetup
python setup_artificial_database_trex.py --trex_data_folder "../../trex_data/data/TREx" --neo4j_uri bolt://localhost:7687 --neo4j_user neo4j --neo4j_password <neo4j password>
cd ../..
```

## **Optional**: Create filtering file that can be applied to taskfiles or result files 

This example creates a filtering file that keeps a tenth of the original tasks at random 
```
cd Consistent-QA-KG\KG-RAG\filtering
python generate_filtering_file.py --proportion_to_keep 0.1 --trex_task_dir ../../../pararel/data/trex_lms_vocab
cd ../../..
```

Here is an example how this can be applied to the pararel_taskfiles to create a filtered subset

```
cd Consistent-QA-KG\KG-RAG\filtering
python filter_jsonl.py --jsonl_dir ../../../pararel_taskfiles --output_dir ../../../pararel_taskfiles_filtered
cd ../../..
```

## Ollama models download

Download and install from https://ollama.com/download

```
ollama pull tomasonjo/llama3-text2cypher-demo
ollama pull llama3:8b
```

## Perform runs 

### Filtered run example 

First an example of how to perform a run using the filtered files created in the previous step 

```
cd Consistent-QA-KG\KG-RAG
python T2Cgraphrag.py --ollama_model_name "tomasonjo/llama3-text2cypher-demo" --examples_file examples/simple_examples.json --prompt_template_file prompt_templates/prompt_template.txt --pararel_task_dir ../../pararel_taskfiles_filtered --run_dir ../../runs  --relations "P36,P101" --neo4j_uri bolt://localhost:7687 --neo4j_user neo4j --neo4j_password <neo4j password>
cd ../..
```

The following sections describes the commands to replicate the data used in the Thesis

### Configuration A

```
cd Consistent-QA-KG\KG-RAG
python T2Cgraphrag.py --ollama_model_name "tomasonjo/llama3-text2cypher-demo" --examples_file examples/general_examples.json --prompt_template_file prompt_templates/prompt_template.txt --pararel_task_dir ../../pararel_taskfiles_filtered --run_dir ../../runs 
cd ../..
```

### Configuration B

```
cd Consistent-QA-KG\KG-RAG
python T2Cgraphrag.py --ollama_model_name "tomasonjo/llama3-text2cypher-demo" --examples_file examples/simple_examples.json --prompt_template_file prompt_templates/prompt_template.txt --pararel_task_dir ../../pararel_taskfiles_filtered --run_dir ../../runs 
cd ../..
```

### Configuration C

```
cd Consistent-QA-KG\KG-RAG
python T2Cgraphrag.py --ollama_model_name "tomasonjo/llama3-text2cypher-demo" --examples_file examples/extended_examples.json --prompt_template_file prompt_templates/prompt_template.txt --pararel_task_dir ../../pararel_taskfiles_filtered --run_dir ../../runs 
cd ../..
```

### Configuration D

```
cd Consistent-QA-KG\KG-RAG
python T2Cgraphrag.py --ollama_model_name "tomasonjo/llama3-text2cypher-demo" --prompt_template_file prompt_templates/prompt_template.txt --pararel_task_dir ../../pararel_taskfiles_filtered --run_dir ../../runs 
cd ../..
```

### Configuration E

```
cd Consistent-QA-KG\KG-RAG
python T2Cgraphrag.py --ollama_model_name "llama3:8b" --examples_file examples/simple_examples.json --prompt_template_file prompt_templates/prompt_template.txt --pararel_task_dir ../../pararel_taskfiles_filtered --run_dir ../../runs  
cd ../..
```

### Configuration F

Note: The cohere api key is taken from a system variable named COHERE_API_KEY 

```
cd Consistent-QA-KG\KG-RAG
python T2Cgraphrag-cohere.py --examples_file examples/simple_examples.json --prompt_template_file prompt_templates/prompt_template.txt --pararel_task_dir ../../pararel_taskfiles_filtered --run_dir ../../runs  --real_mode --cohere_quota 200 --cohere_model command-r-08-2024 
cd ../..
```

## Analysis

After any successful run, move results files to a separate directory. E.g.

```
mkdir completed_run_filtered_dataset\inputs
copy results\2025_05_28-22_29_06\results\* completed_run_filtered_dataset\inputs
```


### Run analysis, this will create completed_run_filtered_dataset\outputs

```
cd Consistent-QA-KG\Analysis
python T2Canalysis.py --work-dir ../../completed_run_filtered_dataset --graph-path ../../pararel/data/pattern_data/graphs_json --relations_label_path ../Neo4jSetup/artificial_labels.csv
cd ../..
```


