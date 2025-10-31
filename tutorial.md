<h1 align='center'>Synthetic Data Generation and Unsloth Tutorial</h1>

## 📚 Table of Contents:

- [Synthetic Data Kit: Data Generation](#synthetic-data-generation)
- [Unsloth: Fine-Tuning and saving the model](#fine-tuning)

## Synthetic Data Generation

In this section, we use the CLI from synthetic-data-kit to generate datasets

### Testing Synthetic Data Kit Command

Please make sure you are running vllm by opening a terminal and typing `vllm serve Unsloth/Llama-3.3-70B-Instruct   --port 8001   --max-model-len 48000   --gpu-memory-utilization 0.85`


```python
!synthetic-data-kit --help
```

### Exploring Synthetic Data Kit CLI

This command displays the help menu for the `synthetic-data-kit` CLI tool, showing available commands:
- **system-check**: Verify LLM provider server is running
- **ingest**: Parse documents (PDF, HTML, YouTube, etc.) into clean text
- **create**: Generate synthetic content (Q&A pairs, instructions, etc.) using LLM
- **curate**: Filter and clean generated content based on quality scores
- **save-as**: Convert data to different formats (fine-tuning format, JSON, etc.)
- **server**: Launch web interface for the toolkit


```python
!synthetic-data-kit -c tutorial_config.yaml system-check
```

    Loading config from: /usr/local/lib/python3.12/dist-packages/synthetic_data_kit/config.yaml
    Config has LLM provider set to: api-endpoint
    Loading config from: /usr/local/lib/python3.12/dist-packages/synthetic_data_kit/config.yaml
    Config has LLM provider set to: api-endpoint
    Loading config from: tutorial_config.yaml
    Config has LLM provider set to: vllm
    [1;34mEnvironment variable check:[0m
    API_ENDPOINT_KEY: Not found
    get_llm_provider returning: vllm
    [?25l[32m vLLM server is running at [0m[4;94mhttp://localhost:8001/v1[0m
    [2KAvailable models: [1m{[0m[32m'object'[0m: [32m'list'[0m, [32m'data'[0m: [1m[[0m[1m{[0m[32m'id'[0m: 
    [32m'Unsloth/Llama-3.3-70B-Instruct'[0m, [32m'object'[0m: [32m'model'[0m, [32m'created'[0m: [1;36m1761679259[0m, 
    [32m'owned_by'[0m: [32m'vllm'[0m, [32m'root'[0m: [32m'Unsloth/Llama-3.3-70B-Instruct'[0m, [32m'parent'[0m: [3;35mNone[0m, 
    [32m'max_model_len'[0m: [1;36m48000[0m, [32m'permission'[0m: [1m[[0m[1m{[0m[32m'id'[0m: 
    [32m'modelperm-dad8f95f2c464f67a1721d2db0dff059'[0m, [32m'object'[0m: [32m'model_permission'[0m, 
    [32m'created'[0m: [1;36m1761679259[0m, [32m'allow_create_engine'[0m: [3;91mFalse[0m, [32m'allow_sampling'[0m: [3;92mTrue[0m, 
    [32m'allow_logprobs'[0m: [3;92mTrue[0m, [32m'allow_search_indices'[0m: [3;91mFalse[0m, [32m'allow_view'[0m: [3;92mTrue[0m, 
    [32m'allow_fine_tuning'[0m: [3;91mFalse[0m, [32m'organization'[0m: [32m'*'[0m, [32m'group'[0m: [3;35mNone[0m, [32m'is_blocking'[0m: 
    [3;91mFalse[0m[1m}[0m[1m][0m[1m}[0m[1m][0m[1m}[0m
    [2K[32m⠋[0m Checking vLLM server at http://localhost:8001/v1...
    [1A[2K


```python
mkdir -p logical_reasoning/sources logical_reasoning/data/input logical_reasoning/data/parsed logical_reasoning/data/generated logical_reasoning/data/curated logical_reasoning/data/final
```

### Verifying LLM Server Status

This command checks if the vLLM server is running and accessible at `http://localhost:8001/v1`. It displays:
- Server status and endpoint
- Available models (here: Unsloth/Llama-3.3-70B-Instruct)
- Model configuration (max context length: 48000 tokens)

The system is configured to use the vLLM provider as specified in `tutorial_config.yaml`.

### Creating Project Directory Structure

This command creates a well-organized directory structure for the logical reasoning project:
- `sources/`: Store original source documents (PDFs, etc.)
- `data/input/`: Input files for processing
- `data/parsed/`: Parsed text files after document ingestion
- `data/generated/`: Generated synthetic Q&A pairs
- `data/curated/`: Quality-filtered data after curation
- `data/final/`: Final formatted data ready for fine-tuning


```python
cd logical_reasoning
```

    /workspace/AIAC/logical_reasoning


### Navigating to Project Directory

Changes the current working directory to `logical_reasoning/` where all subsequent operations will take place.


```python
# !wget -P sources/ -q --show-progress   "https://www.csus.edu/indiv/d/dowdenb/4/logical-reasoning-archives/logical-reasoning-2017-12-02.pdf"   "https://people.cs.umass.edu/~pthomas/solutions/Liar_Truth.pdf"
```

### Downloading Source Documents

Downloads two PDF documents related to logical reasoning and liar/truth puzzles:
1. "Logical Reasoning" textbook from CSU Sacramento
2. "Liar and Truth Teller Puzzles" from UMass

These documents will serve as the knowledge base for generating synthetic training data. The `-q` flag runs wget in quiet mode, and `--show-progress` displays a progress bar.


```python
!cp sources/* data/input/
```

### Copying Source Files to Input Directory

Copies all downloaded source documents from `sources/` to `data/input/` to prepare them for the ingestion pipeline.


```python
!synthetic-data-kit ingest ./data/input/
```

    Loading config from: /usr/local/lib/python3.12/dist-packages/synthetic_data_kit/config.yaml
    Config has LLM provider set to: api-endpoint
    Loading config from: /usr/local/lib/python3.12/dist-packages/synthetic_data_kit/config.yaml
    Config has LLM provider set to: api-endpoint
    Loading config from: /usr/local/lib/python3.12/dist-packages/synthetic_data_kit/config.yaml
    Config has LLM provider set to: api-endpoint
    [34mProcessing directory: [0m[1;34m.[0m[1;35m/data/input/[0m
    [34mFound [0m[1;36m9[0m[34m supported files to process[0m
    WARNING:pdfminer.pdfinterp:Cannot set gray stroke color because /'P47' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray non-stroke color because /'P47' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray stroke color because /'P48' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray non-stroke color because /'P48' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray stroke color because /'P49' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray non-stroke color because /'P49' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray stroke color because /'P50' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray non-stroke color because /'P50' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray stroke color because /'P51' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray non-stroke color because /'P51' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray stroke color because /'P52' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray non-stroke color because /'P52' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray stroke color because /'P54' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray non-stroke color because /'P54' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray stroke color because /'P57' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray non-stroke color because /'P57' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray stroke color because /'P58' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray non-stroke color because /'P58' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray stroke color because /'P59' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray non-stroke color because /'P59' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray stroke color because /'P60' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray non-stroke color because /'P60' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray stroke color because /'P61' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray non-stroke color because /'P61' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray stroke color because /'P62' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray non-stroke color because /'P62' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray stroke color because /'P63' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray non-stroke color because /'P63' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray stroke color because /'P63' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray non-stroke color because /'P63' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray stroke color because /'P72' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray non-stroke color because /'P72' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray stroke color because /'P72' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray non-stroke color because /'P72' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray stroke color because /'P82' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray non-stroke color because /'P82' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray stroke color because /'P85' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray non-stroke color because /'P85' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray stroke color because /'P86' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray non-stroke color because /'P86' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray stroke color because /'P87' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray non-stroke color because /'P87' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray stroke color because /'P93' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray non-stroke color because /'P93' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray stroke color because /'P94' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray non-stroke color because /'P94' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray stroke color because /'P95' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray non-stroke color because /'P95' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray stroke color because /'P96' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray non-stroke color because /'P96' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray stroke color because /'P99' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray non-stroke color because /'P99' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray stroke color because /'P100' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray non-stroke color because /'P100' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray stroke color because /'P101' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray non-stroke color because /'P101' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray stroke color because /'P102' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray non-stroke color because /'P102' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray stroke color because /'P96' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray non-stroke color because /'P96' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray stroke color because /'P99' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray non-stroke color because /'P99' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray stroke color because /'P100' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray non-stroke color because /'P100' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray stroke color because /'P101' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray non-stroke color because /'P101' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray stroke color because /'P102' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray non-stroke color because /'P102' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray stroke color because /'P124' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray non-stroke color because /'P124' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray stroke color because /'P172' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray non-stroke color because /'P172' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray stroke color because /'P172' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray non-stroke color because /'P172' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray stroke color because /'P182' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray non-stroke color because /'P182' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray stroke color because /'P183' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray non-stroke color because /'P183' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray stroke color because /'P184' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray non-stroke color because /'P184' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray stroke color because /'P185' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray non-stroke color because /'P185' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray stroke color because /'P214' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray non-stroke color because /'P214' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray stroke color because /'P214' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray non-stroke color because /'P214' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray stroke color because /'P222' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray non-stroke color because /'P222' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray stroke color because /'P223' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray non-stroke color because /'P223' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray stroke color because /'P224' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray non-stroke color because /'P224' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray stroke color because /'P225' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray non-stroke color because /'P225' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray stroke color because /'P268' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray non-stroke color because /'P268' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray stroke color because /'P268' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray non-stroke color because /'P268' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray stroke color because /'P276' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray non-stroke color because /'P276' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray stroke color because /'P277' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray non-stroke color because /'P277' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray stroke color because /'P278' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray non-stroke color because /'P278' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray stroke color because /'P279' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray non-stroke color because /'P279' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray stroke color because /'P314' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray non-stroke color because /'P314' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray stroke color because /'P314' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray non-stroke color because /'P314' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray stroke color because /'P322' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray non-stroke color because /'P322' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray stroke color because /'P323' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray non-stroke color because /'P323' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray stroke color because /'P324' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray non-stroke color because /'P324' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray stroke color because /'P325' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray non-stroke color because /'P325' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray stroke color because /'P368' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray non-stroke color because /'P368' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray stroke color because /'P368' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray non-stroke color because /'P368' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray stroke color because /'P376' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray non-stroke color because /'P376' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray stroke color because /'P377' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray non-stroke color because /'P377' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray stroke color because /'P378' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray non-stroke color because /'P378' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray stroke color because /'P379' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray non-stroke color because /'P379' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray stroke color because /'P421' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray non-stroke color because /'P421' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray stroke color because /'P421' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray non-stroke color because /'P421' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray stroke color because /'P429' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray non-stroke color because /'P429' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray stroke color because /'P430' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray non-stroke color because /'P430' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray stroke color because /'P431' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray non-stroke color because /'P431' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray stroke color because /'P432' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray non-stroke color because /'P432' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray stroke color because /'P481' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray non-stroke color because /'P481' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray stroke color because /'P481' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray non-stroke color because /'P481' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray stroke color because /'P489' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray non-stroke color because /'P489' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray stroke color because /'P490' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray non-stroke color because /'P490' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray stroke color because /'P491' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray non-stroke color because /'P491' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray stroke color because /'P492' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray non-stroke color because /'P492' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray stroke color because /'P539' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray non-stroke color because /'P539' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray stroke color because /'P539' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray non-stroke color because /'P539' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray stroke color because /'P547' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray non-stroke color because /'P547' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray stroke color because /'P548' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray non-stroke color because /'P548' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray stroke color because /'P549' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray non-stroke color because /'P549' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray stroke color because /'P550' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray non-stroke color because /'P550' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray stroke color because /'P599' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray non-stroke color because /'P599' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray stroke color because /'P599' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray non-stroke color because /'P599' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray stroke color because /'P607' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray non-stroke color because /'P607' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray stroke color because /'P608' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray non-stroke color because /'P608' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray stroke color because /'P609' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray non-stroke color because /'P609' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray stroke color because /'P610' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray non-stroke color because /'P610' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray stroke color because /'P656' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray non-stroke color because /'P656' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray stroke color because /'P656' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray non-stroke color because /'P656' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray stroke color because /'P664' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray non-stroke color because /'P664' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray stroke color because /'P665' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray non-stroke color because /'P665' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray stroke color because /'P666' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray non-stroke color because /'P666' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray stroke color because /'P667' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray non-stroke color because /'P667' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray stroke color because /'P710' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray non-stroke color because /'P710' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray stroke color because /'P710' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray non-stroke color because /'P710' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray stroke color because /'P718' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray non-stroke color because /'P718' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray stroke color because /'P719' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray non-stroke color because /'P719' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray stroke color because /'P720' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray non-stroke color because /'P720' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray stroke color because /'P721' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray non-stroke color because /'P721' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray stroke color because /'P749' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray non-stroke color because /'P749' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray stroke color because /'P749' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray non-stroke color because /'P749' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray stroke color because /'P757' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray non-stroke color because /'P757' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray stroke color because /'P758' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray non-stroke color because /'P758' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray stroke color because /'P759' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray non-stroke color because /'P759' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray stroke color because /'P760' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray non-stroke color because /'P760' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray stroke color because /'P804' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray non-stroke color because /'P804' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray stroke color because /'P804' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray non-stroke color because /'P804' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray stroke color because /'P812' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray non-stroke color because /'P812' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray stroke color because /'P813' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray non-stroke color because /'P813' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray stroke color because /'P814' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray non-stroke color because /'P814' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray stroke color because /'P815' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray non-stroke color because /'P815' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray stroke color because /'P866' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray non-stroke color because /'P866' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray stroke color because /'P866' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray non-stroke color because /'P866' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray stroke color because /'P874' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray non-stroke color because /'P874' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray stroke color because /'P875' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray non-stroke color because /'P875' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray stroke color because /'P876' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray non-stroke color because /'P876' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray stroke color because /'P877' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray non-stroke color because /'P877' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray stroke color because /'P919' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray non-stroke color because /'P919' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray stroke color because /'P919' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray non-stroke color because /'P919' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray stroke color because /'P927' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray non-stroke color because /'P927' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray stroke color because /'P928' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray non-stroke color because /'P928' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray stroke color because /'P929' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray non-stroke color because /'P929' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray stroke color because /'P930' is an invalid float value
    WARNING:pdfminer.pdfinterp:Cannot set gray non-stroke color because /'P930' is an invalid float value
    [32m✓ Circular Arrangement MCQ [0m[1;32m[[0m[32mFree PDF[0m[1;32m][0m[32m - Objective Question Answer for Circular [0m
    [32mArrangement Quiz - Download Now!.pdf[0m
    [32m✓ SA_Set_No_173.pdf[0m
    [32m✓ Seating_Arrangement_PDF_1.pdf[0m
    [32m✓ Seating_Arrangement_PDF_2.pdf[0m
    [32m✓ Seating_Arrangement_PDF_3.pdf[0m
    [32m✓ blood-relation-puzzle.pdf[0m
    [32m✓ blood-relationship.pdf[0m
    [32m✓ circular-arrangement-with-blood-relation.pdf[0m
    [32m✓ seating-arrangement-logical-reasoning-.pdf[0m
    
    [1m==================================================[0m
    [1;34mProcessing Summary:[0m
    Total files: [1;36m9[0m
    [32mSuccessful: [0m[1;36m9[0m
    [32mFailed: [0m[1;36m0[0m
    [1m==================================================[0m
    [32m✅ All files processed successfully![0m


### Ingesting and Parsing Documents

This command processes the PDF files in `data/input/` using the synthetic-data-kit's **ingest** command:
- Extracts text content from PDFs
- Cleans and normalizes the text
- Saves parsed text files to `data/parsed/`

The output shows successful processing of 2 PDF files (Liar_Truth.pdf and logical-reasoning-2017-12-02.pdf).

Note: This will take about 10 minutes, set `--verbose` flag to see progress or reduce the `num-pairs` for a faster test


```python
# !synthetic-data-kit -c ../tutorial_config.yaml create ./data/parsed/ --type qa --num-pairs 50
!synthetic-data-kit -c ../tutorial_config.yaml create ./data/parsed/ --type qa --num-pairs 167 --verbose
!synthetic-data-kit -c ../tutorial_config.yaml create ./data/parsed/ --type cot --num-pairs 50 --verbose
!synthetic-data-kit -c ../tutorial_config.yaml curate ./data/generated/ --threshold 7.0
!synthetic-data-kit save-as ./data/curated/ --format ft --verbose
```

    Loading config from: /usr/local/lib/python3.12/dist-packages/synthetic_data_kit/config.yaml
    Config has LLM provider set to: api-endpoint
    Loading config from: /usr/local/lib/python3.12/dist-packages/synthetic_data_kit/config.yaml
    Config has LLM provider set to: api-endpoint
    Loading config from: ../tutorial_config.yaml
    Config has LLM provider set to: vllm
    get_llm_provider returning: vllm
    [32m🔗 Using vllm provider[0m
    [34mProcessing directory: [0m[1;34m.[0m[1;35m/data/parsed/[0m[34m for qa generation[0m
    [34mFound [0m[1;36m9[0m[34m qa files to process[0m
    [2KLoading config from: ../tutorial_config.yaml━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m (0/9) [33m0:00:00[0m
    [2KConfig has LLM provider set to: vllm━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m (0/9) [33m0:00:00[0m
    [2KL Using vllm provider0m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m (0/9) [33m0:00:00[0m
    [2KLoading config from: ../tutorial_config.yaml━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m (0/9) [33m0:00:00[0m
    [2KConfig has LLM provider set to: vllm━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m (0/9) [33m0:00:00[0m
    [2KGenerating document summary...━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m (0/9) [33m0:00:00[0m
    Generating qa content [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m (0/9) [33m0:00:00[0mINFO:synthetic_data_kit.models.llm_client:Sending request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating qa content [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m (0/9) [33m0:00:08[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    [2KSummary generated (964 chars)
    [2KGenerating QA pairs...m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m (0/9) [33m0:00:08[0m
    [2KDocument split into 9 chunks━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m (0/9) [33m0:00:08[0m
    [2KUsing batch size of 16m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m (0/9) [33m0:00:08[0m
    [2KProcessing 9 chunks to generate QA pairs...━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m (0/9) [33m0:00:08[0m
    [2KProcessing batch 1/1 with 9 chunks━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m (0/9) [33m0:00:08[0m[?25lGenerating QA pairs [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m [33m0:00:00[0m [36m-:--:--[0m
    Generating qa content [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m (0/9) [33m0:00:08[0mINFO:synthetic_data_kit.models.llm_client:Processing batch 1/1 with 9 requests
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating qa content [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m (0/9) [33m0:02:49[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating QA pairs [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m [33m0:05:07[0m [36m-:--:--[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating qa content [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m (0/9) [33m0:08:05[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating qa content [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m (0/9) [33m0:10:55[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating qa content [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m (0/9) [33m0:13:44[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating qa content [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m (0/9) [33m0:16:18[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating qa content [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m (0/9) [33m0:19:07[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating qa content [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m (0/9) [33m0:21:57[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating qa content [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m (0/9) [33m0:24:45[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    [2KParsing response of length 15569
    [2KSuccessfully parsed 20 QA pairs━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m (0/9) [33m0:24:45[0m
    [2K  Generated 20 pairs from chunk 1 (total: 20/167)━━━━━━━━━━━━━[0m [35m  0%[0m (0/9) [33m0:24:45[0m
    [2KParsing response of length 14043━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m (0/9) [33m0:24:45[0m
    [2KSuccessfully parsed 20 QA pairs━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m (0/9) [33m0:24:45[0m
    [2K  Generated 20 pairs from chunk 2 (total: 40/167)━━━━━━━━━━━━━[0m [35m  0%[0m (0/9) [33m0:24:45[0m
    [2KParsing response of length 15984━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m (0/9) [33m0:24:45[0m
    [2KFalling back to regex pattern matching━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m (0/9) [33m0:24:45[0m
    [2KExtracted 19 QA pairs with regex━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m (0/9) [33m0:24:45[0m
    [2K  Generated 19 pairs from chunk 3 (total: 59/167)━━━━━━━━━━━━━[0m [35m  0%[0m (0/9) [33m0:24:45[0m
    [2KParsing response of length 15981━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m (0/9) [33m0:24:45[0m
    [2KFalling back to regex pattern matching━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m (0/9) [33m0:24:45[0m
    [2KExtracted 18 QA pairs with regex━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m (0/9) [33m0:24:45[0m
    [2K  Generated 18 pairs from chunk 4 (total: 77/167)━━━━━━━━━━━━━[0m [35m  0%[0m (0/9) [33m0:24:45[0m
    [2KParsing response of length 14853━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m (0/9) [33m0:24:45[0m
    [2KFalling back to regex pattern matching━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m (0/9) [33m0:24:45[0m
    [2KExtracted 20 QA pairs with regex━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m (0/9) [33m0:24:45[0m
    [2K  Generated 20 pairs from chunk 5 (total: 97/167)━━━━━━━━━━━━━[0m [35m  0%[0m (0/9) [33m0:24:45[0m
    [2KParsing response of length 14387━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m (0/9) [33m0:24:45[0m
    [2KSuccessfully parsed 20 QA pairs━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m (0/9) [33m0:24:45[0m
    [2K  Generated 20 pairs from chunk 6 (total: 117/167)━━━━━━━━━━━━[0m [35m  0%[0m (0/9) [33m0:24:45[0m
    [2KParsing response of length 16240━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m (0/9) [33m0:24:45[0m
    [2KFalling back to regex pattern matching━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m (0/9) [33m0:24:45[0m
    [2KExtracted 18 QA pairs with regex━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m (0/9) [33m0:24:45[0m
    [2K  Generated 18 pairs from chunk 7 (total: 135/167)━━━━━━━━━━━━[0m [35m  0%[0m (0/9) [33m0:24:45[0m
    [2KParsing response of length 17460━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m (0/9) [33m0:24:45[0m
    [2KFalling back to regex pattern matching━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m (0/9) [33m0:24:45[0m
    [2KExtracted 18 QA pairs with regex━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m (0/9) [33m0:24:45[0m
    [2K  Generated 18 pairs from chunk 8 (total: 153/167)━━━━━━━━━━━━[0m [35m  0%[0m (0/9) [33m0:24:45[0m
    [2KParsing response of length 15763━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m (0/9) [33m0:24:45[0m
    [2KFalling back to regex pattern matching━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m (0/9) [33m0:24:45[0m
    [2KExtracted 16 QA pairs with regex━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m (0/9) [33m0:24:45[0m
    [2K  Generated 14 pairs from chunk 9 (total: 167/167)━━━━━━━━━━━━[0m [35m  0%[0m (0/9) [33m0:24:45[0m
    [2KGenerating QA pairs [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m [33m0:24:37[0m [36m-:--:--[0m
    [2KGenerated 167 QA pairs total (requested: 167)
    [2KSaving result to data/generated/Circular Arrangement MCQ [Free PDF] - Objective  (0/9) [33m0:24:45[0m
    Question Answer for Circular Arrangement Quiz - Download Now!_qa_pairs.json
    [2KSuccessfully wrote test file to data/generated/test_write.json[0m [35m  0%[0m (0/9) [33m0:24:45[0m
    [2KSuccessfully wrote result to data/generated/Circular Arrangement MCQ [Free PDF]  (0/9) [33m0:24:45[0m
    - Objective Question Answer for Circular Arrangement Quiz - Download 
    Now!_qa_pairs.json
    [2K[32m✓ Generated qa from Circular Arrangement MCQ [0m[1;32m[[0m[32mFree PDF[0m[1;32m][0m[32m - Objective Question [0m
    [32mAnswer for Circular Arrangement Quiz - Download Now!.txt -> Circular Arrangement[0m
    [32mMCQ [0m[1;32m[[0m[32mFree PDF[0m[1;32m][0m[32m - Objective Question Answer for Circular Arrangement Quiz - [0m
    [32mDownload Now!_qa_pairs.json[0m
    [2KLoading config from: ../tutorial_config.yaml━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m[32m [0m[35m  0%[0m[32m [0m[32m(0/9)[0m[32m [0m[33m0:24:45[0m
    [2KConfig has LLM provider set to: vllm━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m (0/9) [33m0:24:45[0m
    [2KL Using vllm provider0m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m (0/9) [33m0:24:45[0m
    [2KLoading config from: ../tutorial_config.yaml━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m (0/9) [33m0:24:45[0m
    [2KConfig has LLM provider set to: vllm━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m (0/9) [33m0:24:45[0m
    [2KGenerating document summary...━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m (0/9) [33m0:24:45[0m
    Generating qa content [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m (0/9) [33m0:24:45[0mINFO:synthetic_data_kit.models.llm_client:Sending request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating qa content [91m━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 11%[0m (1/9) [33m0:24:51[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    [2KSummary generated (875 chars)
    [2KGenerating QA pairs...m━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 11%[0m (1/9) [33m0:24:51[0m
    [2KDocument split into 3 chunks[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 11%[0m (1/9) [33m0:24:51[0m
    [2KUsing batch size of 16m━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 11%[0m (1/9) [33m0:24:51[0m
    [2KProcessing 3 chunks to generate QA pairs...90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 11%[0m (1/9) [33m0:24:51[0m
    [2KProcessing batch 1/1 with 3 chunks0m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 11%[0m (1/9) [33m0:24:51[0m[?25lGenerating QA pairs [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m [33m0:00:00[0m [36m-:--:--[0m
    Generating qa content [91m━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 11%[0m (1/9) [33m0:24:51[0mINFO:synthetic_data_kit.models.llm_client:Processing batch 1/1 with 3 requests
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating QA pairs [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m [33m0:02:49[0m [36m-:--:--[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating QA pairs [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m [33m0:05:39[0m [36m-:--:--[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating qa content [91m━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 11%[0m (1/9) [33m0:33:19[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    [2KParsing response of length 17745
    [2KFalling back to regex pattern matching[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 11%[0m (1/9) [33m0:33:19[0m
    [2KExtracted 32 QA pairs with regex[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 11%[0m (1/9) [33m0:33:19[0m
    [2K  Generated 32 pairs from chunk 1 (total: 32/167)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 11%[0m (1/9) [33m0:33:19[0m
    [2KParsing response of length 16990[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 11%[0m (1/9) [33m0:33:19[0m
    [2KFalling back to regex pattern matching[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 11%[0m (1/9) [33m0:33:19[0m
    [2KExtracted 18 QA pairs with regex[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 11%[0m (1/9) [33m0:33:19[0m
    [2K  Generated 18 pairs from chunk 2 (total: 50/167)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 11%[0m (1/9) [33m0:33:19[0m
    [2KParsing response of length 15078[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 11%[0m (1/9) [33m0:33:19[0m
    [2KFalling back to regex pattern matching[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 11%[0m (1/9) [33m0:33:19[0m
    [2KExtracted 27 QA pairs with regex[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 11%[0m (1/9) [33m0:33:19[0m
    [2K  Generated 27 pairs from chunk 3 (total: 77/167)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 11%[0m (1/9) [33m0:33:19[0m
    [2KGenerating QA pairs [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m [33m0:08:27[0m [36m-:--:--[0m9[0m
    [2KGenerated 77 QA pairs total (requested: 167)
    [2KSaving result to data/generated/SA_Set_No_173_qa_pairs.json━━━━━━━━━━━━━━━━━━━━━[0m [35m 11%[0m (1/9) [33m0:33:19[0m
    [2KSuccessfully wrote test file to data/generated/test_write.json━━━━━━━━━━━━━━━━━━[0m [35m 11%[0m (1/9) [33m0:33:19[0m
    [2KSuccessfully wrote result to data/generated/SA_Set_No_173_qa_pairs.json━━━━━━━━━[0m [35m 11%[0m (1/9) [33m0:33:19[0m
    [2K[32m✓ Generated qa from SA_Set_No_173.txt -> SA_Set_No_173_qa_pairs.json[0m━━━[0m [35m 11%[0m (1/9) [33m0:33:19[0m
    [2KLoading config from: ../tutorial_config.yaml━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m[32m [0m[35m 11%[0m[32m [0m[32m(1/9)[0m[32m [0m[33m0:33:19[0m
    [2KConfig has LLM provider set to: vllm╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 11%[0m (1/9) [33m0:33:19[0m
    [2KL Using vllm provider1m━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 11%[0m (1/9) [33m0:33:19[0m
    [2KLoading config from: ../tutorial_config.yaml0m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 11%[0m (1/9) [33m0:33:19[0m
    [2KConfig has LLM provider set to: vllm╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 11%[0m (1/9) [33m0:33:19[0m
    [2KGenerating document summary...m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 11%[0m (1/9) [33m0:33:19[0m
    Generating qa content [91m━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 11%[0m (1/9) [33m0:33:19[0mINFO:synthetic_data_kit.models.llm_client:Sending request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating qa content [91m━━━━━━━━[0m[91m╸[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m0:33:24[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    [2KSummary generated (658 chars)
    [2KGenerating QA pairs...m━━━━━━━━[0m[91m╸[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m0:33:24[0m
    [2KDocument split into 33 chunks━━[0m[91m╸[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m0:33:24[0m
    [2KUsing batch size of 16m━━━━━━━━[0m[91m╸[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m0:33:24[0m
    [2KProcessing 33 chunks to generate QA pairs...m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m0:33:24[0m
    [2KProcessing batch 1/3 with 16 chunks[91m╸[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m0:33:24[0m[?25lGenerating QA pairs [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m [33m0:00:00[0m [36m-:--:--[0m
    Generating qa content [91m━━━━━━━━[0m[91m╸[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m0:33:24[0mINFO:synthetic_data_kit.models.llm_client:Processing batch 1/1 with 16 requests
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating QA pairs [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m [33m0:01:06[0m [36m-:--:--[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating QA pairs [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m [33m0:01:56[0m [36m-:--:--[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating QA pairs [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m [33m0:02:34[0m [36m-:--:--[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating qa content [91m━━━━━━━━[0m[91m╸[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m0:36:51[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating qa content [91m━━━━━━━━[0m[91m╸[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m0:37:39[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating qa content [91m━━━━━━━━[0m[91m╸[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m0:38:30[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating qa content [91m━━━━━━━━[0m[91m╸[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m0:39:21[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating qa content [91m━━━━━━━━[0m[91m╸[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m0:40:12[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating QA pairs [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m [33m0:07:32[0m [36m-:--:--[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating QA pairs [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m [33m0:08:22[0m [36m-:--:--[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating qa content [91m━━━━━━━━[0m[91m╸[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m0:42:30[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating qa content [91m━━━━━━━━[0m[91m╸[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m0:43:25[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating qa content [91m━━━━━━━━[0m[91m╸[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m0:44:29[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating qa content [91m━━━━━━━━[0m[91m╸[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m0:45:42[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating qa content [91m━━━━━━━━[0m[91m╸[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m0:46:42[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating qa content [91m━━━━━━━━[0m[91m╸[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m0:47:35[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    [2KParsing response of length 6703
    [2KSuccessfully parsed 5 QA pairs━[0m[91m╸[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m0:47:35[0m
    [2K  Generated 5 pairs from chunk 1 (total: 5/167)90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m0:47:35[0m
    [2KParsing response of length 4542[0m[91m╸[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m0:47:35[0m
    [2KSuccessfully parsed 5 QA pairs━[0m[91m╸[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m0:47:35[0m
    [2K  Generated 5 pairs from chunk 2 (total: 10/167)0m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m0:47:35[0m
    [2KParsing response of length 3653[0m[91m╸[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m0:47:35[0m
    [2KSuccessfully parsed 5 QA pairs━[0m[91m╸[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m0:47:35[0m
    [2K  Generated 5 pairs from chunk 3 (total: 15/167)0m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m0:47:35[0m
    [2KParsing response of length 5571[0m[91m╸[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m0:47:35[0m
    [2KSuccessfully parsed 5 QA pairs━[0m[91m╸[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m0:47:35[0m
    [2K  Generated 5 pairs from chunk 4 (total: 20/167)0m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m0:47:35[0m
    [2KParsing response of length 4870[0m[91m╸[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m0:47:35[0m
    [2KSuccessfully parsed 5 QA pairs━[0m[91m╸[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m0:47:35[0m
    [2K  Generated 5 pairs from chunk 5 (total: 25/167)0m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m0:47:35[0m
    [2KParsing response of length 4335[0m[91m╸[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m0:47:35[0m
    [2KSuccessfully parsed 5 QA pairs━[0m[91m╸[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m0:47:35[0m
    [2K  Generated 5 pairs from chunk 6 (total: 30/167)0m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m0:47:35[0m
    [2KParsing response of length 4685[0m[91m╸[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m0:47:35[0m
    [2KSuccessfully parsed 5 QA pairs━[0m[91m╸[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m0:47:35[0m
    [2K  Generated 5 pairs from chunk 7 (total: 35/167)0m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m0:47:35[0m
    [2KParsing response of length 4884[0m[91m╸[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m0:47:35[0m
    [2KSuccessfully parsed 5 QA pairs━[0m[91m╸[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m0:47:35[0m
    [2K  Generated 5 pairs from chunk 8 (total: 40/167)0m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m0:47:35[0m
    [2KParsing response of length 4282[0m[91m╸[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m0:47:35[0m
    [2KSuccessfully parsed 5 QA pairs━[0m[91m╸[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m0:47:35[0m
    [2K  Generated 5 pairs from chunk 9 (total: 45/167)0m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m0:47:35[0m
    [2KParsing response of length 5115[0m[91m╸[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m0:47:35[0m
    [2KSuccessfully parsed 5 QA pairs━[0m[91m╸[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m0:47:35[0m
    [2K  Generated 5 pairs from chunk 10 (total: 50/167)m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m0:47:35[0m
    [2KParsing response of length 4106[0m[91m╸[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m0:47:35[0m
    [2KSuccessfully parsed 5 QA pairs━[0m[91m╸[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m0:47:35[0m
    [2K  Generated 5 pairs from chunk 11 (total: 55/167)m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m0:47:35[0m
    [2KParsing response of length 4594[0m[91m╸[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m0:47:35[0m
    [2KSuccessfully parsed 5 QA pairs━[0m[91m╸[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m0:47:35[0m
    [2K  Generated 5 pairs from chunk 12 (total: 60/167)m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m0:47:35[0m
    [2KParsing response of length 6647[0m[91m╸[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m0:47:35[0m
    [2KSuccessfully parsed 5 QA pairs━[0m[91m╸[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m0:47:35[0m
    [2K  Generated 5 pairs from chunk 13 (total: 65/167)m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m0:47:35[0m
    [2KParsing response of length 7527[0m[91m╸[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m0:47:35[0m
    [2KSuccessfully parsed 5 QA pairs━[0m[91m╸[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m0:47:35[0m
    [2K  Generated 5 pairs from chunk 14 (total: 70/167)m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m0:47:35[0m
    [2KParsing response of length 6200[0m[91m╸[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m0:47:35[0m
    [2KSuccessfully parsed 5 QA pairs━[0m[91m╸[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m0:47:35[0m
    [2K  Generated 5 pairs from chunk 15 (total: 75/167)m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m0:47:35[0m
    [2KParsing response of length 5196[0m[91m╸[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m0:47:35[0m
    [2KSuccessfully parsed 5 QA pairs━[0m[91m╸[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m0:47:35[0m
    [2K  Generated 5 pairs from chunk 16 (total: 80/167)m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m0:47:35[0m
    [2KProcessing batch 2/3 with 16 chunks[91m╸[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m0:47:35[0m
    Generating qa content [91m━━━━━━━━[0m[91m╸[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m0:47:35[0mINFO:synthetic_data_kit.models.llm_client:Processing batch 1/1 with 16 requests
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating qa content [91m━━━━━━━━[0m[91m╸[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m0:48:15[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating qa content [91m━━━━━━━━[0m[91m╸[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m0:49:01[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating qa content [91m━━━━━━━━[0m[91m╸[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m0:50:06[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating qa content [91m━━━━━━━━[0m[91m╸[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m0:51:10[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating QA pairs [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m [33m0:18:39[0m [36m-:--:--[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating qa content [91m━━━━━━━━[0m[91m╸[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m0:52:44[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating qa content [91m━━━━━━━━[0m[91m╸[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m0:53:43[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating qa content [91m━━━━━━━━[0m[91m╸[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m0:54:29[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating qa content [91m━━━━━━━━[0m[91m╸[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m0:55:15[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating QA pairs [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m [33m0:22:36[0m [36m-:--:--[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating qa content [91m━━━━━━━━[0m[91m╸[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m0:56:41[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating qa content [91m━━━━━━━━[0m[91m╸[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m0:57:21[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating qa content [91m━━━━━━━━[0m[91m╸[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m0:58:08[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating qa content [91m━━━━━━━━[0m[91m╸[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m0:58:50[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating qa content [91m━━━━━━━━[0m[91m╸[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m0:59:34[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating qa content [91m━━━━━━━━[0m[91m╸[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m1:00:20[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    [2KParsing response of length 4270
    [2KSuccessfully parsed 5 QA pairs━[0m[91m╸[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m1:00:20[0m
    [2K  Generated 5 pairs from chunk 17 (total: 85/167)m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m1:00:20[0m
    [2KParsing response of length 4614[0m[91m╸[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m1:00:20[0m
    [2KSuccessfully parsed 5 QA pairs━[0m[91m╸[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m1:00:20[0m
    [2K  Generated 5 pairs from chunk 18 (total: 90/167)m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m1:00:20[0m
    [2KParsing response of length 6247[0m[91m╸[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m1:00:20[0m
    [2KSuccessfully parsed 5 QA pairs━[0m[91m╸[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m1:00:20[0m
    [2K  Generated 5 pairs from chunk 19 (total: 95/167)m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m1:00:20[0m
    [2KParsing response of length 6502[0m[91m╸[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m1:00:20[0m
    [2KSuccessfully parsed 5 QA pairs━[0m[91m╸[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m1:00:20[0m
    [2K  Generated 5 pairs from chunk 20 (total: 100/167)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m1:00:20[0m
    [2KParsing response of length 5606[0m[91m╸[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m1:00:20[0m
    [2KSuccessfully parsed 5 QA pairs━[0m[91m╸[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m1:00:20[0m
    [2K  Generated 5 pairs from chunk 21 (total: 105/167)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m1:00:20[0m
    [2KParsing response of length 4183[0m[91m╸[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m1:00:20[0m
    [2KSuccessfully parsed 5 QA pairs━[0m[91m╸[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m1:00:20[0m
    [2K  Generated 5 pairs from chunk 22 (total: 110/167)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m1:00:20[0m
    [2KParsing response of length 5714[0m[91m╸[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m1:00:20[0m
    [2KSuccessfully parsed 5 QA pairs━[0m[91m╸[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m1:00:20[0m
    [2K  Generated 5 pairs from chunk 23 (total: 115/167)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m1:00:20[0m
    [2KParsing response of length 4739[0m[91m╸[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m1:00:20[0m
    [2KSuccessfully parsed 5 QA pairs━[0m[91m╸[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m1:00:20[0m
    [2K  Generated 5 pairs from chunk 24 (total: 120/167)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m1:00:20[0m
    [2KParsing response of length 4579[0m[91m╸[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m1:00:20[0m
    [2KSuccessfully parsed 5 QA pairs━[0m[91m╸[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m1:00:20[0m
    [2K  Generated 5 pairs from chunk 25 (total: 125/167)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m1:00:20[0m
    [2KParsing response of length 4313[0m[91m╸[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m1:00:20[0m
    [2KSuccessfully parsed 5 QA pairs━[0m[91m╸[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m1:00:20[0m
    [2K  Generated 5 pairs from chunk 26 (total: 130/167)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m1:00:20[0m
    [2KParsing response of length 4355[0m[91m╸[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m1:00:20[0m
    [2KSuccessfully parsed 5 QA pairs━[0m[91m╸[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m1:00:20[0m
    [2K  Generated 5 pairs from chunk 27 (total: 135/167)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m1:00:20[0m
    [2KParsing response of length 4132[0m[91m╸[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m1:00:20[0m
    [2KSuccessfully parsed 5 QA pairs━[0m[91m╸[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m1:00:20[0m
    [2K  Generated 5 pairs from chunk 28 (total: 140/167)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m1:00:20[0m
    [2KParsing response of length 4964[0m[91m╸[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m1:00:20[0m
    [2KSuccessfully parsed 5 QA pairs━[0m[91m╸[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m1:00:20[0m
    [2K  Generated 5 pairs from chunk 29 (total: 145/167)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m1:00:20[0m
    [2KParsing response of length 4152[0m[91m╸[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m1:00:20[0m
    [2KSuccessfully parsed 5 QA pairs━[0m[91m╸[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m1:00:20[0m
    [2K  Generated 5 pairs from chunk 30 (total: 150/167)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m1:00:20[0m
    [2KParsing response of length 4318[0m[91m╸[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m1:00:20[0m
    [2KSuccessfully parsed 5 QA pairs━[0m[91m╸[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m1:00:20[0m
    [2K  Generated 5 pairs from chunk 31 (total: 155/167)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m1:00:20[0m
    [2KParsing response of length 4628[0m[91m╸[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m1:00:20[0m
    [2KSuccessfully parsed 5 QA pairs━[0m[91m╸[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m1:00:20[0m
    [2K  Generated 5 pairs from chunk 32 (total: 160/167)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m1:00:20[0m
    [2KProcessing batch 3/3 with 1 chunksm[91m╸[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m1:00:20[0m
    Generating qa content [91m━━━━━━━━[0m[91m╸[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m1:00:20[0mINFO:synthetic_data_kit.models.llm_client:Processing batch 1/1 with 1 requests
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating qa content [91m━━━━━━━━[0m[91m╸[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m1:01:06[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    [2KParsing response of length 4665
    [2KSuccessfully parsed 5 QA pairs━[0m[91m╸[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m1:01:06[0m
    [2K  Generated 5 pairs from chunk 33 (total: 165/167)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m1:01:06[0m
    [2KGenerating QA pairs [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m [33m0:27:41[0m [36m-:--:--[0m6[0m
    [2KGenerated 165 QA pairs total (requested: 167)
    [2KSaving result to data/generated/Seating_Arrangement_PDF_1_qa_pairs.json━━━━━━━━━[0m [35m 22%[0m (2/9) [33m1:01:06[0m
    [2KSuccessfully wrote test file to data/generated/test_write.json━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m1:01:06[0m
    [2KSuccessfully wrote result to ━━[0m[91m╸[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m1:01:06[0m
    data/generated/Seating_Arrangement_PDF_1_qa_pairs.json
    [2K[32m✓ Generated qa from Seating_Arrangement_PDF_1.txt -> [0m━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m1:01:06[0m
    [32mSeating_Arrangement_PDF_1_qa_pairs.json[0m
    [2KLoading config from: ../tutorial_config.yaml━━━━━[0m[91m╸[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m[32m [0m[35m 22%[0m[32m [0m[32m(2/9)[0m[32m [0m[33m1:01:06[0m
    [2KConfig has LLM provider set to: vllm[91m╸[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m1:01:06[0m
    [2KL Using vllm provider1m━━━━━━━━[0m[91m╸[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m1:01:06[0m
    [2KLoading config from: ../tutorial_config.yamlm[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m1:01:06[0m
    [2KConfig has LLM provider set to: vllm[91m╸[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m1:01:06[0m
    [2KGenerating document summary...━[0m[91m╸[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m1:01:06[0m
    Generating qa content [91m━━━━━━━━[0m[91m╸[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m1:01:06[0mINFO:synthetic_data_kit.models.llm_client:Sending request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating qa content [91m━━━━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:01:12[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    [2KSummary generated (827 chars)
    [2KGenerating QA pairs...m━━━━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:01:12[0m
    [2KDocument split into 73 chunks━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:01:12[0m
    [2KUsing batch size of 16m━━━━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:01:12[0m
    [2KProcessing 73 chunks to generate QA pairs...m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:01:12[0m
    [2KProcessing batch 1/5 with 16 chunks━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:01:12[0m[?25lGenerating QA pairs [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m [33m0:00:00[0m [36m-:--:--[0m
    Generating qa content [91m━━━━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:01:12[0mINFO:synthetic_data_kit.models.llm_client:Processing batch 1/1 with 16 requests
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating QA pairs [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m [33m0:00:36[0m [36m-:--:--[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating QA pairs [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m [33m0:00:56[0m [36m-:--:--[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating QA pairs [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m [33m0:01:16[0m [36m-:--:--[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating qa content [91m━━━━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:02:58[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating QA pairs [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m [33m0:02:08[0m [36m-:--:--[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating QA pairs [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m [33m0:02:28[0m [36m-:--:--[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating qa content [91m━━━━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:04:01[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating QA pairs [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m [33m0:03:11[0m [36m-:--:--[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating QA pairs [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m [33m0:03:38[0m [36m-:--:--[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating QA pairs [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m [33m0:04:02[0m [36m-:--:--[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating QA pairs [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m [33m0:04:26[0m [36m-:--:--[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating qa content [91m━━━━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:05:57[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating qa content [91m━━━━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:06:17[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating QA pairs [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m [33m0:05:24[0m [36m-:--:--[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating QA pairs [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m [33m0:05:44[0m [36m-:--:--[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating QA pairs [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m [33m0:06:13[0m [36m-:--:--[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    [2KParsing response of length 3660
    [2KSuccessfully parsed 2 QA pairs━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:07:26[0m
    [2K  Generated 2 pairs from chunk 1 (total: 2/167)[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:07:26[0m
    [2KParsing response of length 1931━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:07:26[0m
    [2KSuccessfully parsed 2 QA pairs━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:07:26[0m
    [2K  Generated 2 pairs from chunk 2 (total: 4/167)[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:07:26[0m
    [2KParsing response of length 1847━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:07:26[0m
    [2KSuccessfully parsed 2 QA pairs━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:07:26[0m
    [2K  Generated 2 pairs from chunk 3 (total: 6/167)[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:07:26[0m
    [2KParsing response of length 2794━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:07:26[0m
    [2KSuccessfully parsed 2 QA pairs━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:07:26[0m
    [2K  Generated 2 pairs from chunk 4 (total: 8/167)[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:07:26[0m
    [2KParsing response of length 2086━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:07:26[0m
    [2KSuccessfully parsed 2 QA pairs━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:07:26[0m
    [2K  Generated 2 pairs from chunk 5 (total: 10/167)0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:07:26[0m
    [2KParsing response of length 1855━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:07:26[0m
    [2KSuccessfully parsed 2 QA pairs━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:07:26[0m
    [2K  Generated 2 pairs from chunk 6 (total: 12/167)0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:07:26[0m
    [2KParsing response of length 1704━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:07:26[0m
    [2KSuccessfully parsed 2 QA pairs━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:07:26[0m
    [2K  Generated 2 pairs from chunk 7 (total: 14/167)0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:07:26[0m
    [2KParsing response of length 2249━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:07:26[0m
    [2KSuccessfully parsed 2 QA pairs━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:07:26[0m
    [2K  Generated 2 pairs from chunk 8 (total: 16/167)0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:07:26[0m
    [2KParsing response of length 2752━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:07:26[0m
    [2KSuccessfully parsed 2 QA pairs━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:07:26[0m
    [2K  Generated 2 pairs from chunk 9 (total: 18/167)m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:07:26[0m
    [2KParsing response of length 2316━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:07:26[0m
    [2KSuccessfully parsed 2 QA pairs━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:07:26[0m
    [2K  Generated 2 pairs from chunk 10 (total: 20/167)m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:07:26[0m
    [2KParsing response of length 2347━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:07:26[0m
    [2KSuccessfully parsed 2 QA pairs━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:07:26[0m
    [2K  Generated 2 pairs from chunk 11 (total: 22/167)m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:07:26[0m
    [2KParsing response of length 2033━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:07:26[0m
    [2KSuccessfully parsed 2 QA pairs━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:07:26[0m
    [2K  Generated 2 pairs from chunk 12 (total: 24/167)m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:07:26[0m
    [2KParsing response of length 1854━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:07:26[0m
    [2KSuccessfully parsed 2 QA pairs━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:07:26[0m
    [2K  Generated 2 pairs from chunk 13 (total: 26/167)m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:07:26[0m
    [2KParsing response of length 1931━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:07:26[0m
    [2KSuccessfully parsed 2 QA pairs━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:07:26[0m
    [2K  Generated 2 pairs from chunk 14 (total: 28/167)m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:07:26[0m
    [2KParsing response of length 1983━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:07:26[0m
    [2KSuccessfully parsed 2 QA pairs━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:07:26[0m
    [2K  Generated 2 pairs from chunk 15 (total: 30/167)m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:07:26[0m
    [2KParsing response of length 2794━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:07:26[0m
    [2KSuccessfully parsed 2 QA pairs━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:07:26[0m
    [2K  Generated 2 pairs from chunk 16 (total: 32/167)m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:07:26[0m
    [2KProcessing batch 2/5 with 16 chunks━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:07:26[0m
    Generating qa content [91m━━━━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:07:26[0mINFO:synthetic_data_kit.models.llm_client:Processing batch 1/1 with 16 requests
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating qa content [91m━━━━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:07:48[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating QA pairs [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m [33m0:06:58[0m [36m-:--:--[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating QA pairs [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m [33m0:07:18[0m [36m-:--:--[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating qa content [91m━━━━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:09:04[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating qa content [91m━━━━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:09:30[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating qa content [91m━━━━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:09:54[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating qa content [91m━━━━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:10:18[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating QA pairs [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m [33m0:09:27[0m [36m-:--:--[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating QA pairs [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m [33m0:09:55[0m [36m-:--:--[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating qa content [91m━━━━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:11:28[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating qa content [91m━━━━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:11:47[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating QA pairs [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m [33m0:10:55[0m [36m-:--:--[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating qa content [91m━━━━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:12:32[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating QA pairs [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m [33m0:11:47[0m [36m-:--:--[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating qa content [91m━━━━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:13:36[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating QA pairs [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m [33m0:12:43[0m [36m-:--:--[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    [2KParsing response of length 2192
    [2KSuccessfully parsed 2 QA pairs━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:13:55[0m
    [2K  Generated 2 pairs from chunk 17 (total: 34/167)m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:13:55[0m
    [2KParsing response of length 2142━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:13:55[0m
    [2KSuccessfully parsed 2 QA pairs━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:13:55[0m
    [2K  Generated 2 pairs from chunk 18 (total: 36/167)m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:13:55[0m
    [2KParsing response of length 1984━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:13:55[0m
    [2KSuccessfully parsed 2 QA pairs━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:13:55[0m
    [2K  Generated 2 pairs from chunk 19 (total: 38/167)m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:13:55[0m
    [2KParsing response of length 3423━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:13:55[0m
    [2KSuccessfully parsed 2 QA pairs━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:13:55[0m
    [2K  Generated 2 pairs from chunk 20 (total: 40/167)m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:13:55[0m
    [2KParsing response of length 2812━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:13:55[0m
    [2KSuccessfully parsed 2 QA pairs━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:13:55[0m
    [2K  Generated 2 pairs from chunk 21 (total: 42/167)m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:13:55[0m
    [2KParsing response of length 2148━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:13:55[0m
    [2KSuccessfully parsed 2 QA pairs━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:13:55[0m
    [2K  Generated 2 pairs from chunk 22 (total: 44/167)m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:13:55[0m
    [2KParsing response of length 2282━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:13:55[0m
    [2KSuccessfully parsed 2 QA pairs━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:13:55[0m
    [2K  Generated 2 pairs from chunk 23 (total: 46/167)m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:13:55[0m
    [2KParsing response of length 1917━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:13:55[0m
    [2KSuccessfully parsed 2 QA pairs━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:13:55[0m
    [2K  Generated 2 pairs from chunk 24 (total: 48/167)m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:13:55[0m
    [2KParsing response of length 2845━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:13:55[0m
    [2KSuccessfully parsed 2 QA pairs━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:13:55[0m
    [2K  Generated 2 pairs from chunk 25 (total: 50/167)m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:13:55[0m
    [2KParsing response of length 1900━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:13:55[0m
    [2KSuccessfully parsed 2 QA pairs━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:13:55[0m
    [2K  Generated 2 pairs from chunk 26 (total: 52/167)m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:13:55[0m
    [2KParsing response of length 1736━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:13:55[0m
    [2KSuccessfully parsed 2 QA pairs━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:13:55[0m
    [2K  Generated 2 pairs from chunk 27 (total: 54/167)m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:13:55[0m
    [2KParsing response of length 1933━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:13:55[0m
    [2KSuccessfully parsed 2 QA pairs━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:13:55[0m
    [2K  Generated 2 pairs from chunk 28 (total: 56/167)m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:13:55[0m
    [2KParsing response of length 2322━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:13:55[0m
    [2KSuccessfully parsed 2 QA pairs━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:13:55[0m
    [2K  Generated 2 pairs from chunk 29 (total: 58/167)m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:13:55[0m
    [2KParsing response of length 2595━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:13:55[0m
    [2KSuccessfully parsed 2 QA pairs━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:13:55[0m
    [2K  Generated 2 pairs from chunk 30 (total: 60/167)m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:13:55[0m
    [2KParsing response of length 3476━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:13:55[0m
    [2KSuccessfully parsed 2 QA pairs━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:13:55[0m
    [2K  Generated 2 pairs from chunk 31 (total: 62/167)m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:13:55[0m
    [2KParsing response of length 1777━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:13:55[0m
    [2KSuccessfully parsed 2 QA pairs━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:13:55[0m
    [2K  Generated 2 pairs from chunk 32 (total: 64/167)m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:13:55[0m
    [2KProcessing batch 3/5 with 16 chunks━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:13:55[0m
    Generating qa content [91m━━━━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:13:55[0mINFO:synthetic_data_kit.models.llm_client:Processing batch 1/1 with 16 requests
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating QA pairs [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m [33m0:13:02[0m [36m-:--:--[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating QA pairs [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m [33m0:13:28[0m [36m-:--:--[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating QA pairs [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m [33m0:13:47[0m [36m-:--:--[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating QA pairs [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m [33m0:14:08[0m [36m-:--:--[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating QA pairs [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m [33m0:14:38[0m [36m-:--:--[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating QA pairs [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m [33m0:15:05[0m [36m-:--:--[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating QA pairs [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m [33m0:15:28[0m [36m-:--:--[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating QA pairs [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m [33m0:15:50[0m [36m-:--:--[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating QA pairs [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m [33m0:16:16[0m [36m-:--:--[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating QA pairs [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m [33m0:16:37[0m [36m-:--:--[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating QA pairs [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m [33m0:16:57[0m [36m-:--:--[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating QA pairs [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m [33m0:17:14[0m [36m-:--:--[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating QA pairs [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m [33m0:17:40[0m [36m-:--:--[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating QA pairs [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m [33m0:17:57[0m [36m-:--:--[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating QA pairs [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m [33m0:18:15[0m [36m-:--:--[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating QA pairs [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m [33m0:18:38[0m [36m-:--:--[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    [2KParsing response of length 1840
    [2KSuccessfully parsed 2 QA pairs━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:19:51[0m
    [2K  Generated 2 pairs from chunk 33 (total: 66/167)m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:19:51[0m
    [2KParsing response of length 2379━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:19:51[0m
    [2KSuccessfully parsed 2 QA pairs━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:19:51[0m
    [2K  Generated 2 pairs from chunk 34 (total: 68/167)m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:19:51[0m
    [2KParsing response of length 1841━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:19:51[0m
    [2KSuccessfully parsed 2 QA pairs━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:19:51[0m
    [2K  Generated 2 pairs from chunk 35 (total: 70/167)m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:19:51[0m
    [2KParsing response of length 1959━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:19:51[0m
    [2KSuccessfully parsed 2 QA pairs━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:19:51[0m
    [2K  Generated 2 pairs from chunk 36 (total: 72/167)m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:19:51[0m
    [2KParsing response of length 3157━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:19:51[0m
    [2KSuccessfully parsed 2 QA pairs━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:19:51[0m
    [2K  Generated 2 pairs from chunk 37 (total: 74/167)m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:19:51[0m
    [2KParsing response of length 2464━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:19:51[0m
    [2KSuccessfully parsed 2 QA pairs━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:19:51[0m
    [2K  Generated 2 pairs from chunk 38 (total: 76/167)m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:19:51[0m
    [2KParsing response of length 2124━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:19:51[0m
    [2KSuccessfully parsed 2 QA pairs━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:19:51[0m
    [2K  Generated 2 pairs from chunk 39 (total: 78/167)m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:19:51[0m
    [2KParsing response of length 2126━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:19:51[0m
    [2KSuccessfully parsed 2 QA pairs━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:19:51[0m
    [2K  Generated 2 pairs from chunk 40 (total: 80/167)m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:19:51[0m
    [2KParsing response of length 2473━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:19:51[0m
    [2KSuccessfully parsed 2 QA pairs━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:19:51[0m
    [2K  Generated 2 pairs from chunk 41 (total: 82/167)m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:19:51[0m
    [2KParsing response of length 2056━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:19:51[0m
    [2KSuccessfully parsed 2 QA pairs━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:19:51[0m
    [2K  Generated 2 pairs from chunk 42 (total: 84/167)m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:19:51[0m
    [2KParsing response of length 2069━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:19:51[0m
    [2KSuccessfully parsed 2 QA pairs━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:19:51[0m
    [2K  Generated 2 pairs from chunk 43 (total: 86/167)m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:19:51[0m
    [2KParsing response of length 1684━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:19:51[0m
    [2KSuccessfully parsed 2 QA pairs━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:19:51[0m
    [2K  Generated 2 pairs from chunk 44 (total: 88/167)m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:19:51[0m
    [2KParsing response of length 2792━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:19:51[0m
    [2KSuccessfully parsed 2 QA pairs━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:19:51[0m
    [2K  Generated 2 pairs from chunk 45 (total: 90/167)m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:19:51[0m
    [2KParsing response of length 1556━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:19:51[0m
    [2KSuccessfully parsed 2 QA pairs━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:19:51[0m
    [2K  Generated 2 pairs from chunk 46 (total: 92/167)m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:19:51[0m
    [2KParsing response of length 1873━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:19:51[0m
    [2KSuccessfully parsed 2 QA pairs━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:19:51[0m
    [2K  Generated 2 pairs from chunk 47 (total: 94/167)m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:19:51[0m
    [2KParsing response of length 2023━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:19:51[0m
    [2KSuccessfully parsed 2 QA pairs━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:19:51[0m
    [2K  Generated 2 pairs from chunk 48 (total: 96/167)m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:19:51[0m
    [2KProcessing batch 4/5 with 16 chunks━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:19:51[0m
    Generating qa content [91m━━━━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:19:51[0mINFO:synthetic_data_kit.models.llm_client:Processing batch 1/1 with 16 requests
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating QA pairs [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m [33m0:19:15[0m [36m-:--:--[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating QA pairs [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m [33m0:19:48[0m [36m-:--:--[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating QA pairs [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m [33m0:20:17[0m [36m-:--:--[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating QA pairs [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m [33m0:20:43[0m [36m-:--:--[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating QA pairs [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m [33m0:21:06[0m [36m-:--:--[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating qa content [91m━━━━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:22:46[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating qa content [91m━━━━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:23:05[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating QA pairs [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m [33m0:22:18[0m [36m-:--:--[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating QA pairs [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m [33m0:22:37[0m [36m-:--:--[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating QA pairs [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m [33m0:22:57[0m [36m-:--:--[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating QA pairs [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m [33m0:23:24[0m [36m-:--:--[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating qa content [91m━━━━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:25:01[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating QA pairs [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m [33m0:24:12[0m [36m-:--:--[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating qa content [91m━━━━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:25:47[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating QA pairs [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m [33m0:24:53[0m [36m-:--:--[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating qa content [91m━━━━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:26:27[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    [2KParsing response of length 3723
    [2KSuccessfully parsed 2 QA pairs━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:26:27[0m
    [2K  Generated 2 pairs from chunk 49 (total: 98/167)m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:26:27[0m
    [2KParsing response of length 3224━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:26:27[0m
    [2KSuccessfully parsed 2 QA pairs━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:26:27[0m
    [2K  Generated 2 pairs from chunk 50 (total: 100/167)[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:26:27[0m
    [2KParsing response of length 2713━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:26:27[0m
    [2KSuccessfully parsed 2 QA pairs━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:26:27[0m
    [2K  Generated 2 pairs from chunk 51 (total: 102/167)[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:26:27[0m
    [2KParsing response of length 2624━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:26:27[0m
    [2KSuccessfully parsed 2 QA pairs━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:26:27[0m
    [2K  Generated 2 pairs from chunk 52 (total: 104/167)[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:26:27[0m
    [2KParsing response of length 2296━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:26:27[0m
    [2KSuccessfully parsed 2 QA pairs━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:26:27[0m
    [2K  Generated 2 pairs from chunk 53 (total: 106/167)[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:26:27[0m
    [2KParsing response of length 2753━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:26:27[0m
    [2KSuccessfully parsed 2 QA pairs━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:26:27[0m
    [2K  Generated 2 pairs from chunk 54 (total: 108/167)[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:26:27[0m
    [2KParsing response of length 1798━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:26:27[0m
    [2KSuccessfully parsed 2 QA pairs━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:26:27[0m
    [2K  Generated 2 pairs from chunk 55 (total: 110/167)[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:26:27[0m
    [2KParsing response of length 2361━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:26:27[0m
    [2KSuccessfully parsed 2 QA pairs━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:26:27[0m
    [2K  Generated 2 pairs from chunk 56 (total: 112/167)[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:26:27[0m
    [2KParsing response of length 2167━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:26:27[0m
    [2KSuccessfully parsed 2 QA pairs━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:26:27[0m
    [2K  Generated 2 pairs from chunk 57 (total: 114/167)[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:26:27[0m
    [2KParsing response of length 2040━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:26:27[0m
    [2KSuccessfully parsed 2 QA pairs━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:26:27[0m
    [2K  Generated 2 pairs from chunk 58 (total: 116/167)[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:26:27[0m
    [2KParsing response of length 2916━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:26:27[0m
    [2KSuccessfully parsed 2 QA pairs━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:26:27[0m
    [2K  Generated 2 pairs from chunk 59 (total: 118/167)[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:26:27[0m
    [2KParsing response of length 2444━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:26:27[0m
    [2KSuccessfully parsed 2 QA pairs━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:26:27[0m
    [2K  Generated 2 pairs from chunk 60 (total: 120/167)[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:26:27[0m
    [2KParsing response of length 2190━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:26:27[0m
    [2KSuccessfully parsed 2 QA pairs━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:26:27[0m
    [2K  Generated 2 pairs from chunk 61 (total: 122/167)[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:26:27[0m
    [2KParsing response of length 1973━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:26:27[0m
    [2KSuccessfully parsed 2 QA pairs━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:26:27[0m
    [2K  Generated 2 pairs from chunk 62 (total: 124/167)[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:26:27[0m
    [2KParsing response of length 1776━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:26:27[0m
    [2KSuccessfully parsed 2 QA pairs━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:26:27[0m
    [2K  Generated 2 pairs from chunk 63 (total: 126/167)[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:26:27[0m
    [2KParsing response of length 2136━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:26:27[0m
    [2KSuccessfully parsed 2 QA pairs━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:26:27[0m
    [2K  Generated 2 pairs from chunk 64 (total: 128/167)[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:26:27[0m
    [2KProcessing batch 5/5 with 9 chunks━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:26:27[0m
    Generating qa content [91m━━━━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:26:27[0mINFO:synthetic_data_kit.models.llm_client:Processing batch 1/1 with 9 requests
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating qa content [91m━━━━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:26:47[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating qa content [91m━━━━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:27:11[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating qa content [91m━━━━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:27:33[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating qa content [91m━━━━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:27:54[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating QA pairs [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m [33m0:27:07[0m [36m-:--:--[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating QA pairs [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m [33m0:27:26[0m [36m-:--:--[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating QA pairs [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m [33m0:27:48[0m [36m-:--:--[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating QA pairs [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m [33m0:28:09[0m [36m-:--:--[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating QA pairs [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m [33m0:28:32[0m [36m-:--:--[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    [2KParsing response of length 2051
    [2KSuccessfully parsed 2 QA pairs━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:29:44[0m
    [2K  Generated 2 pairs from chunk 65 (total: 130/167)[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:29:44[0m
    [2KParsing response of length 2233━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:29:44[0m
    [2KSuccessfully parsed 2 QA pairs━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:29:44[0m
    [2K  Generated 2 pairs from chunk 66 (total: 132/167)[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:29:44[0m
    [2KParsing response of length 2365━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:29:44[0m
    [2KSuccessfully parsed 2 QA pairs━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:29:44[0m
    [2K  Generated 2 pairs from chunk 67 (total: 134/167)[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:29:44[0m
    [2KParsing response of length 1972━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:29:44[0m
    [2KSuccessfully parsed 2 QA pairs━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:29:44[0m
    [2K  Generated 2 pairs from chunk 68 (total: 136/167)[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:29:44[0m
    [2KParsing response of length 2559━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:29:44[0m
    [2KSuccessfully parsed 2 QA pairs━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:29:44[0m
    [2K  Generated 2 pairs from chunk 69 (total: 138/167)[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:29:44[0m
    [2KParsing response of length 1793━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:29:44[0m
    [2KSuccessfully parsed 2 QA pairs━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:29:44[0m
    [2K  Generated 2 pairs from chunk 70 (total: 140/167)[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:29:44[0m
    [2KParsing response of length 2004━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:29:44[0m
    [2KSuccessfully parsed 2 QA pairs━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:29:44[0m
    [2K  Generated 2 pairs from chunk 71 (total: 142/167)[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:29:44[0m
    [2KParsing response of length 1981━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:29:44[0m
    [2KSuccessfully parsed 2 QA pairs━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:29:44[0m
    [2K  Generated 2 pairs from chunk 72 (total: 144/167)[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:29:44[0m
    [2KParsing response of length 2188━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:29:44[0m
    [2KSuccessfully parsed 2 QA pairs━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:29:44[0m
    [2K  Generated 2 pairs from chunk 73 (total: 146/167)[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:29:44[0m
    [2KGenerating QA pairs [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m [33m0:28:32[0m [36m-:--:--[0m4[0m
    [2KGenerated 146 QA pairs total (requested: 167)
    [2KSaving result to data/generated/Seating_Arrangement_PDF_2_qa_pairs.json━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:29:44[0m
    [2KSuccessfully wrote test file to data/generated/test_write.json━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:29:44[0m
    [2KSuccessfully wrote result to ━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:29:44[0m
    data/generated/Seating_Arrangement_PDF_2_qa_pairs.json
    [2K[32m✓ Generated qa from Seating_Arrangement_PDF_2.txt -> [0m━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:29:44[0m
    [32mSeating_Arrangement_PDF_2_qa_pairs.json[0m
    [2KLoading config from: ../tutorial_config.yaml━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m[32m [0m[35m 33%[0m[32m [0m[32m(3/9)[0m[32m [0m[33m1:29:44[0m
    [2KConfig has LLM provider set to: vllm[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:29:44[0m
    [2KL Using vllm provider1m━━━━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:29:44[0m
    [2KLoading config from: ../tutorial_config.yamlm╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:29:44[0m
    [2KConfig has LLM provider set to: vllm[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:29:44[0m
    [2KGenerating document summary...━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:29:44[0m
    Generating qa content [91m━━━━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m1:29:44[0mINFO:synthetic_data_kit.models.llm_client:Sending request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating qa content [91m━━━━━━━━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━[0m [35m 44%[0m (4/9) [33m1:29:51[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    [2KSummary generated (908 chars)
    [2KGenerating QA pairs...m━━━━━━━━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━[0m [35m 44%[0m (4/9) [33m1:29:51[0m
    [2KDocument split into 30 chunks━━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━[0m [35m 44%[0m (4/9) [33m1:29:51[0m
    [2KUsing batch size of 16m━━━━━━━━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━[0m [35m 44%[0m (4/9) [33m1:29:51[0m
    [2KProcessing 30 chunks to generate QA pairs...[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━[0m [35m 44%[0m (4/9) [33m1:29:51[0m
    [2KProcessing batch 1/2 with 16 chunks━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━[0m [35m 44%[0m (4/9) [33m1:29:51[0m[?25lGenerating QA pairs [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m [33m0:00:00[0m [36m-:--:--[0m
    Generating qa content [91m━━━━━━━━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━[0m [35m 44%[0m (4/9) [33m1:29:51[0mINFO:synthetic_data_kit.models.llm_client:Processing batch 1/1 with 16 requests
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating qa content [91m━━━━━━━━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━[0m [35m 44%[0m (4/9) [33m1:30:54[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating qa content [91m━━━━━━━━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━[0m [35m 44%[0m (4/9) [33m1:31:58[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating qa content [91m━━━━━━━━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━[0m [35m 44%[0m (4/9) [33m1:32:57[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating qa content [91m━━━━━━━━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━[0m [35m 44%[0m (4/9) [33m1:33:52[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating QA pairs [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m [33m0:04:54[0m [36m-:--:--[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating qa content [91m━━━━━━━━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━[0m [35m 44%[0m (4/9) [33m1:35:44[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating QA pairs [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m [33m0:06:54[0m [36m-:--:--[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating QA pairs [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m [33m0:07:45[0m [36m-:--:--[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating QA pairs [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m [33m0:08:46[0m [36m-:--:--[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating QA pairs [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m [33m0:09:40[0m [36m-:--:--[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating QA pairs [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m [33m0:10:47[0m [36m-:--:--[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating QA pairs [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m [33m0:11:38[0m [36m-:--:--[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating qa content [91m━━━━━━━━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━[0m [35m 44%[0m (4/9) [33m1:42:17[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating QA pairs [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m [33m0:13:18[0m [36m-:--:--[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating QA pairs [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m [33m0:14:19[0m [36m-:--:--[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating QA pairs [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m [33m0:15:32[0m [36m-:--:--[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    [2KParsing response of length 5806
    [2KSuccessfully parsed 6 QA pairs━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━[0m [35m 44%[0m (4/9) [33m1:45:24[0m
    [2K  Generated 6 pairs from chunk 1 (total: 6/167)0m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━[0m [35m 44%[0m (4/9) [33m1:45:24[0m
    [2KParsing response of length 6295━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━[0m [35m 44%[0m (4/9) [33m1:45:24[0m
    [2KSuccessfully parsed 6 QA pairs━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━[0m [35m 44%[0m (4/9) [33m1:45:24[0m
    [2K  Generated 6 pairs from chunk 2 (total: 12/167)m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━[0m [35m 44%[0m (4/9) [33m1:45:24[0m
    [2KParsing response of length 5279━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━[0m [35m 44%[0m (4/9) [33m1:45:24[0m
    [2KSuccessfully parsed 6 QA pairs━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━[0m [35m 44%[0m (4/9) [33m1:45:24[0m
    [2K  Generated 6 pairs from chunk 3 (total: 18/167)m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━[0m [35m 44%[0m (4/9) [33m1:45:24[0m
    [2KParsing response of length 4963━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━[0m [35m 44%[0m (4/9) [33m1:45:24[0m
    [2KSuccessfully parsed 6 QA pairs━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━[0m [35m 44%[0m (4/9) [33m1:45:24[0m
    [2K  Generated 6 pairs from chunk 4 (total: 24/167)m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━[0m [35m 44%[0m (4/9) [33m1:45:24[0m
    [2KParsing response of length 5381━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━[0m [35m 44%[0m (4/9) [33m1:45:24[0m
    [2KSuccessfully parsed 6 QA pairs━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━[0m [35m 44%[0m (4/9) [33m1:45:24[0m
    [2K  Generated 6 pairs from chunk 5 (total: 30/167)m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━[0m [35m 44%[0m (4/9) [33m1:45:24[0m
    [2KParsing response of length 5891━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━[0m [35m 44%[0m (4/9) [33m1:45:24[0m
    [2KSuccessfully parsed 6 QA pairs━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━[0m [35m 44%[0m (4/9) [33m1:45:24[0m
    [2K  Generated 6 pairs from chunk 6 (total: 36/167)m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━[0m [35m 44%[0m (4/9) [33m1:45:24[0m
    [2KParsing response of length 5776━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━[0m [35m 44%[0m (4/9) [33m1:45:24[0m
    [2KSuccessfully parsed 6 QA pairs━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━[0m [35m 44%[0m (4/9) [33m1:45:24[0m
    [2K  Generated 6 pairs from chunk 7 (total: 42/167)m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━[0m [35m 44%[0m (4/9) [33m1:45:24[0m
    [2KParsing response of length 4917━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━[0m [35m 44%[0m (4/9) [33m1:45:24[0m
    [2KSuccessfully parsed 6 QA pairs━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━[0m [35m 44%[0m (4/9) [33m1:45:24[0m
    [2K  Generated 6 pairs from chunk 8 (total: 48/167)m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━[0m [35m 44%[0m (4/9) [33m1:45:24[0m
    [2KParsing response of length 5736━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━[0m [35m 44%[0m (4/9) [33m1:45:24[0m
    [2KSuccessfully parsed 6 QA pairs━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━[0m [35m 44%[0m (4/9) [33m1:45:24[0m
    [2K  Generated 6 pairs from chunk 9 (total: 54/167)m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━[0m [35m 44%[0m (4/9) [33m1:45:24[0m
    [2KParsing response of length 5148━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━[0m [35m 44%[0m (4/9) [33m1:45:24[0m
    [2KSuccessfully parsed 6 QA pairs━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━[0m [35m 44%[0m (4/9) [33m1:45:24[0m
    [2K  Generated 6 pairs from chunk 10 (total: 60/167)╺[0m[90m━━━━━━━━━━━━━━━━━━━━━[0m [35m 44%[0m (4/9) [33m1:45:24[0m
    [2KParsing response of length 6721━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━[0m [35m 44%[0m (4/9) [33m1:45:24[0m
    [2KSuccessfully parsed 6 QA pairs━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━[0m [35m 44%[0m (4/9) [33m1:45:24[0m
    [2K  Generated 6 pairs from chunk 11 (total: 66/167)╺[0m[90m━━━━━━━━━━━━━━━━━━━━━[0m [35m 44%[0m (4/9) [33m1:45:24[0m
    [2KParsing response of length 4653━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━[0m [35m 44%[0m (4/9) [33m1:45:24[0m
    [2KSuccessfully parsed 6 QA pairs━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━[0m [35m 44%[0m (4/9) [33m1:45:24[0m
    [2K  Generated 6 pairs from chunk 12 (total: 72/167)╺[0m[90m━━━━━━━━━━━━━━━━━━━━━[0m [35m 44%[0m (4/9) [33m1:45:24[0m
    [2KParsing response of length 4606━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━[0m [35m 44%[0m (4/9) [33m1:45:24[0m
    [2KSuccessfully parsed 6 QA pairs━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━[0m [35m 44%[0m (4/9) [33m1:45:24[0m
    [2K  Generated 6 pairs from chunk 13 (total: 78/167)╺[0m[90m━━━━━━━━━━━━━━━━━━━━━[0m [35m 44%[0m (4/9) [33m1:45:24[0m
    [2KParsing response of length 5226━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━[0m [35m 44%[0m (4/9) [33m1:45:24[0m
    [2KSuccessfully parsed 6 QA pairs━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━[0m [35m 44%[0m (4/9) [33m1:45:24[0m
    [2K  Generated 6 pairs from chunk 14 (total: 84/167)╺[0m[90m━━━━━━━━━━━━━━━━━━━━━[0m [35m 44%[0m (4/9) [33m1:45:24[0m
    [2KParsing response of length 5781━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━[0m [35m 44%[0m (4/9) [33m1:45:24[0m
    [2KSuccessfully parsed 6 QA pairs━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━[0m [35m 44%[0m (4/9) [33m1:45:24[0m
    [2K  Generated 6 pairs from chunk 15 (total: 90/167)╺[0m[90m━━━━━━━━━━━━━━━━━━━━━[0m [35m 44%[0m (4/9) [33m1:45:24[0m
    [2KParsing response of length 7326━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━[0m [35m 44%[0m (4/9) [33m1:45:24[0m
    [2KSuccessfully parsed 6 QA pairs━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━[0m [35m 44%[0m (4/9) [33m1:45:24[0m
    [2K  Generated 6 pairs from chunk 16 (total: 96/167)╺[0m[90m━━━━━━━━━━━━━━━━━━━━━[0m [35m 44%[0m (4/9) [33m1:45:24[0m
    [2KProcessing batch 2/2 with 14 chunks━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━[0m [35m 44%[0m (4/9) [33m1:45:24[0m
    Generating qa content [91m━━━━━━━━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━[0m [35m 44%[0m (4/9) [33m1:45:24[0mINFO:synthetic_data_kit.models.llm_client:Processing batch 1/1 with 14 requests
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating qa content [91m━━━━━━━━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━[0m [35m 44%[0m (4/9) [33m1:46:34[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating qa content [91m━━━━━━━━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━[0m [35m 44%[0m (4/9) [33m1:47:31[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating QA pairs [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m [33m0:18:33[0m [36m-:--:--[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating QA pairs [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m [33m0:19:26[0m [36m-:--:--[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating qa content [91m━━━━━━━━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━[0m [35m 44%[0m (4/9) [33m1:50:15[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating qa content [91m━━━━━━━━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━[0m [35m 44%[0m (4/9) [33m1:51:06[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating QA pairs [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m [33m0:22:12[0m [36m-:--:--[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating QA pairs [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m [33m0:23:40[0m [36m-:--:--[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating QA pairs [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m [33m0:24:37[0m [36m-:--:--[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating qa content [91m━━━━━━━━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━[0m [35m 44%[0m (4/9) [33m1:55:28[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating QA pairs [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m [33m0:26:28[0m [36m-:--:--[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating QA pairs [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m [33m0:27:25[0m [36m-:--:--[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating QA pairs [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m [33m0:28:39[0m [36m-:--:--[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating QA pairs [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m [33m0:29:41[0m [36m-:--:--[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    [2KParsing response of length 7499
    [2KSuccessfully parsed 6 QA pairs━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━[0m [35m 44%[0m (4/9) [33m1:59:32[0m
    [2K  Generated 6 pairs from chunk 17 (total: 102/167)[0m[90m━━━━━━━━━━━━━━━━━━━━━[0m [35m 44%[0m (4/9) [33m1:59:32[0m
    [2KParsing response of length 5761━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━[0m [35m 44%[0m (4/9) [33m1:59:32[0m
    [2KSuccessfully parsed 6 QA pairs━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━[0m [35m 44%[0m (4/9) [33m1:59:32[0m
    [2K  Generated 6 pairs from chunk 18 (total: 108/167)[0m[90m━━━━━━━━━━━━━━━━━━━━━[0m [35m 44%[0m (4/9) [33m1:59:32[0m
    [2KParsing response of length 5565━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━[0m [35m 44%[0m (4/9) [33m1:59:32[0m
    [2KSuccessfully parsed 6 QA pairs━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━[0m [35m 44%[0m (4/9) [33m1:59:32[0m
    [2K  Generated 6 pairs from chunk 19 (total: 114/167)[0m[90m━━━━━━━━━━━━━━━━━━━━━[0m [35m 44%[0m (4/9) [33m1:59:32[0m
    [2KParsing response of length 5418━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━[0m [35m 44%[0m (4/9) [33m1:59:32[0m
    [2KSuccessfully parsed 6 QA pairs━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━[0m [35m 44%[0m (4/9) [33m1:59:32[0m
    [2K  Generated 6 pairs from chunk 20 (total: 120/167)[0m[90m━━━━━━━━━━━━━━━━━━━━━[0m [35m 44%[0m (4/9) [33m1:59:32[0m
    [2KParsing response of length 6238━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━[0m [35m 44%[0m (4/9) [33m1:59:32[0m
    [2KSuccessfully parsed 6 QA pairs━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━[0m [35m 44%[0m (4/9) [33m1:59:32[0m
    [2K  Generated 6 pairs from chunk 21 (total: 126/167)[0m[90m━━━━━━━━━━━━━━━━━━━━━[0m [35m 44%[0m (4/9) [33m1:59:32[0m
    [2KParsing response of length 4953━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━[0m [35m 44%[0m (4/9) [33m1:59:32[0m
    [2KSuccessfully parsed 6 QA pairs━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━[0m [35m 44%[0m (4/9) [33m1:59:32[0m
    [2K  Generated 6 pairs from chunk 22 (total: 132/167)[0m[90m━━━━━━━━━━━━━━━━━━━━━[0m [35m 44%[0m (4/9) [33m1:59:32[0m
    [2KParsing response of length 6154━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━[0m [35m 44%[0m (4/9) [33m1:59:32[0m
    [2KSuccessfully parsed 6 QA pairs━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━[0m [35m 44%[0m (4/9) [33m1:59:32[0m
    [2K  Generated 6 pairs from chunk 23 (total: 138/167)[0m[90m━━━━━━━━━━━━━━━━━━━━━[0m [35m 44%[0m (4/9) [33m1:59:32[0m
    [2KParsing response of length 8716━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━[0m [35m 44%[0m (4/9) [33m1:59:32[0m
    [2KSuccessfully parsed 6 QA pairs━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━[0m [35m 44%[0m (4/9) [33m1:59:32[0m
    [2K  Generated 6 pairs from chunk 24 (total: 144/167)[0m[90m━━━━━━━━━━━━━━━━━━━━━[0m [35m 44%[0m (4/9) [33m1:59:32[0m
    [2KParsing response of length 5910━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━[0m [35m 44%[0m (4/9) [33m1:59:32[0m
    [2KSuccessfully parsed 6 QA pairs━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━[0m [35m 44%[0m (4/9) [33m1:59:32[0m
    [2K  Generated 6 pairs from chunk 25 (total: 150/167)[0m[90m━━━━━━━━━━━━━━━━━━━━━[0m [35m 44%[0m (4/9) [33m1:59:32[0m
    [2KParsing response of length 6179━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━[0m [35m 44%[0m (4/9) [33m1:59:32[0m
    [2KSuccessfully parsed 6 QA pairs━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━[0m [35m 44%[0m (4/9) [33m1:59:32[0m
    [2K  Generated 6 pairs from chunk 26 (total: 156/167)[0m[90m━━━━━━━━━━━━━━━━━━━━━[0m [35m 44%[0m (4/9) [33m1:59:32[0m
    [2KParsing response of length 5138━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━[0m [35m 44%[0m (4/9) [33m1:59:32[0m
    [2KSuccessfully parsed 6 QA pairs━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━[0m [35m 44%[0m (4/9) [33m1:59:32[0m
    [2K  Generated 6 pairs from chunk 27 (total: 162/167)[0m[90m━━━━━━━━━━━━━━━━━━━━━[0m [35m 44%[0m (4/9) [33m1:59:32[0m
    [2KParsing response of length 6147━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━[0m [35m 44%[0m (4/9) [33m1:59:32[0m
    [2KSuccessfully parsed 6 QA pairs━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━[0m [35m 44%[0m (4/9) [33m1:59:32[0m
    [2K  Generated 5 pairs from chunk 28 (total: 167/167)[0m[90m━━━━━━━━━━━━━━━━━━━━━[0m [35m 44%[0m (4/9) [33m1:59:32[0m
    [2KGenerating QA pairs [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m [33m0:29:41[0m [36m-:--:--[0m2[0m
    [2KGenerated 167 QA pairs total (requested: 167)
    [2KSaving result to data/generated/Seating_Arrangement_PDF_3_qa_pairs.json━━━━━━━━━[0m [35m 44%[0m (4/9) [33m1:59:32[0m
    [2KSuccessfully wrote test file to data/generated/test_write.json━━━━━━━━━━━━━━━━━━[0m [35m 44%[0m (4/9) [33m1:59:32[0m
    [2KSuccessfully wrote result to ━━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━[0m [35m 44%[0m (4/9) [33m1:59:32[0m
    data/generated/Seating_Arrangement_PDF_3_qa_pairs.json
    [2K[32m✓ Generated qa from Seating_Arrangement_PDF_3.txt -> [0m━━━━━━━━━━━━━━━━━━[0m [35m 44%[0m (4/9) [33m1:59:32[0m
    [32mSeating_Arrangement_PDF_3_qa_pairs.json[0m
    [2KLoading config from: ../tutorial_config.yaml━━━━━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━[0m[32m [0m[35m 44%[0m[32m [0m[32m(4/9)[0m[32m [0m[33m1:59:32[0m
    [2KConfig has LLM provider set to: vllm━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━[0m [35m 44%[0m (4/9) [33m1:59:32[0m
    [2KL Using vllm provider1m━━━━━━━━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━[0m [35m 44%[0m (4/9) [33m1:59:32[0m
    [2KLoading config from: ../tutorial_config.yaml[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━[0m [35m 44%[0m (4/9) [33m1:59:32[0m
    [2KConfig has LLM provider set to: vllm━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━[0m [35m 44%[0m (4/9) [33m1:59:32[0m
    [2KGenerating document summary...━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━[0m [35m 44%[0m (4/9) [33m1:59:32[0m
    Generating qa content [91m━━━━━━━━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━[0m [35m 44%[0m (4/9) [33m1:59:32[0mINFO:synthetic_data_kit.models.llm_client:Sending request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating qa content [91m━━━━━━━━━━━━━━━━━━━━━[0m[91m╸[0m[90m━━━━━━━━━━━━━━━━━[0m [35m 56%[0m (5/9) [33m1:59:37[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    [2KSummary generated (796 chars)
    [2KGenerating QA pairs...m━━━━━━━━━━━━━━━━━━━━━[0m[91m╸[0m[90m━━━━━━━━━━━━━━━━━[0m [35m 56%[0m (5/9) [33m1:59:37[0m
    [2KDocument split into 1 chunks━━━━━━━━━━━━━━━━[0m[91m╸[0m[90m━━━━━━━━━━━━━━━━━[0m [35m 56%[0m (5/9) [33m1:59:37[0m
    [2KUsing batch size of 16m━━━━━━━━━━━━━━━━━━━━━[0m[91m╸[0m[90m━━━━━━━━━━━━━━━━━[0m [35m 56%[0m (5/9) [33m1:59:37[0m
    [2KProcessing 1 chunks to generate QA pairs...━[0m[91m╸[0m[90m━━━━━━━━━━━━━━━━━[0m [35m 56%[0m (5/9) [33m1:59:37[0m
    [2KProcessing batch 1/1 with 1 chunks━━━━━━━━━━[0m[91m╸[0m[90m━━━━━━━━━━━━━━━━━[0m [35m 56%[0m (5/9) [33m1:59:37[0m[?25lGenerating QA pairs [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m [33m0:00:00[0m [36m-:--:--[0m
    Generating qa content [91m━━━━━━━━━━━━━━━━━━━━━[0m[91m╸[0m[90m━━━━━━━━━━━━━━━━━[0m [35m 56%[0m (5/9) [33m1:59:37[0mINFO:synthetic_data_kit.models.llm_client:Processing batch 1/1 with 1 requests
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating QA pairs [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m [33m0:02:49[0m [36m-:--:--[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    [2KParsing response of length 17267
    [2KDirect JSON parsing failed: Extra data: line 1 column 7958 (char 7957)━━━━━━━━━━[0m [35m 56%[0m (5/9) [33m2:02:27[0m
    [2KAttempted to parse: [ { "question": "In a circular arrangement of 8 people ━━━━━[0m [35m 56%[0m (5/9) [33m2:02:27[0m
    around a conference table, A must sit next to both B and C, but B and C cannot 
    sit next to each other. D must sit opposite A, and E must sit next...
    [2KFalling back to regex pattern matching━━━━━━[0m[91m╸[0m[90m━━━━━━━━━━━━━━━━━[0m [35m 56%[0m (5/9) [33m2:02:27[0m
    [2KExtracted 17 QA pairs with regex━━━━━━━━━━━━[0m[91m╸[0m[90m━━━━━━━━━━━━━━━━━[0m [35m 56%[0m (5/9) [33m2:02:27[0m
    [2K  Generated 17 pairs from chunk 1 (total: 17/167)[0m[91m╸[0m[90m━━━━━━━━━━━━━━━━━[0m [35m 56%[0m (5/9) [33m2:02:27[0m
    [2KGenerating QA pairs [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m [33m0:02:49[0m [36m-:--:--[0m7[0m
    [2KGenerated 17 QA pairs total (requested: 167)
    [2KSaving result to data/generated/blood-relation-puzzle_qa_pairs.json━━━━━━━━━━━━━[0m [35m 56%[0m (5/9) [33m2:02:27[0m
    [2KSuccessfully wrote test file to data/generated/test_write.jsonm━━━━━━━━━━━━━━━━━[0m [35m 56%[0m (5/9) [33m2:02:27[0m
    [2KSuccessfully wrote result to data/generated/blood-relation-puzzle_qa_pairs.json━[0m [35m 56%[0m (5/9) [33m2:02:27[0m
    [2K[32m✓ Generated qa from blood-relation-puzzle.txt -> [0m[90m━━━━━━━━━━━━━━━━━[0m [35m 56%[0m (5/9) [33m2:02:27[0m
    [32mblood-relation-puzzle_qa_pairs.json[0m
    [2KLoading config from: ../tutorial_config.yaml━━━━━━━━━━━━━━━━━━[0m[91m╸[0m[90m━━━━━━━━━━━━━━━━━[0m[32m [0m[35m 56%[0m[32m [0m[32m(5/9)[0m[32m [0m[33m2:02:27[0m
    [2KConfig has LLM provider set to: vllm━━━━━━━━[0m[91m╸[0m[90m━━━━━━━━━━━━━━━━━[0m [35m 56%[0m (5/9) [33m2:02:27[0m
    [2KL Using vllm provider1m━━━━━━━━━━━━━━━━━━━━━[0m[91m╸[0m[90m━━━━━━━━━━━━━━━━━[0m [35m 56%[0m (5/9) [33m2:02:27[0m
    [2KLoading config from: ../tutorial_config.yaml[0m[91m╸[0m[90m━━━━━━━━━━━━━━━━━[0m [35m 56%[0m (5/9) [33m2:02:27[0m
    [2KConfig has LLM provider set to: vllm━━━━━━━━[0m[91m╸[0m[90m━━━━━━━━━━━━━━━━━[0m [35m 56%[0m (5/9) [33m2:02:27[0m
    [2KGenerating document summary...━━━━━━━━━━━━━━[0m[91m╸[0m[90m━━━━━━━━━━━━━━━━━[0m [35m 56%[0m (5/9) [33m2:02:27[0m
    Generating qa content [91m━━━━━━━━━━━━━━━━━━━━━[0m[91m╸[0m[90m━━━━━━━━━━━━━━━━━[0m [35m 56%[0m (5/9) [33m2:02:27[0mINFO:synthetic_data_kit.models.llm_client:Sending request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating qa content [91m━━━━━━━━━━━━━━━━━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━[0m [35m 67%[0m (6/9) [33m2:02:33[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    [2KSummary generated (883 chars)
    [2KGenerating QA pairs...m━━━━━━━━━━━━━━━━━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━[0m [35m 67%[0m (6/9) [33m2:02:33[0m
    [2KDocument split into 7 chunks━━━━━━━━━━━━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━[0m [35m 67%[0m (6/9) [33m2:02:33[0m
    [2KUsing batch size of 16m━━━━━━━━━━━━━━━━━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━[0m [35m 67%[0m (6/9) [33m2:02:33[0m
    [2KProcessing 7 chunks to generate QA pairs...━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━[0m [35m 67%[0m (6/9) [33m2:02:33[0m
    [2KProcessing batch 1/1 with 7 chunks━━━━━━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━[0m [35m 67%[0m (6/9) [33m2:02:33[0m[?25lGenerating QA pairs [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m [33m0:00:00[0m [36m-:--:--[0m
    Generating qa content [91m━━━━━━━━━━━━━━━━━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━[0m [35m 67%[0m (6/9) [33m2:02:33[0mINFO:synthetic_data_kit.models.llm_client:Processing batch 1/1 with 7 requests
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating QA pairs [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m [33m0:02:49[0m [36m-:--:--[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating QA pairs [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m [33m0:05:39[0m [36m-:--:--[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating QA pairs [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m [33m0:08:12[0m [36m-:--:--[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating QA pairs [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m [33m0:11:02[0m [36m-:--:--[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating QA pairs [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m [33m0:13:27[0m [36m-:--:--[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating QA pairs [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m [33m0:15:52[0m [36m-:--:--[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating QA pairs [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m [33m0:18:40[0m [36m-:--:--[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    [2KParsing response of length 16431
    [2KFalling back to regex pattern matching━━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━[0m [35m 67%[0m (6/9) [33m2:21:14[0m
    [2KExtracted 18 QA pairs with regex━━━━━━━━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━[0m [35m 67%[0m (6/9) [33m2:21:14[0m
    [2K  Generated 18 pairs from chunk 1 (total: 18/167)[0m[90m╺[0m[90m━━━━━━━━━━━━[0m [35m 67%[0m (6/9) [33m2:21:14[0m
    [2KParsing response of length 15858━━━━━━━━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━[0m [35m 67%[0m (6/9) [33m2:21:14[0m
    [2KFalling back to regex pattern matching━━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━[0m [35m 67%[0m (6/9) [33m2:21:14[0m
    [2KExtracted 18 QA pairs with regex━━━━━━━━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━[0m [35m 67%[0m (6/9) [33m2:21:14[0m
    [2K  Generated 18 pairs from chunk 2 (total: 36/167)[0m[90m╺[0m[90m━━━━━━━━━━━━[0m [35m 67%[0m (6/9) [33m2:21:14[0m
    [2KParsing response of length 14003━━━━━━━━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━[0m [35m 67%[0m (6/9) [33m2:21:14[0m
    [2KSuccessfully parsed 21 QA pairs━━━━━━━━━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━[0m [35m 67%[0m (6/9) [33m2:21:14[0m
    [2K  Generated 21 pairs from chunk 3 (total: 57/167)[0m[90m╺[0m[90m━━━━━━━━━━━━[0m [35m 67%[0m (6/9) [33m2:21:14[0m
    [2KParsing response of length 15171━━━━━━━━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━[0m [35m 67%[0m (6/9) [33m2:21:14[0m
    [2KFalling back to regex pattern matching━━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━[0m [35m 67%[0m (6/9) [33m2:21:14[0m
    [2KExtracted 23 QA pairs with regex━━━━━━━━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━[0m [35m 67%[0m (6/9) [33m2:21:14[0m
    [2K  Generated 23 pairs from chunk 4 (total: 80/167)[0m[90m╺[0m[90m━━━━━━━━━━━━[0m [35m 67%[0m (6/9) [33m2:21:14[0m
    [2KParsing response of length 14433━━━━━━━━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━[0m [35m 67%[0m (6/9) [33m2:21:14[0m
    [2KSuccessfully parsed 23 QA pairs━━━━━━━━━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━[0m [35m 67%[0m (6/9) [33m2:21:14[0m
    [2K  Generated 23 pairs from chunk 5 (total: 103/167)[0m[90m╺[0m[90m━━━━━━━━━━━━[0m [35m 67%[0m (6/9) [33m2:21:14[0m
    [2KParsing response of length 12858━━━━━━━━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━[0m [35m 67%[0m (6/9) [33m2:21:14[0m
    [2KSuccessfully parsed 25 QA pairs━━━━━━━━━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━[0m [35m 67%[0m (6/9) [33m2:21:14[0m
    [2K  Generated 25 pairs from chunk 6 (total: 128/167)[0m[90m╺[0m[90m━━━━━━━━━━━━[0m [35m 67%[0m (6/9) [33m2:21:14[0m
    [2KParsing response of length 16409━━━━━━━━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━[0m [35m 67%[0m (6/9) [33m2:21:14[0m
    [2KFalling back to regex pattern matching━━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━[0m [35m 67%[0m (6/9) [33m2:21:14[0m
    [2KExtracted 18 QA pairs with regex━━━━━━━━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━[0m [35m 67%[0m (6/9) [33m2:21:14[0m
    [2K  Generated 18 pairs from chunk 7 (total: 146/167)[0m[90m╺[0m[90m━━━━━━━━━━━━[0m [35m 67%[0m (6/9) [33m2:21:14[0m
    [2KGenerating QA pairs [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m [33m0:18:40[0m [36m-:--:--[0m4[0m
    [2KGenerated 146 QA pairs total (requested: 167)
    [2KSaving result to data/generated/blood-relationship_qa_pairs.json[90m━━━━━━━━━━━━[0m [35m 67%[0m (6/9) [33m2:21:14[0m
    [2KSuccessfully wrote test file to data/generated/test_write.json╺[0m[90m━━━━━━━━━━━━[0m [35m 67%[0m (6/9) [33m2:21:15[0m
    [2KSuccessfully wrote result to data/generated/blood-relationship_qa_pairs.json━━━━[0m [35m 67%[0m (6/9) [33m2:21:15[0m
    [2K[32m✓ Generated qa from blood-relationship.txt -> blood-relationship_qa_pairs.json[0m35m 67%[0m (6/9) [33m2:21:15[0m
    [2KLoading config from: ../tutorial_config.yaml━━━━━━━━━━━━━━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━[0m[32m [0m[35m 67%[0m[32m [0m[32m(6/9)[0m[32m [0m[33m2:21:15[0m
    [2KConfig has LLM provider set to: vllm━━━━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━[0m [35m 67%[0m (6/9) [33m2:21:15[0m
    [2KL Using vllm provider1m━━━━━━━━━━━━━━━━━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━[0m [35m 67%[0m (6/9) [33m2:21:15[0m
    [2KLoading config from: ../tutorial_config.yaml━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━[0m [35m 67%[0m (6/9) [33m2:21:15[0m
    [2KConfig has LLM provider set to: vllm━━━━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━[0m [35m 67%[0m (6/9) [33m2:21:15[0m
    [2KGenerating document summary...━━━━━━━━━━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━[0m [35m 67%[0m (6/9) [33m2:21:15[0m
    Generating qa content [91m━━━━━━━━━━━━━━━━━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━[0m [35m 67%[0m (6/9) [33m2:21:15[0mINFO:synthetic_data_kit.models.llm_client:Sending request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating qa content [91m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━[0m [35m 78%[0m (7/9) [33m2:21:20[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    [2KSummary generated (707 chars)
    [2KGenerating QA pairs...m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━[0m [35m 78%[0m (7/9) [33m2:21:20[0m
    [2KDocument split into 9 chunks━━━━━━━━━━━━━━━━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━[0m [35m 78%[0m (7/9) [33m2:21:20[0m
    [2KUsing batch size of 16m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━[0m [35m 78%[0m (7/9) [33m2:21:20[0m
    [2KProcessing 9 chunks to generate QA pairs...━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━[0m [35m 78%[0m (7/9) [33m2:21:20[0m
    [2KProcessing batch 1/1 with 9 chunks━━━━━━━━━━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━[0m [35m 78%[0m (7/9) [33m2:21:20[0m[?25lGenerating QA pairs [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m [33m0:00:00[0m [36m-:--:--[0m
    Generating qa content [91m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━[0m [35m 78%[0m (7/9) [33m2:21:20[0mINFO:synthetic_data_kit.models.llm_client:Processing batch 1/1 with 9 requests
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating qa content [91m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━[0m [35m 78%[0m (7/9) [33m2:23:06[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating QA pairs [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m [33m0:04:35[0m [36m-:--:--[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating qa content [91m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━[0m [35m 78%[0m (7/9) [33m2:28:44[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating QA pairs [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m [33m0:10:13[0m [36m-:--:--[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating QA pairs [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m [33m0:12:58[0m [36m-:--:--[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating QA pairs [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m [33m0:15:30[0m [36m-:--:--[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating QA pairs [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m [33m0:18:19[0m [36m-:--:--[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating QA pairs [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m [33m0:21:08[0m [36m-:--:--[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating QA pairs [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m [33m0:23:57[0m [36m-:--:--[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    [2KParsing response of length 11119
    [2KSuccessfully parsed 20 QA pairs━━━━━━━━━━━━━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━[0m [35m 78%[0m (7/9) [33m2:45:17[0m
    [2K  Generated 20 pairs from chunk 1 (total: 20/167)━━━━[0m[90m╺[0m[90m━━━━━━━━[0m [35m 78%[0m (7/9) [33m2:45:17[0m
    [2KParsing response of length 15582━━━━━━━━━━━━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━[0m [35m 78%[0m (7/9) [33m2:45:17[0m
    [2KFalling back to regex pattern matching━━━━━━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━[0m [35m 78%[0m (7/9) [33m2:45:17[0m
    [2KExtracted 16 QA pairs with regex━━━━━━━━━━━━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━[0m [35m 78%[0m (7/9) [33m2:45:17[0m
    [2K  Generated 16 pairs from chunk 2 (total: 36/167)━━━━[0m[90m╺[0m[90m━━━━━━━━[0m [35m 78%[0m (7/9) [33m2:45:17[0m
    [2KParsing response of length 15692━━━━━━━━━━━━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━[0m [35m 78%[0m (7/9) [33m2:45:17[0m
    [2KFalling back to regex pattern matching━━━━━━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━[0m [35m 78%[0m (7/9) [33m2:45:17[0m
    [2KExtracted 17 QA pairs with regex━━━━━━━━━━━━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━[0m [35m 78%[0m (7/9) [33m2:45:17[0m
    [2K  Generated 17 pairs from chunk 3 (total: 53/167)━━━━[0m[90m╺[0m[90m━━━━━━━━[0m [35m 78%[0m (7/9) [33m2:45:17[0m
    [2KParsing response of length 15632━━━━━━━━━━━━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━[0m [35m 78%[0m (7/9) [33m2:45:17[0m
    [2KFalling back to regex pattern matching━━━━━━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━[0m [35m 78%[0m (7/9) [33m2:45:17[0m
    [2KExtracted 16 QA pairs with regex━━━━━━━━━━━━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━[0m [35m 78%[0m (7/9) [33m2:45:17[0m
    [2K  Generated 16 pairs from chunk 4 (total: 69/167)━━━━[0m[90m╺[0m[90m━━━━━━━━[0m [35m 78%[0m (7/9) [33m2:45:17[0m
    [2KParsing response of length 16540━━━━━━━━━━━━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━[0m [35m 78%[0m (7/9) [33m2:45:17[0m
    [2KSuccessfully parsed 20 QA pairs━━━━━━━━━━━━━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━[0m [35m 78%[0m (7/9) [33m2:45:17[0m
    [2K  Generated 20 pairs from chunk 5 (total: 89/167)━━━━[0m[90m╺[0m[90m━━━━━━━━[0m [35m 78%[0m (7/9) [33m2:45:17[0m
    [2KParsing response of length 14582━━━━━━━━━━━━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━[0m [35m 78%[0m (7/9) [33m2:45:17[0m
    [2KSuccessfully parsed 20 QA pairs━━━━━━━━━━━━━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━[0m [35m 78%[0m (7/9) [33m2:45:17[0m
    [2K  Generated 20 pairs from chunk 6 (total: 109/167)━━━[0m[90m╺[0m[90m━━━━━━━━[0m [35m 78%[0m (7/9) [33m2:45:17[0m
    [2KParsing response of length 15312━━━━━━━━━━━━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━[0m [35m 78%[0m (7/9) [33m2:45:17[0m
    [2KFalling back to regex pattern matching━━━━━━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━[0m [35m 78%[0m (7/9) [33m2:45:17[0m
    [2KExtracted 17 QA pairs with regex━━━━━━━━━━━━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━[0m [35m 78%[0m (7/9) [33m2:45:17[0m
    [2K  Generated 17 pairs from chunk 7 (total: 126/167)━━━[0m[90m╺[0m[90m━━━━━━━━[0m [35m 78%[0m (7/9) [33m2:45:17[0m
    [2KParsing response of length 15698━━━━━━━━━━━━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━[0m [35m 78%[0m (7/9) [33m2:45:17[0m
    [2KFalling back to regex pattern matching━━━━━━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━[0m [35m 78%[0m (7/9) [33m2:45:17[0m
    [2KExtracted 18 QA pairs with regex━━━━━━━━━━━━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━[0m [35m 78%[0m (7/9) [33m2:45:17[0m
    [2K  Generated 18 pairs from chunk 8 (total: 144/167)━━━[0m[90m╺[0m[90m━━━━━━━━[0m [35m 78%[0m (7/9) [33m2:45:17[0m
    [2KParsing response of length 15474━━━━━━━━━━━━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━[0m [35m 78%[0m (7/9) [33m2:45:17[0m
    [2KFalling back to regex pattern matching━━━━━━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━[0m [35m 78%[0m (7/9) [33m2:45:17[0m
    [2KExtracted 16 QA pairs with regex━━━━━━━━━━━━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━[0m [35m 78%[0m (7/9) [33m2:45:17[0m
    [2K  Generated 16 pairs from chunk 9 (total: 160/167)━━━[0m[90m╺[0m[90m━━━━━━━━[0m [35m 78%[0m (7/9) [33m2:45:17[0m
    [2KGenerating QA pairs [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m [33m0:23:57[0m [36m-:--:--[0m7[0m
    [2KGenerated 160 QA pairs total (requested: 167)
    [2KSaving result to  [91m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━[0m [35m 78%[0m (7/9) [33m2:45:17[0m
    data/generated/circular-arrangement-with-blood-relation_qa_pairs.json
    [2KSuccessfully wrote test file to data/generated/test_write.json╺[0m[90m━━━━━━━━[0m [35m 78%[0m (7/9) [33m2:45:17[0m
    [2KSuccessfully wrote result to ━━━━━━━━━━━━━━━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━[0m [35m 78%[0m (7/9) [33m2:45:17[0m
    data/generated/circular-arrangement-with-blood-relation_qa_pairs.json
    [2K[32m✓ Generated qa from circular-arrangement-with-blood-relation.txt -> [0m━━━[0m [35m 78%[0m (7/9) [33m2:45:17[0m
    [32mcircular-arrangement-with-blood-relation_qa_pairs.json[0m
    [2KLoading config from: ../tutorial_config.yaml━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━[0m[32m [0m[35m 78%[0m[32m [0m[32m(7/9)[0m[32m [0m[33m2:45:17[0m
    [2KConfig has LLM provider set to: vllm━━━━━━━━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━[0m [35m 78%[0m (7/9) [33m2:45:17[0m
    [2KL Using vllm provider1m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━[0m [35m 78%[0m (7/9) [33m2:45:17[0m
    [2KLoading config from: ../tutorial_config.yaml━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━[0m [35m 78%[0m (7/9) [33m2:45:17[0m
    [2KConfig has LLM provider set to: vllm━━━━━━━━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━[0m [35m 78%[0m (7/9) [33m2:45:17[0m
    [2KGenerating document summary...━━━━━━━━━━━━━━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━[0m [35m 78%[0m (7/9) [33m2:45:17[0m
    Generating qa content [91m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━[0m [35m 78%[0m (7/9) [33m2:45:17[0mINFO:synthetic_data_kit.models.llm_client:Sending request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating qa content [91m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m[91m╸[0m[90m━━━━[0m [35m 89%[0m (8/9) [33m2:45:23[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    [2KSummary generated (790 chars)
    [2KGenerating QA pairs...m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m[91m╸[0m[90m━━━━[0m [35m 89%[0m (8/9) [33m2:45:23[0m
    [2KDocument split into 5 chunks━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m[91m╸[0m[90m━━━━[0m [35m 89%[0m (8/9) [33m2:45:23[0m
    [2KUsing batch size of 16m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m[91m╸[0m[90m━━━━[0m [35m 89%[0m (8/9) [33m2:45:23[0m
    [2KProcessing 5 chunks to generate QA pairs...━━━━━━━━━━━━━━[0m[91m╸[0m[90m━━━━[0m [35m 89%[0m (8/9) [33m2:45:23[0m
    [2KProcessing batch 1/1 with 5 chunks━━━━━━━━━━━━━━━━━━━━━━━[0m[91m╸[0m[90m━━━━[0m [35m 89%[0m (8/9) [33m2:45:23[0m[?25lGenerating QA pairs [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m [33m0:00:00[0m [36m-:--:--[0m
    Generating qa content [91m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m[91m╸[0m[90m━━━━[0m [35m 89%[0m (8/9) [33m2:45:23[0mINFO:synthetic_data_kit.models.llm_client:Processing batch 1/1 with 5 requests
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating QA pairs [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m [33m0:02:49[0m [36m-:--:--[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating QA pairs [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m [33m0:05:39[0m [36m-:--:--[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating QA pairs [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m [33m0:08:28[0m [36m-:--:--[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating QA pairs [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m [33m0:11:17[0m [36m-:--:--[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    INFO:synthetic_data_kit.models.llm_client:Sending batch request to vLLM model Unsloth/Llama-3.3-70B-Instruct...
    [2KGenerating QA pairs [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m [33m0:14:05[0m [36m-:--:--[0mINFO:synthetic_data_kit.models.llm_client:Received response with status code: 200
    [2KParsing response of length 16670
    [2KFalling back to regex pattern matching━━━━━━━━━━━━━━━━━━━[0m[91m╸[0m[90m━━━━[0m [35m 89%[0m (8/9) [33m2:59:29[0m
    [2KExtracted 21 QA pairs with regex━━━━━━━━━━━━━━━━━━━━━━━━━[0m[91m╸[0m[90m━━━━[0m [35m 89%[0m (8/9) [33m2:59:29[0m
    [2K  Generated 21 pairs from chunk 1 (total: 21/167)━━━━━━━━[0m[91m╸[0m[90m━━━━[0m [35m 89%[0m (8/9) [33m2:59:29[0m
    [2KParsing response of length 15380━━━━━━━━━━━━━━━━━━━━━━━━━[0m[91m╸[0m[90m━━━━[0m [35m 89%[0m (8/9) [33m2:59:29[0m
    [2KFalling back to regex pattern matching━━━━━━━━━━━━━━━━━━━[0m[91m╸[0m[90m━━━━[0m [35m 89%[0m (8/9) [33m2:59:29[0m
    [2KExtracted 19 QA pairs with regex━━━━━━━━━━━━━━━━━━━━━━━━━[0m[91m╸[0m[90m━━━━[0m [35m 89%[0m (8/9) [33m2:59:29[0m
    [2K  Generated 19 pairs from chunk 2 (total: 40/167)━━━━━━━━[0m[91m╸[0m[90m━━━━[0m [35m 89%[0m (8/9) [33m2:59:29[0m
    [2KParsing response of length 16825━━━━━━━━━━━━━━━━━━━━━━━━━[0m[91m╸[0m[90m━━━━[0m [35m 89%[0m (8/9) [33m2:59:29[0m
    [2KFalling back to regex pattern matching━━━━━━━━━━━━━━━━━━━[0m[91m╸[0m[90m━━━━[0m [35m 89%[0m (8/9) [33m2:59:29[0m
    [2KExtracted 21 QA pairs with regex━━━━━━━━━━━━━━━━━━━━━━━━━[0m[91m╸[0m[90m━━━━[0m [35m 89%[0m (8/9) [33m2:59:29[0m
    [2K  Generated 21 pairs from chunk 3 (total: 61/167)━━━━━━━━[0m[91m╸[0m[90m━━━━[0m [35m 89%[0m (8/9) [33m2:59:29[0m
    [2KParsing response of length 15873━━━━━━━━━━━━━━━━━━━━━━━━━[0m[91m╸[0m[90m━━━━[0m [35m 89%[0m (8/9) [33m2:59:29[0m
    [2KFalling back to regex pattern matching━━━━━━━━━━━━━━━━━━━[0m[91m╸[0m[90m━━━━[0m [35m 89%[0m (8/9) [33m2:59:29[0m
    [2KExtracted 18 QA pairs with regex━━━━━━━━━━━━━━━━━━━━━━━━━[0m[91m╸[0m[90m━━━━[0m [35m 89%[0m (8/9) [33m2:59:29[0m
    [2K  Generated 18 pairs from chunk 4 (total: 79/167)━━━━━━━━[0m[91m╸[0m[90m━━━━[0m [35m 89%[0m (8/9) [33m2:59:29[0m
    [2KParsing response of length 14668━━━━━━━━━━━━━━━━━━━━━━━━━[0m[91m╸[0m[90m━━━━[0m [35m 89%[0m (8/9) [33m2:59:29[0m
    [2KFalling back to regex pattern matching━━━━━━━━━━━━━━━━━━━[0m[91m╸[0m[90m━━━━[0m [35m 89%[0m (8/9) [33m2:59:29[0m
    [2KExtracted 23 QA pairs with regex━━━━━━━━━━━━━━━━━━━━━━━━━[0m[91m╸[0m[90m━━━━[0m [35m 89%[0m (8/9) [33m2:59:29[0m
    [2K  Generated 23 pairs from chunk 5 (total: 102/167)━━━━━━━[0m[91m╸[0m[90m━━━━[0m [35m 89%[0m (8/9) [33m2:59:29[0m
    [2KGenerating QA pairs [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m [33m0:14:06[0m [36m-:--:--[0m9[0m
    [2KGenerated 102 QA pairs total (requested: 167)
    [2KSaving result to  [91m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m[91m╸[0m[90m━━━━[0m [35m 89%[0m (8/9) [33m2:59:29[0m
    data/generated/seating-arrangement-logical-reasoning-_qa_pairs.json
    [2KSuccessfully wrote test file to data/generated/test_write.json[91m╸[0m[90m━━━━[0m [35m 89%[0m (8/9) [33m2:59:29[0m
    [2KSuccessfully wrote result to ━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m[91m╸[0m[90m━━━━[0m [35m 89%[0m (8/9) [33m2:59:29[0m
    data/generated/seating-arrangement-logical-reasoning-_qa_pairs.json
    [2K[32m✓ Generated qa from seating-arrangement-logical-reasoning-.txt -> [0mm━━━━[0m [35m 89%[0m (8/9) [33m2:59:29[0m
    [32mseating-arrangement-logical-reasoning-_qa_pairs.json[0m
    [2KGenerating qa content [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m100%[0m (9/9) [33m2:59:29[0m [0m[35m 89%[0m[32m [0m[32m(8/9)[0m[32m [0m[33m2:59:29[0m
    [?25h
    [1m==================================================[0m
    [1;34mContent Generation Summary [0m[1;34m([0m[1;34mqa[0m[1;34m)[0m[1;34m:[0m
    Total files: [1;36m9[0m
    [32mSuccessful: [0m[1;36m9[0m
    [32mFailed: [0m[1;36m0[0m
    [1m==================================================[0m
    [32m✅ All files processed successfully![0m
    Loading config from: /usr/local/lib/python3.12/dist-packages/synthetic_data_kit/config.yaml
    Config has LLM provider set to: api-endpoint
    Loading config from: /usr/local/lib/python3.12/dist-packages/synthetic_data_kit/config.yaml
    Config has LLM provider set to: api-endpoint
    Loading config from: ../tutorial_config.yaml
    Config has LLM provider set to: vllm
    get_llm_provider returning: vllm
    [32m🔗 Using vllm provider[0m
    [34mProcessing directory: [0m[1;34m.[0m[1;35m/data/parsed/[0m[34m for cot generation[0m
    [34mFound [0m[1;36m9[0m[34m cot files to process[0m
    [2KLoading config from: ../tutorial_config.yaml━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m (0/9) [33m0:00:00[0m
    [2KConfig has LLM provider set to: vllm━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m (0/9) [33m0:00:00[0m
    [2KL Using vllm provider90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m (0/9) [33m0:00:00[0m
    [2KProcessing 9 chunks to generate CoT examples...━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m (0/9) [33m0:00:04[0m
    [2KBatch processing complete.0m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m (0/9) [33m0:07:02[0m
    [2KGenerated 36 CoT examples total (requested: 50)━━━━━━━━━━━━━━━[0m [35m  0%[0m (0/9) [33m0:07:02[0m
    [2KGenerated 36 chain-of-thought examples━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m (0/9) [33m0:07:02[0m
    [2Krating cot content [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m (0/9) [33m0:07:02[0m
    [2KFirst CoT Example: [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m (0/9) [33m0:07:02[0m
    [2KQuestion: In a circular arrangement of 12 people around a conference table, A 0m (0/9) [33m0:07:02[0m
    demands to sit next to both B and C, but B and C have an irreconcilable personal
    conflict and refuse to sit next to each other under any circumstances. D, who is
    A's business partner, must sit directly opposite A. E, D's personal assistant, 
    must sit next to D. F is mediating between two specific executives, G and H. 
    Given these complex interpersonal and positional constraints, demonstrate 
    through systematic analysis which specific pair cannot be seated together in any
    valid arrangement.
    [2KReasoning (first 100 chars): Step 1: Analyze the circular arrangement problem 0m (0/9) [33m0:07:02[0m
    with multiple constraints. Step 2: A's requirement ...
    [2KAnswer: Based on the geometric constraints of the 12-person circular table and m (0/9) [33m0:07:02[0m
    the complex interpersonal requirements, B and C cannot sit together under any 
    circumstances.
    [2K[32m✓ Generated cot from Circular Arrangement MCQ [0m[1;32m[[0m[32mFree PDF[0m[1;32m][0m[32m - Objective Question [0m
    [32mAnswer for Circular Arrangement Quiz - Download Now!.txt -> Circular Arrangement[0m
    [32mMCQ [0m[1;32m[[0m[32mFree PDF[0m[1;32m][0m[32m - Objective Question Answer for Circular Arrangement Quiz - [0m
    [32mDownload Now!_cot_examples.json[0m
    [2KLoading config from: ../tutorial_config.yaml━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m[32m [0m[35m  0%[0m[32m [0m[32m(0/9)[0m[32m [0m[33m0:07:02[0m
    [2KConfig has LLM provider set to: vllm━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m (0/9) [33m0:07:02[0m
    [2KL Using vllm provider90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m (0/9) [33m0:07:02[0m
    [2KGenerated 5 chain-of-thought examples[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 11%[0m (1/9) [33m0:08:11[0m
    [2Krating cot content [91m━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 11%[0m (1/9) [33m0:08:11[0m
    [2KFirst CoT Example: [91m━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 11%[0m (1/9) [33m0:08:11[0m
    [2KQuestion: In a circular seating arrangement of 12 people, A insists on sitting ━[0m [35m 11%[0m (1/9) [33m0:08:11[0m
    next to B, but C cannot sit next to either A or B. D must sit directly opposite 
    E, and F must sit two seats to the left of G. H is sitting three seats to the 
    right of I, and J is sitting two seats to the right of K. Given these complex 
    constraints, determine which pair of individuals cannot sit together in any 
    valid arrangement.
    [2KReasoning (first 100 chars): Step 1: Analyze the circular seating arrangement ━━[0m [35m 11%[0m (1/9) [33m0:08:11[0m
    and identify the constraints. Step 2: A must sit ne...
    [2KAnswer: Based on the constraints, C and A cannot sit together in any valid ━━━━━[0m [35m 11%[0m (1/9) [33m0:08:11[0m
    arrangement because their positioning would create an irresolvable conflict with
    the other constraints.
    [2K[32m✓ Generated cot from SA_Set_No_173.txt -> SA_Set_No_173_cot_examples.json[0m0m [35m 11%[0m (1/9) [33m0:08:11[0m
    [2KLoading config from: ../tutorial_config.yaml━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m[32m [0m[35m 11%[0m[32m [0m[32m(1/9)[0m[32m [0m[33m0:08:11[0m
    [2KConfig has LLM provider set to: vllmm╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 11%[0m (1/9) [33m0:08:11[0m
    [2KL Using vllm provider91m━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 11%[0m (1/9) [33m0:08:11[0m
    [2KProcessing 33 chunks to generate CoT examples...0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m0:08:16[0m
    [2KBatch processing complete.1m━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m0:23:46[0m
    [2KGenerated 50 CoT examples total (requested: 50)[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m0:23:46[0m
    [2KGenerated 50 chain-of-thought examples90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m0:23:46[0m
    [2Krating cot content [91m━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m0:23:46[0m
    [2KFirst CoT Example: [91m━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m0:23:46[0m
    [2KQuestion: In a complex circular seating arrangement of 12 people, A insists on ━[0m [35m 22%[0m (2/9) [33m0:23:46[0m
    sitting next to B, but B refuses to sit next to C. D must sit directly opposite 
    A, while E must sit next to D. F is seated between G and H, and I is seated 
    between J and K. Given these constraints, determine which pair cannot be seated 
    together in any valid arrangement.
    [2KReasoning (first 100 chars): Step 1: Analyze the given constraints and identify [0m [35m 22%[0m (2/9) [33m0:23:46[0m
    the key relationships between individuals. Step 2...
    [2KAnswer: The pair that cannot be seated together is B and C, as their positioning[0m [35m 22%[0m (2/9) [33m0:23:46[0m
    would create an irresolvable conflict with A's adjacency requirement and B's 
    separation constraint.
    [2K[32m✓ Generated cot from Seating_Arrangement_PDF_1.txt -> [0m━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m0:23:46[0m
    [32mSeating_Arrangement_PDF_1_cot_examples.json[0m
    [2KLoading config from: ../tutorial_config.yaml━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m[32m [0m[35m 22%[0m[32m [0m[32m(2/9)[0m[32m [0m[33m0:23:46[0m
    [2KConfig has LLM provider set to: vllm[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m0:23:46[0m
    [2KL Using vllm provider91m━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 22%[0m (2/9) [33m0:23:46[0m
    [2KProcessing 73 chunks to generate CoT examples...m╸[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m0:23:51[0m
    [2KBatch processing complete.1m━━━━━━━━━━━━[0m[91m╸[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m0:37:09[0m
    [2KGenerated 50 CoT examples total (requested: 50)[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m0:37:09[0m
    [2KGenerated 50 chain-of-thought examples0m[91m╸[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m0:37:09[0m
    [2Krating cot content [91m━━━━━━━━━━━━[0m[91m╸[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m0:37:09[0m
    [2KFirst CoT Example: [91m━━━━━━━━━━━━[0m[91m╸[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m0:37:09[0m
    [2KQuestion: In a circular arrangement of 8 ministers around a table, each ━━━━━━━━[0m [35m 33%[0m (3/9) [33m0:37:09[0m
    representing a different ministry, the following conditions apply: Radha sits 
    second to the right of the representative of Railways, and the representatives 
    of Food Processing Industries and Railways are immediate neighbors. Two people 
    sit between the representative of Food Processing Industries and Nitin. Prabhu 
    and Prakash are immediate neighbors of each other, but neither is an immediate 
    neighbor of Nitin or the representative of Food Processing Industries. The 
    representative of Defence sits second to the right of Paswan, and Ravi and the 
    representative of Agriculture are immediate neighbors of each other. Nitin is 
    not the representative of Agriculture, and only one person sits between Prabhu 
    and the representative of Finance. Shankar sits third to the left of the 
    representative of Law & Order, and the representative from HRD sits second to 
    the left of the representative of Health and Family Welfare. Determine the 
    seating arrangement and identify who sits between Nitin and the representative 
    from Food Processing Industries.
    [2KReasoning (first 100 chars): Step 1: Analyze the given conditions to identify ━━[0m [35m 33%[0m (3/9) [33m0:37:09[0m
    the relationships between the ministers and their r...
    [2KAnswer: Shankar and Ravi sit between Nitin and the representative from Food ━━━━[0m [35m 33%[0m (3/9) [33m0:37:09[0m
    Processing Industries.
    [2K[32m✓ Generated cot from Seating_Arrangement_PDF_2.txt -> [0m━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m0:37:09[0m
    [32mSeating_Arrangement_PDF_2_cot_examples.json[0m
    [2KLoading config from: ../tutorial_config.yaml━━━━━━━━━━[0m[91m╸[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m[32m [0m[35m 33%[0m[32m [0m[32m(3/9)[0m[32m [0m[33m0:37:09[0m
    [2KConfig has LLM provider set to: vllm[0m[91m╸[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m0:37:09[0m
    [2KL Using vllm provider91m━━━━━━━━━━━━[0m[91m╸[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m 33%[0m (3/9) [33m0:37:09[0m
    [2KProcessing 30 chunks to generate CoT examples...[91m╸[0m[90m━━━━━━━━━━━━━━━━━━━━━[0m [35m 44%[0m (4/9) [33m0:37:14[0m
    [2KBatch processing complete.1m━━━━━━━━━━━━━━━━[0m[91m╸[0m[90m━━━━━━━━━━━━━━━━━━━━━[0m [35m 44%[0m (4/9) [33m0:51:15[0m
    [2KGenerated 50 CoT examples total (requested: 50)1m╸[0m[90m━━━━━━━━━━━━━━━━━━━━━[0m [35m 44%[0m (4/9) [33m0:51:15[0m
    [2KGenerated 50 chain-of-thought examples━━[0m[91m╸[0m[90m━━━━━━━━━━━━━━━━━━━━━[0m [35m 44%[0m (4/9) [33m0:51:15[0m
    [2Krating cot content [91m━━━━━━━━━━━━━━━━[0m[91m╸[0m[90m━━━━━━━━━━━━━━━━━━━━━[0m [35m 44%[0m (4/9) [33m0:51:15[0m
    [2KFirst CoT Example: [91m━━━━━━━━━━━━━━━━[0m[91m╸[0m[90m━━━━━━━━━━━━━━━━━━━━━[0m [35m 44%[0m (4/9) [33m0:51:15[0m
    [2KQuestion: In a circular arrangement of 12 people around a conference table, A ━━[0m [35m 44%[0m (4/9) [33m0:51:15[0m
    insists on sitting next to both B and C, but B and C have a personal conflict 
    and refuse to sit next to each other. D, who is A's business partner, must sit 
    directly opposite A. E, D's personal assistant, must sit next to D. F is 
    mediating between two executives, G and H, who have specific seating 
    requirements relative to each other. Given these complex constraints, 
    demonstrate through systematic analysis which specific pair cannot be seated 
    together in any valid arrangement.
    [2KReasoning (first 100 chars): Step 1: Analyze the circular arrangement and ━━━━━━[0m [35m 44%[0m (4/9) [33m0:51:15[0m
    identify the constraints. A must sit next to B and C, b...
    [2KAnswer: Based on the constraints, B and C cannot be seated together in any valid[0m [35m 44%[0m (4/9) [33m0:51:15[0m
    arrangement because their positioning would create an irresolvable conflict with
    A's dual adjacency requirement, B and C's separation constraint, D's opposite 
    positioning, E's adjacency to D, and F's mediation between G and H.
    [2K[32m✓ Generated cot from Seating_Arrangement_PDF_3.txt -> [0m━━━━━━━━━━━━━━━━━[0m [35m 44%[0m (4/9) [33m0:51:15[0m
    [32mSeating_Arrangement_PDF_3_cot_examples.json[0m
    [2KLoading config from: ../tutorial_config.yaml━━━━━━━━━━━━━━[0m[91m╸[0m[90m━━━━━━━━━━━━━━━━━━━━━[0m[32m [0m[35m 44%[0m[32m [0m[32m(4/9)[0m[32m [0m[33m0:51:15[0m
    [2KConfig has LLM provider set to: vllm━━━━[0m[91m╸[0m[90m━━━━━━━━━━━━━━━━━━━━━[0m [35m 44%[0m (4/9) [33m0:51:15[0m
    [2KL Using vllm provider91m━━━━━━━━━━━━━━━━[0m[91m╸[0m[90m━━━━━━━━━━━━━━━━━━━━━[0m [35m 44%[0m (4/9) [33m0:51:15[0m
    [2KGenerated 5 chain-of-thought examples━━━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━[0m [35m 56%[0m (5/9) [33m0:52:19[0m
    [2Krating cot content [91m━━━━━━━━━━━━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━[0m [35m 56%[0m (5/9) [33m0:52:19[0m
    [2KFirst CoT Example: [91m━━━━━━━━━━━━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━[0m [35m 56%[0m (5/9) [33m0:52:19[0m
    [2KQuestion: In a circular seating arrangement of 8 people, A insists on sitting ━━[0m [35m 56%[0m (5/9) [33m0:52:19[0m
    next to B, but B refuses to sit next to C. D must sit directly opposite A, while
    E must sit next to D. F is mediating between G and H, and G has specific seating
    requirements relative to H. Given these complex constraints, determine which 
    specific pair cannot be seated together in any valid arrangement.
    [2KReasoning (first 100 chars): Step 1: Analyze the circular seating arrangement ━━[0m [35m 56%[0m (5/9) [33m0:52:19[0m
    with multiple constraints. Step 2: A's requirement ...
    [2KAnswer: Based on the geometric constraints and complex interpersonal ━━━━━━━━━━━[0m [35m 56%[0m (5/9) [33m0:52:19[0m
    requirements, B and C cannot sit together under any circumstances.
    [2K[32m✓ Generated cot from blood-relation-puzzle.txt -> [0m[90m━━━━━━━━━━━━━━━━[0m [35m 56%[0m (5/9) [33m0:52:19[0m
    [32mblood-relation-puzzle_cot_examples.json[0m
    [2KLoading config from: ../tutorial_config.yaml━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━[0m [35m 56%[0m (5/9) [33m0:52:19[0m[32m [0m[32m(5/9)[0m[32m [0m[33m0:52:19[0m
    [2KConfig has LLM provider set to: vllm━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━[0m [35m 56%[0m (5/9) [33m0:52:19[0m
    [2KL Using vllm provider91m━━━━━━━━━━━━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━[0m [35m 56%[0m (5/9) [33m0:52:19[0m
    [2KProcessing 7 chunks to generate CoT examples...━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━[0m [35m 67%[0m (6/9) [33m0:52:24[0m
    [2KBatch processing complete.1m━━━━━━━━━━━━━━━━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━[0m [35m 67%[0m (6/9) [33m0:59:37[0m
    [2KGenerated 34 CoT examples total (requested: 50)━━[0m[90m╺[0m[90m━━━━━━━━━━━━[0m [35m 67%[0m (6/9) [33m0:59:37[0m
    [2KGenerated 34 chain-of-thought examples━━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━[0m [35m 67%[0m (6/9) [33m0:59:37[0m
    [2Krating cot content [91m━━━━━━━━━━━━━━━━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━[0m [35m 67%[0m (6/9) [33m0:59:37[0m
    [2KFirst CoT Example: [91m━━━━━━━━━━━━━━━━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━[0m [35m 67%[0m (6/9) [33m0:59:37[0m
    [2KQuestion: In a circular seating arrangement of 12 people, A insists on sitting ━[0m [35m 67%[0m (6/9) [33m0:59:37[0m
    next to B, but C refuses to sit next to either A or B. D must sit directly 
    opposite E, while F and G must sit next to each other. H has a specific seating 
    requirement relative to I, and J must sit at least three seats away from K. 
    Given these complex constraints, determine which specific pair cannot be seated 
    together in any valid arrangement.
    [2KReasoning (first 100 chars): Step 1: Analyze the constraints to identify the ━━━[0m [35m 67%[0m (6/9) [33m0:59:37[0m
    most restrictive ones. A's requirement to sit next t...
    [2KAnswer: Based on the complex constraints and systematic analysis, C and A cannot[0m [35m 67%[0m (6/9) [33m0:59:37[0m
    sit together under any circumstances because their positioning would create an 
    irresolvable conflict with B's adjacency requirement, D's opposite positioning, 
    F and G's unit, and H's specific requirement relative to I.
    [2K[32m✓ Generated cot from blood-relationship.txt -> [0m0m╺[0m[90m━━━━━━━━━━━━[0m [35m 67%[0m (6/9) [33m0:59:37[0m
    [32mblood-relationship_cot_examples.json[0m
    [2KLoading config from: ../tutorial_config.yaml━━━━━━━━━━━━━━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━[0m[32m [0m[35m 67%[0m[32m [0m[32m(6/9)[0m[32m [0m[33m0:59:37[0m
    [2KConfig has LLM provider set to: vllm━━━━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━[0m [35m 67%[0m (6/9) [33m0:59:37[0m
    [2KL Using vllm provider91m━━━━━━━━━━━━━━━━━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━[0m [35m 67%[0m (6/9) [33m0:59:37[0m
    [2KProcessing 9 chunks to generate CoT examples...━━━━━━━━━━[0m[91m╸[0m[90m━━━━━━━━[0m [35m 78%[0m (7/9) [33m0:59:41[0m
    [2KBatch processing complete.1m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m[91m╸[0m[90m━━━━━━━━[0m [35m 78%[0m (7/9) [33m1:06:56[0m
    [2KGenerated 33 CoT examples total (requested: 50)━━━━━━[0m[91m╸[0m[90m━━━━━━━━[0m [35m 78%[0m (7/9) [33m1:06:56[0m
    [2KGenerated 33 chain-of-thought examples━━━━━━━━━━━━━━━[0m[91m╸[0m[90m━━━━━━━━[0m [35m 78%[0m (7/9) [33m1:06:56[0m
    [2Krating cot content [91m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m[91m╸[0m[90m━━━━━━━━[0m [35m 78%[0m (7/9) [33m1:06:56[0m
    [2KFirst CoT Example: [91m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m[91m╸[0m[90m━━━━━━━━[0m [35m 78%[0m (7/9) [33m1:06:56[0m
    [2KQuestion: Eight persons from three generations – P, Q, R, S, T, U, V, and W are [0m [35m 78%[0m (7/9) [33m1:06:56[0m
    sitting around a circular table facing the center. Two married couples are in 
    the family. Either both or none of the parents are alive. What is the position 
    of R’s sister with respect to S’s sister?
    [2KReasoning (first 100 chars): Step 1: Let's start by analyzing the given [90m━━━━━━━━[0m [35m 78%[0m (7/9) [33m1:06:56[0m
    information. We have eight persons from three generations...
    [2KAnswer: Immediate rightm━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m[91m╸[0m[90m━━━━━━━━[0m [35m 78%[0m (7/9) [33m1:06:56[0m
    [2K[32m✓ Generated cot from circular-arrangement-with-blood-relation.txt -> [0m━━[0m [35m 78%[0m (7/9) [33m1:06:56[0m
    [32mcircular-arrangement-with-blood-relation_cot_examples.json[0m
    [2KLoading config from: ../tutorial_config.yaml━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m[91m╸[0m[90m━━━━━━━━[0m[32m [0m[35m 78%[0m[32m [0m[32m(7/9)[0m[32m [0m[33m1:06:56[0m
    [2KConfig has LLM provider set to: vllm━━━━━━━━━━━━━━━━━[0m[91m╸[0m[90m━━━━━━━━[0m [35m 78%[0m (7/9) [33m1:06:56[0m
    [2KL Using vllm provider91m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m[91m╸[0m[90m━━━━━━━━[0m [35m 78%[0m (7/9) [33m1:06:56[0m
    [2KProcessing 5 chunks to generate CoT examples...━━━━━━━━━━━━━━[0m[91m╸[0m[90m━━━━[0m [35m 89%[0m (8/9) [33m1:07:01[0m
    [2KBatch processing complete.1m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m[91m╸[0m[90m━━━━[0m [35m 89%[0m (8/9) [33m1:11:03[0m
    [2KGenerated 22 CoT examples total (requested: 50)━━━━━━━━━━[0m[91m╸[0m[90m━━━━[0m [35m 89%[0m (8/9) [33m1:11:03[0m
    [2KGenerated 22 chain-of-thought examples━━━━━━━━━━━━━━━━━━━[0m[91m╸[0m[90m━━━━[0m [35m 89%[0m (8/9) [33m1:11:03[0m
    [2Krating cot content [91m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m[91m╸[0m[90m━━━━[0m [35m 89%[0m (8/9) [33m1:11:03[0m
    [2KFirst CoT Example: [91m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m[91m╸[0m[90m━━━━[0m [35m 89%[0m (8/9) [33m1:11:03[0m
    [2KQuestion: In a circular arrangement of 12 people around a conference table, A ━━[0m [35m 89%[0m (8/9) [33m1:11:03[0m
    demands to sit next to both B and C, but B and C have an irreconcilable personal
    conflict and refuse to sit next to each other under any circumstances. D, who is
    A's business partner, must sit directly opposite A. E, D's personal assistant, 
    must sit next to D. F is mediating between two specific executives, G and H, who
    have conflicting interests. Given these complex interpersonal and positional 
    constraints, demonstrate through systematic analysis which specific pair cannot 
    be seated together in any valid arrangement.
    [2KReasoning (first 100 chars): Step 1: Analyze the circular arrangement and the ━━[0m [35m 89%[0m (8/9) [33m1:11:03[0m
    constraints provided. A must sit next to B and C, b...
    [2KAnswer: Based on the geometric constraints of the 12-person circular table and ━[0m [35m 89%[0m (8/9) [33m1:11:03[0m
    the complex interpersonal requirements, B and C cannot sit together under any 
    circumstances because their positioning would create an irresolvable conflict 
    with A's dual adjacency requirement, B and C's separation constraint, D's 
    opposite positioning, E's adjacency to D, and F's mediating role.
    [2K[32m✓ Generated cot from seating-arrangement-logical-reasoning-.txt -> [0m━━━━[0m [35m 89%[0m (8/9) [33m1:11:03[0m
    [32mseating-arrangement-logical-reasoning-_cot_examples.json[0m
    [2KGenerating cot content [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m100%[0m (9/9) [33m1:11:03[0m [0m[35m 89%[0m[32m [0m[32m(8/9)[0m[32m [0m[33m1:11:03[0m
    [?25h
    [1m==================================================[0m
    [1;34mContent Generation Summary [0m[1;34m([0m[1;34mcot[0m[1;34m)[0m[1;34m:[0m
    Total files: [1;36m9[0m
    [32mSuccessful: [0m[1;36m9[0m
    [32mFailed: [0m[1;36m0[0m
    [1m==================================================[0m
    [32m✅ All files processed successfully![0m
    Loading config from: /usr/local/lib/python3.12/dist-packages/synthetic_data_kit/config.yaml
    Config has LLM provider set to: api-endpoint
    Loading config from: /usr/local/lib/python3.12/dist-packages/synthetic_data_kit/config.yaml
    Config has LLM provider set to: api-endpoint
    Loading config from: ../tutorial_config.yaml
    Config has LLM provider set to: vllm
    get_llm_provider returning: vllm
    [32m🔗 Using vllm provider[0m
    [34mProcessing directory: [0m[1;34m.[0m[1;35m/data/generated/[0m[34m for curation[0m
    [34mFound [0m[1;36m19[0m[34m JSON files to curate[0m
    [31m✗ Circular Arrangement MCQ [0m[1;31m[[0m[31mFree PDF[0m[1;31m][0m[31m - Objective Question Answer for Circular [0m
    [31mArrangement Quiz - Download Now!_cot_examples.json: No QA pairs found in the [0m
    [31minput file[0m
    Loading config from: ../tutorial_config.yaml
    Config has LLM provider set to: vllm
    Loading config from: ../tutorial_config.yaml
    Config has LLM provider set to: vllm
    Processing 34 batches of QA pairs...
    Batch processing complete.                                                      
    Rated 167 QA pairs
    Retained 139 pairs (threshold: 7.0)
    Average score: 7.6
    [32m✓ Circular Arrangement MCQ [0m[1;32m[[0m[32mFree PDF[0m[1;32m][0m[32m - Objective Question Answer for Circular [0m
    [32mArrangement Quiz - Download Now!_qa_pairs.json[0m
    [31m✗ SA_Set_No_173_cot_examples.json: No QA pairs found in the input file[0m
    Loading config from: ../tutorial_config.yaml
    Config has LLM provider set to: vllm
    Loading config from: ../tutorial_config.yaml
    Config has LLM provider set to: vllm
    Processing 16 batches of QA pairs...
    Batch processing complete.                                                      
    Rated 77 QA pairs
    Retained 33 pairs (threshold: 7.0)
    Average score: 5.7
    [32m✓ SA_Set_No_173_qa_pairs.json[0m
    [31m✗ Seating_Arrangement_PDF_1_cot_examples.json: No QA pairs found in the input [0m
    [31mfile[0m
    Loading config from: ../tutorial_config.yaml
    Config has LLM provider set to: vllm
    Loading config from: ../tutorial_config.yaml
    Config has LLM provider set to: vllm
    Processing 33 batches of QA pairs...
    Batch processing complete.                                                      
    Rated 165 QA pairs
    Retained 146 pairs (threshold: 7.0)
    Average score: 7.7
    [32m✓ Seating_Arrangement_PDF_1_qa_pairs.json[0m
    [31m✗ Seating_Arrangement_PDF_2_cot_examples.json: No QA pairs found in the input [0m
    [31mfile[0m
    Loading config from: ../tutorial_config.yaml
    Config has LLM provider set to: vllm
    Loading config from: ../tutorial_config.yaml
    Config has LLM provider set to: vllm
    Processing 30 batches of QA pairs...
    Batch processing complete.                                                      
    Rated 146 QA pairs
    Retained 130 pairs (threshold: 7.0)
    Average score: 7.8
    [32m✓ Seating_Arrangement_PDF_2_qa_pairs.json[0m
    [31m✗ Seating_Arrangement_PDF_3_cot_examples.json: No QA pairs found in the input [0m
    [31mfile[0m
    Loading config from: ../tutorial_config.yaml
    Config has LLM provider set to: vllm
    Loading config from: ../tutorial_config.yaml
    Config has LLM provider set to: vllm
    Processing 34 batches of QA pairs...
    Batch processing complete.                                                      
    Rated 167 QA pairs
    Retained 143 pairs (threshold: 7.0)
    Average score: 7.7
    [32m✓ Seating_Arrangement_PDF_3_qa_pairs.json[0m
    [31m✗ blood-relation-puzzle_cot_examples.json: No QA pairs found in the input file[0m
    Loading config from: ../tutorial_config.yaml
    Config has LLM provider set to: vllm
    Loading config from: ../tutorial_config.yaml
    Config has LLM provider set to: vllm
    Processing 4 batches of QA pairs...
    Batch processing complete.                                                      
    Rated 17 QA pairs
    Retained 15 pairs (threshold: 7.0)
    Average score: 7.7
    [32m✓ blood-relation-puzzle_qa_pairs.json[0m
    [31m✗ blood-relationship_cot_examples.json: No QA pairs found in the input file[0m
    Loading config from: ../tutorial_config.yaml
    Config has LLM provider set to: vllm
    Loading config from: ../tutorial_config.yaml
    Config has LLM provider set to: vllm
    Processing 30 batches of QA pairs...
    Batch processing complete.                                                      
    Rated 146 QA pairs
    Retained 106 pairs (threshold: 7.0)
    Average score: 7.1
    [32m✓ blood-relationship_qa_pairs.json[0m
    [31m✗ circular-arrangement-with-blood-relation_cot_examples.json: No QA pairs found [0m
    [31min the input file[0m
    Loading config from: ../tutorial_config.yaml
    Config has LLM provider set to: vllm
    Loading config from: ../tutorial_config.yaml
    Config has LLM provider set to: vllm
    Processing 32 batches of QA pairs...
    Batch processing complete.                                                      
    Rated 160 QA pairs
    Retained 127 pairs (threshold: 7.0)
    Average score: 7.4
    [32m✓ circular-arrangement-with-blood-relation_qa_pairs.json[0m
    [31m✗ seating-arrangement-logical-reasoning-_cot_examples.json: No QA pairs found in[0m
    [31mthe input file[0m
    Loading config from: ../tutorial_config.yaml
    Config has LLM provider set to: vllm
    Loading config from: ../tutorial_config.yaml
    Config has LLM provider set to: vllm
    Processing 21 batches of QA pairs...
    Batch processing complete.                                                      
    Rated 102 QA pairs
    Retained 86 pairs (threshold: 7.0)
    Average score: 7.5
    [32m✓ seating-arrangement-logical-reasoning-_qa_pairs.json[0m
    [31m✗ test_write.json: No QA pairs found in the input file[0m
    
    [1m==================================================[0m
    [1;34mCuration Summary [0m[1;34m([0m[1;34mthreshold: [0m[1;36m7.0[0m[1;34m)[0m[1;34m:[0m
    Total files: [1;36m19[0m
    [32mSuccessful: [0m[1;36m9[0m
    [31mFailed: [0m[1;36m10[0m
    [1m==================================================[0m
    [33m⚠️  Completed with [0m[1;36m10[0m[33m errors[0m
    Loading config from: /usr/local/lib/python3.12/dist-packages/synthetic_data_kit/config.yaml
    Config has LLM provider set to: api-endpoint
    Loading config from: /usr/local/lib/python3.12/dist-packages/synthetic_data_kit/config.yaml
    Config has LLM provider set to: api-endpoint
    Loading config from: /usr/local/lib/python3.12/dist-packages/synthetic_data_kit/config.yaml
    Config has LLM provider set to: api-endpoint
    [34mProcessing directory: [0m[1;34m.[0m[1;35m/data/curated/[0m[34m for format conversion to ft[0m
    [34mFound [0m[1;36m9[0m[34m JSON files to convert to ft format[0m
    [2K[32m✓ Converted Circular Arrangement MCQ [0m[1;32m[[0m[32mFree PDF[0m[1;32m][0m[32m - Objective Question Answer for [0m
    [32mCircular Arrangement Quiz - Download Now!_qa_pairs_cleaned.json -> Circular [0m
    [32mArrangement MCQ [0m[1;32m[[0m[32mFree PDF[0m[1;32m][0m[32m - Objective Question Answer for Circular Arrangement [0m
    [32mQuiz - Download Now!_qa_pairs_cleaned_ft.json [0m[1;32m([0m[32mft, json[0m[1;32m)[0m
    [2K[32m✓ Converted SA_Set_No_173_qa_pairs_cleaned.json -> [0m━━━━━━━━━━━━━━━━━━━━[0m[32m [0m[35m  0%[0m[32m [0m[32m(0/9)[0m[32m [0m[33m0:00:00[0m
    [32mSA_Set_No_173_qa_pairs_cleaned_ft.json [0m[1;32m([0m[32mft, json[0m[1;32m)[0m
    [2K[32m✓ Converted Seating_Arrangement_PDF_1_qa_pairs_cleaned.json -> [0m━━━━━━━━[0m[32m [0m[35m  0%[0m[32m [0m[32m(0/9)[0m[32m [0m[33m0:00:00[0m
    [32mSeating_Arrangement_PDF_1_qa_pairs_cleaned_ft.json [0m[1;32m([0m[32mft, json[0m[1;32m)[0m
    [2K[32m✓ Converted Seating_Arrangement_PDF_2_qa_pairs_cleaned.json -> [0m━━━━━━━━[0m[32m [0m[35m  0%[0m[32m [0m[32m(0/9)[0m[32m [0m[33m0:00:00[0m
    [32mSeating_Arrangement_PDF_2_qa_pairs_cleaned_ft.json [0m[1;32m([0m[32mft, json[0m[1;32m)[0m
    [2K[32m✓ Converted Seating_Arrangement_PDF_3_qa_pairs_cleaned.json -> [0m━━━━━━━━[0m[32m [0m[35m  0%[0m[32m [0m[32m(0/9)[0m[32m [0m[33m0:00:00[0m
    [32mSeating_Arrangement_PDF_3_qa_pairs_cleaned_ft.json [0m[1;32m([0m[32mft, json[0m[1;32m)[0m
    [2K[32m✓ Converted blood-relation-puzzle_qa_pairs_cleaned.json -> [0m━━━━━━━━━━━━[0m[32m [0m[35m  0%[0m[32m [0m[32m(0/9)[0m[32m [0m[33m0:00:00[0m
    [32mblood-relation-puzzle_qa_pairs_cleaned_ft.json [0m[1;32m([0m[32mft, json[0m[1;32m)[0m
    [2K[32m✓ Converted blood-relationship_qa_pairs_cleaned.json -> [0m━━━━━━━━━━━━━━━[0m[32m [0m[35m  0%[0m[32m [0m[32m(0/9)[0m[32m [0m[33m0:00:00[0m
    [32mblood-relationship_qa_pairs_cleaned_ft.json [0m[1;32m([0m[32mft, json[0m[1;32m)[0m
    [2K[32m✓ Converted circular-arrangement-with-blood-relation_qa_pairs_cleaned.json -> [0m2m [0m[35m  0%[0m[32m [0m[32m(0/9)[0m[32m [0m[33m0:00:00[0m
    [32mcircular-arrangement-with-blood-relation_qa_pairs_cleaned_ft.json [0m[1;32m([0m[32mft, json[0m[1;32m)[0m
    [2K[32m✓ Converted seating-arrangement-logical-reasoning-_qa_pairs_cleaned.json -> [0m[32m [0m[35m  0%[0m[32m [0m[32m(0/9)[0m[32m [0m[33m0:00:00[0m
    [32mseating-arrangement-logical-reasoning-_qa_pairs_cleaned_ft.json [0m[1;32m([0m[32mft, json[0m[1;32m)[0m
    [2KConverting to ft format [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m100%[0m (9/9) [33m0:00:00[0m[32m [0m[32m(0/9)[0m[32m [0m[33m0:00:00[0m
    [?25h
    [1m==================================================[0m
    [1;34mFormat Conversion Summary [0m[1;34m([0m[1;34mft, json[0m[1;34m)[0m[1;34m:[0m
    Total files: [1;36m9[0m
    [32mSuccessful: [0m[1;36m9[0m
    [32mFailed: [0m[1;36m0[0m
    [1m==================================================[0m
    [32m✅ All files converted successfully![0m


### Generating Synthetic Q&A Pairs

This command uses the synthetic-data-kit's **create** command to generate Q&A pairs from the parsed text:
- Reads parsed text files from `data/parsed/`
- Uses the vLLM provider with Llama-3.3-70B-Instruct model
- Generates 50 Q&A pairs per file (`--num-pairs 50`)
- Type is set to `qa` for question-answer pair generation
- Outputs are saved to `data/generated/`

The process chunks the text and generates questions with corresponding answers. This took about 10 minutes for the full run. Use `--verbose` flag to see detailed progress or reduce `--num-pairs` for faster testing.


```python
# !synthetic-data-kit -c ../tutorial_config.yaml curate ./data/generated/ --threshold 7.0
```

### Curating and Quality Filtering

This command uses the **curate** function to filter generated Q&A pairs based on quality:
- Evaluates each Q&A pair using quality metrics
- Filters pairs with quality score above threshold (7.0/10)
- Removes low-quality, inconsistent, or malformed pairs
- Saves curated data to `data/curated/`

This ensures only high-quality synthetic data is used for fine-tuning.


```python
# !synthetic-data-kit save-as ./data/curated/ --format ft
```

### Converting to Fine-Tuning Format

This command uses the **save-as** function to convert curated Q&A pairs to fine-tuning format:
- Reads curated JSON files from `data/curated/`
- Converts to format `ft` (fine-tuning format with messages structure)
- Outputs are saved to `data/final/` with proper conversation format
- The resulting format is compatible with standard fine-tuning pipelines

Successfully converted 2 files to fine-tuning format.


```python
from huggingface_hub.utils import disable_progress_bars
disable_progress_bars()
```


```python
cd logical_reasoning
```

    /workspace/AIAC/logical_reasoning



```python
import json
import glob
from pathlib import Path
from datasets import Dataset

# ===== CONFIGURATION =====
data_dir = "./data/final"  # Change this to your data directory

# ===== STEP 1: Find all FT files =====
data_path = Path(data_dir)
ft_files = glob.glob(str(data_path / "*.json"))

# ===== STEP 2: Load and convert all files =====
all_data = []

for file_path in ft_files:
    # Load the JSON file
    with open(file_path, 'r') as f:
        ft_data = json.load(f)
    
    # Convert each item
    for item in ft_data:
        if 'messages' not in item:
            continue
        
        # Extract only user and assistant messages
        conversation = []
        for msg in item['messages']:
            if msg['role'] == 'user' or msg['role'] == 'assistant':
                conversation.append({
                    "role": msg['role'],
                    "content": msg['content']
                })
        
        # Add to our data if we have at least one exchange
        if len(conversation) > 0:
            all_data.append({
                "conversations": conversation
            })

print(f"\n🎯 Total conversations: {len(all_data)}")

# ===== STEP 3: Create HuggingFace Dataset =====
dataset = Dataset.from_list(all_data)

# ===== STEP 4: Preview the data =====
print(json.dumps(dataset[0], indent=2))
```

    
    🎯 Total conversations: 925
    {
      "conversations": [
        {
          "content": "In a linear arrangement of 7 people - A, B, C, D, E, F, and G - the following conditions apply: C sits next to D, E sits next to F, A sits at one of the ends, and B sits third from the other end. Given that G must sit next to A and that the pair CD must be separated from the pair EF by at least one person, who sits in the middle position?",
          "role": "user"
        },
        {
          "content": "Step 1: Determine the possible positions for A and B, given that A is at one end and B is third from the other end. This gives us two initial configurations: A _ _ _ _ _ B and B _ _ _ _ _ A. Step 2: Place G next to A in both configurations, as G must sit next to A. This results in A G _ _ _ _ B and B _ _ _ _ _ A G. Step 3: Considering the constraint that C and D sit next to each other, and E and F sit next to each other, we can represent these pairs as CD and EF. Step 4: Apply the condition that CD must be separated from EF by at least one person. This means the pairs cannot be adjacent. Step 5: Examine the possible placements of CD and EF in both configurations, ensuring they are separated and that all other conditions are met. Step 6: In the configuration A G _ _ _ _ B, to satisfy the separation condition and the constraint that B is third from the other end, CD and EF must be placed with a gap between them, which can be achieved by placing one pair next to G and the other pair closer to B, but not adjacent. Step 7: Considering the requirement for a middle position and the constraints, the only feasible arrangement that allows for all conditions to be met and for the pairs to be separated by at least one person is when one of the pairs (either CD or EF) is at the second and third positions from A, and the other pair is at the fourth and fifth positions from A, with the remaining person (either part of the pair or a single person) in the middle. Step 8: Given the constraints and the need for CD and EF to be separated, and considering A and G's fixed positions, the middle position must be occupied by the person who is not part of the CD or EF pairs and is not A, B, or G, which leaves us with the person who can fit without violating any given conditions.",
          "role": "assistant"
        }
      ]
    }


### Loading and Converting Data to HuggingFace Dataset

This cell performs comprehensive data processing:

1. **Finding Files**: Locates all JSON files in `data/final/` directory
2. **Loading Data**: Reads each JSON file containing fine-tuning formatted data
3. **Format Conversion**: Extracts user and assistant messages from the fine-tuning format
4. **Structuring Conversations**: Creates a standardized conversation format with role-content pairs
5. **Creating Dataset**: Converts the processed data into a HuggingFace Dataset object

The output shows 74 total conversations were successfully loaded and formatted. The preview displays a sample conversation showing a knight-and-knave logic puzzle with its solution.

## Fine-Tuning

### Note: Please remember to shutdown the vLLM instance!


```python
!pip install --upgrade -qqq --no-cache-dir --force-reinstall --no-deps unsloth unsloth_zoo
!python -c "import unsloth; print(unsloth.__version__)"
```

    🦥 Unsloth: Will patch your computer to enable 2x faster free finetuning.
    INFO 10-29 05:56:20 [__init__.py:225] Automatically detected platform rocm.
    🦥 Unsloth Zoo will now patch everything to make training faster!
    2025.10.11



```python
import os
import json
import glob
import torch
import shutil
from pathlib import Path
from datasets import Dataset
```

### Importing Standard Libraries

Imports essential Python libraries for fine-tuning:
- `os`, `json`, `glob`: File system operations and JSON handling
- `torch`: PyTorch deep learning framework
- `shutil`: File operations
- `Path`: Path manipulation
- `Dataset`: HuggingFace datasets library for data handling


```python
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, standardize_sharegpt, train_on_responses_only
from trl import SFTConfig, SFTTrainer
from transformers import DataCollatorForSeq2Seq
```

    🦥 Unsloth: Will patch your computer to enable 2x faster free finetuning.
    INFO 10-29 05:56:48 [__init__.py:225] Automatically detected platform rocm.
    🦥 Unsloth Zoo will now patch everything to make training faster!


### Importing Unsloth and Training Libraries

Imports specialized libraries for efficient fine-tuning:
- `FastLanguageModel` from Unsloth: Optimized model loading and training
- `get_chat_template`, `standardize_sharegpt`, `train_on_responses_only`: Chat formatting utilities
- `SFTConfig`, `SFTTrainer`: Supervised fine-tuning configuration and trainer from TRL
- `DataCollatorForSeq2Seq`: Handles batching and padding for sequence-to-sequence training

### Setup Unsloth model and tokenizer for ROCm without bitsandbytes


```python
max_seq_length = 1024
dtype = torch.bfloat16  # Explicit bfloat16 for ROCm
load_in_4bit = False  

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.3-70B-Instruct",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
    device_map="auto",
    torch_dtype=torch.bfloat16,  # Explicit for ROCm
    trust_remote_code=True,
)
```

    Unsloth: WARNING `trust_remote_code` is True.
    Are you certain you want to do remote code execution?
    ==((====))==  Unsloth 2025.10.11: Fast Llama patching. Transformers: 4.56.2. vLLM: 0.11.1rc3.dev39+gf417746ad.rocm700.
       \\   /|    AMD GPU Device. Num GPUs = 1. Max memory: 191.688 GB. Platform: Linux.
    O^O/ \_/ \    Torch: 2.9.0a0+git1c57644. ROCm Toolkit: 7.0.51831-a3e329ad8. Triton: 3.4.0
    \        /    Bfloat16 = TRUE. FA [Xformers = None. FA2 = True]
     "-____-"     Free license: http://github.com/unslothai/unsloth
    Unsloth: Fast downloading is enabled - ignore downloading bars which are red colored!


    `torch_dtype` is deprecated! Use `dtype` instead!



```python
print(f"✅ Loaded: Llama-3.3-70B-Instruct (bfloat16, ROCm compatible)")

# Add LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=64,  # Higher rank for 70B model
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj"],
    lora_alpha=64,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)
```

    ✅ Loaded: Llama-3.3-70B-Instruct (bfloat16, ROCm compatible)


    Unsloth 2025.10.11 patched 80 layers with 80 QKV layers, 80 O layers and 80 MLP layers.


### Loading Llama-3.3-70B Model with LoRA

This cell sets up the model for efficient fine-tuning on AMD ROCm hardware:

**Model Configuration:**
- Model: Llama-3.3-70B-Instruct (70 billion parameters)
- Data type: bfloat16 for ROCm compatibility
- No quantization (load_in_4bit=False) to avoid bitsandbytes dependency
- Max sequence length: 1024 tokens

**LoRA (Low-Rank Adaptation) Configuration:**
- Rank (r): 64 - Higher rank for the large 70B model
- Target modules: All attention and MLP layers (q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj)
- LoRA alpha: 64
- Dropout: 0 (no dropout)
- Gradient checkpointing: "unsloth" for memory efficiency

LoRA enables efficient fine-tuning by only training small adapter layers instead of the entire 70B model, making it feasible to train on a single AMD MI300X GPU with 192GB HBM3 memory.


```python
"""Prepare dataset with proper chat template and tensor compatibility"""
print("🔧 Preparing dataset for training...")

# Set chat template
tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")

# Ensure pad token is set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Formatting function that ensures proper tensor conversion
def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = []
    
    for convo in convos:
        # Ensure conversation is in correct format
        if isinstance(convo, list) and all(isinstance(msg, dict) for msg in convo):
            text = tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False)
            texts.append(text)
        else:
            print(f"⚠️  Skipping malformed conversation: {type(convo)}")
            continue
    
    return {"text": texts}

dataset = standardize_sharegpt(dataset)

dataset = dataset.map(formatting_prompts_func, batched=True, remove_columns=dataset.column_names)

dataset = dataset.filter(lambda x: len(x["text"].strip()) > 0)

print(f"✅ Prepared {len(dataset)} valid examples for training")

# Show sample
if len(dataset) > 0:
    print(f"📝 Sample formatted text:")
    print(dataset["text"][0][:200] + "...")
```

    🔧 Preparing dataset for training...



    Unsloth: Standardizing formats (num_proc=20):   0%|          | 0/925 [00:00<?, ? examples/s]



    Map:   0%|          | 0/925 [00:00<?, ? examples/s]



    Filter:   0%|          | 0/925 [00:00<?, ? examples/s]


    ✅ Prepared 925 valid examples for training
    📝 Sample formatted text:
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>
    
    Cutting Knowledge Date: December 2023
    Today Date: 26 July 2024
    
    <|eot_id|><|start_header_id|>user<|end_header_id|>
    
    In a linear arrangement...


### Preparing Dataset with Chat Template

This cell formats the dataset for fine-tuning:

**Steps:**
1. **Set Chat Template**: Applies Llama-3.1 chat template formatting
2. **Configure Padding**: Sets pad token to eos token if not already set
3. **Format Conversations**: The `formatting_prompts_func` function:
   - Takes raw conversations from the dataset
   - Applies the chat template to format them properly
   - Validates conversation structure (list of dicts with role/content)
   - Filters out malformed conversations
4. **Standardize Format**: Uses `standardize_sharegpt` to normalize the data structure
5. **Apply Formatting**: Maps the formatting function across all examples
6. **Remove Empty**: Filters out any empty or invalid formatted texts

The output shows 74 valid examples were successfully prepared. A sample of the formatted text is displayed, showing the proper Llama-3.1 chat template structure with system, user, and assistant headers.


```python
"""Train model with ROCm-optimized settings"""
# Ensure tokenizer has proper padding
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

# Setup trainer with ROCm-friendly settings and proper data handling
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    packing=False,
    args=SFTConfig(
        per_device_train_batch_size=64,  # 🚀 MI300X can handle this with 192GB HBM3!
        gradient_accumulation_steps=1,   # Effective batch size = 8*2 = 16
        warmup_steps=5,
        num_train_epochs=6,
        learning_rate=1e-4,
        logging_steps=1,
        optim="adamw_8bit",  # Pure torch optimizer
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="logical_reasoning_rocm_outputs",
        report_to="none",
        bf16=True,
        dataloader_pin_memory=False,
        remove_unused_columns=True,  # Remove unused columns to avoid tensor issues
        gradient_checkpointing=True,
        dataloader_num_workers=0,  # Single worker for ROCm stability
    ),
)

# Train only on responses
trainer = train_on_responses_only(
    trainer,
    instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
    response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
)

FastLanguageModel.for_training(model)
trainer_stats = trainer.train()
```


    Unsloth: Tokenizing ["text"] (num_proc=24):   0%|          | 0/925 [00:00<?, ? examples/s]



    Map (num_proc=24):   0%|          | 0/925 [00:00<?, ? examples/s]


    ==((====))==  Unsloth - 2x faster free finetuning | Num GPUs used = 1
       \\   /|    Num examples = 925 | Num Epochs = 3 | Total steps = 45
    O^O/ \_/ \    Batch size per device = 64 | Gradient accumulation steps = 1
    \        /    Data Parallel GPUs = 1 | Total batch size (64 x 1 x 1) = 64
     "-____-"     Trainable parameters = 828,375,040 of 71,382,081,536 (1.16% trained)




    <div>

      <progress value='45' max='45' style='width:300px; height:20px; vertical-align: middle;'></progress>
      [45/45 21:01, Epoch 3/3]
    </div>
    <table border="1" class="dataframe">
  <thead>
 <tr style="text-align: left;">
      <th>Step</th>
      <th>Training Loss</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td>0.900000</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.959200</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.836800</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.904600</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.651300</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.527800</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.503700</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.424800</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.417500</td>
    </tr>
    <tr>
      <td>10</td>
      <td>0.371400</td>
    </tr>
    <tr>
      <td>11</td>
      <td>0.338900</td>
    </tr>
    <tr>
      <td>12</td>
      <td>0.325200</td>
    </tr>
    <tr>
      <td>13</td>
      <td>0.332400</td>
    </tr>
    <tr>
      <td>14</td>
      <td>0.312000</td>
    </tr>
    <tr>
      <td>15</td>
      <td>0.304800</td>
    </tr>
    <tr>
      <td>16</td>
      <td>0.280100</td>
    </tr>
    <tr>
      <td>17</td>
      <td>0.327400</td>
    </tr>
    <tr>
      <td>18</td>
      <td>0.290700</td>
    </tr>
    <tr>
      <td>19</td>
      <td>0.283900</td>
    </tr>
    <tr>
      <td>20</td>
      <td>0.256600</td>
    </tr>
    <tr>
      <td>21</td>
      <td>0.256500</td>
    </tr>
    <tr>
      <td>22</td>
      <td>0.269200</td>
    </tr>
    <tr>
      <td>23</td>
      <td>0.237300</td>
    </tr>
    <tr>
      <td>24</td>
      <td>0.254200</td>
    </tr>
    <tr>
      <td>25</td>
      <td>0.224300</td>
    </tr>
    <tr>
      <td>26</td>
      <td>0.193400</td>
    </tr>
    <tr>
      <td>27</td>
      <td>0.223000</td>
    </tr>
    <tr>
      <td>28</td>
      <td>0.218700</td>
    </tr>
    <tr>
      <td>29</td>
      <td>0.217400</td>
    </tr>
    <tr>
      <td>30</td>
      <td>0.247500</td>
    </tr>
    <tr>
      <td>31</td>
      <td>0.235300</td>
    </tr>
    <tr>
      <td>32</td>
      <td>0.197800</td>
    </tr>
    <tr>
      <td>33</td>
      <td>0.170100</td>
    </tr>
    <tr>
      <td>34</td>
      <td>0.162800</td>
    </tr>
    <tr>
      <td>35</td>
      <td>0.157000</td>
    </tr>
    <tr>
      <td>36</td>
      <td>0.198300</td>
    </tr>
    <tr>
      <td>37</td>
      <td>0.205700</td>
    </tr>
    <tr>
      <td>38</td>
      <td>0.161200</td>
    </tr>
    <tr>
      <td>39</td>
      <td>0.228400</td>
    </tr>
    <tr>
      <td>40</td>
      <td>0.188700</td>
    </tr>
    <tr>
      <td>41</td>
      <td>0.188300</td>
    </tr>
    <tr>
      <td>42</td>
      <td>0.177700</td>
    </tr>
    <tr>
      <td>43</td>
      <td>0.185900</td>
    </tr>
    <tr>
      <td>44</td>
      <td>0.227600</td>
    </tr>
    <tr>
      <td>45</td>
      <td>0.218000</td>
    </tr>
  </tbody>
</table><p>


    Unsloth: Will smartly offload gradients to save VRAM!


    ==((====))==  Unsloth - 2x faster free finetuning | Num GPUs used = 1
       \\   /|    Num examples = 925 | Num Epochs = 3 | Total steps = 45
    O^O/ \_/ \    Batch size per device = 64 | Gradient accumulation steps = 1
    \        /    Data Parallel GPUs = 1 | Total batch size (64 x 1 x 1) = 64
     "-____-"     Trainable parameters = 828,375,040 of 71,382,081,536 (1.16% trained)




    <div>

      <progress value='45' max='45' style='width:300px; height:20px; vertical-align: middle;'></progress>
      [45/45 19:25, Epoch 3/3]
    </div>
    <table border="1" class="dataframe">
  <thead>
 <tr style="text-align: left;">
      <th>Step</th>
      <th>Training Loss</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td>0.172600</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.194800</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.155000</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.168900</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.174200</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.175300</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.189800</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.171600</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.162900</td>
    </tr>
    <tr>
      <td>10</td>
      <td>0.141800</td>
    </tr>
    <tr>
      <td>11</td>
      <td>0.147900</td>
    </tr>
    <tr>
      <td>12</td>
      <td>0.134500</td>
    </tr>
    <tr>
      <td>13</td>
      <td>0.140000</td>
    </tr>
    <tr>
      <td>14</td>
      <td>0.151100</td>
    </tr>
    <tr>
      <td>15</td>
      <td>0.126900</td>
    </tr>
    <tr>
      <td>16</td>
      <td>0.094000</td>
    </tr>
    <tr>
      <td>17</td>
      <td>0.095500</td>
    </tr>
    <tr>
      <td>18</td>
      <td>0.104200</td>
    </tr>
    <tr>
      <td>19</td>
      <td>0.102800</td>
    </tr>
    <tr>
      <td>20</td>
      <td>0.086100</td>
    </tr>
    <tr>
      <td>21</td>
      <td>0.112100</td>
    </tr>
    <tr>
      <td>22</td>
      <td>0.094000</td>
    </tr>
    <tr>
      <td>23</td>
      <td>0.093500</td>
    </tr>
    <tr>
      <td>24</td>
      <td>0.097800</td>
    </tr>
    <tr>
      <td>25</td>
      <td>0.092600</td>
    </tr>
    <tr>
      <td>26</td>
      <td>0.076500</td>
    </tr>
    <tr>
      <td>27</td>
      <td>0.080600</td>
    </tr>
    <tr>
      <td>28</td>
      <td>0.083600</td>
    </tr>
    <tr>
      <td>29</td>
      <td>0.084700</td>
    </tr>
    <tr>
      <td>30</td>
      <td>0.079200</td>
    </tr>
    <tr>
      <td>31</td>
      <td>0.066500</td>
    </tr>
    <tr>
      <td>32</td>
      <td>0.059300</td>
    </tr>
    <tr>
      <td>33</td>
      <td>0.042300</td>
    </tr>
    <tr>
      <td>34</td>
      <td>0.043800</td>
    </tr>
    <tr>
      <td>35</td>
      <td>0.049000</td>
    </tr>
    <tr>
      <td>36</td>
      <td>0.052200</td>
    </tr>
    <tr>
      <td>37</td>
      <td>0.060200</td>
    </tr>
    <tr>
      <td>38</td>
      <td>0.052600</td>
    </tr>
    <tr>
      <td>39</td>
      <td>0.073600</td>
    </tr>
    <tr>
      <td>40</td>
      <td>0.048400</td>
    </tr>
    <tr>
      <td>41</td>
      <td>0.050500</td>
    </tr>
    <tr>
      <td>42</td>
      <td>0.049900</td>
    </tr>
    <tr>
      <td>43</td>
      <td>0.052000</td>
    </tr>
    <tr>
      <td>44</td>
      <td>0.057300</td>
    </tr>
    <tr>
      <td>45</td>
      <td>0.051800</td>
    </tr>
  </tbody>
</table><p>


### Training the Model with ROCm-Optimized Settings

This cell configures and executes the fine-tuning process:

**Training Configuration (SFTConfig):**
- **Batch size**: 64 per device - leveraging the AMD MI300X's massive 192GB HBM3 memory
- **Gradient accumulation**: 1 step
- **Warmup**: 5 steps
- **Epochs**: 1 full pass through the dataset
- **Learning rate**: 1e-4
- **Optimizer**: adamw_8bit for memory efficiency
- **Precision**: bf16 (bfloat16) for ROCm
- **Gradient checkpointing**: Enabled for memory efficiency

**Special Training Mode:**
Uses `train_on_responses_only` to compute loss only on the assistant's responses, not on the user's questions. This focuses the model on learning to generate accurate answers rather than memorizing the input format.

**Key Features:**
- DataCollatorForSeq2Seq handles variable-length sequences with proper padding
- No packing to preserve conversation structure
- Single dataloader worker for ROCm stability
- Gradient checkpointing via Unsloth for memory optimization

The model is then trained on the 74 logical reasoning conversations.


```python
"""Save the trained model"""
print("\n💾 SAVING ROCM-TRAINED MODEL")

# Save LoRA adapters
lora_path = "logical_reasoning_rocm_lora"
model.save_pretrained(lora_path)
tokenizer.save_pretrained(lora_path)
print(f"✅ LoRA adapters saved to: {lora_path}")

# Save merged model
merged_path = "logical_reasoning_rocm_merged"
print("🔄 Saving merged model...")
model.save_pretrained_merged(merged_path, tokenizer, save_method="merged_16bit")
print(f"✅ Merged model saved to: {merged_path}")

print(f"\n🎉 ROCM MODEL READY!")
```

    
    💾 SAVING ROCM-TRAINED MODEL
    ✅ LoRA adapters saved to: logical_reasoning_rocm_lora
    🔄 Saving merged model...
    Found HuggingFace hub cache directory: /root/.cache/huggingface/hub
    Checking cache directory for required files...


    Unsloth: Copying 30 files from cache to `logical_reasoning_rocm_merged`: 100%|██████████| 30/30 [02:13<00:00,  4.43s/it]


    Successfully copied all 30 files from cache to `logical_reasoning_rocm_merged`
    Checking cache directory for required files...
    Cache check failed: tokenizer.model not found in local cache.
    Not all required files found in cache. Will proceed with downloading.


    Unsloth: Preparing safetensor model files: 100%|████████████████████████████████████| 30/30 [00:00<00:00, 564256.14it/s]
    Unsloth: Merging weights into 16bit: 100%|██████████████████████████████████████████████| 30/30 [04:52<00:00,  9.74s/it]


    Unsloth: Merge process complete. Saved to `/workspace/AIAC/logical_reasoning/logical_reasoning_rocm_merged`
    ✅ Merged model saved to: logical_reasoning_rocm_merged
    
    🎉 ROCM MODEL READY!


### Saving the Fine-Tuned Model

This cell saves the trained model in two formats:

1. **LoRA Adapters** (`logical_reasoning_rocm_lora/`):
   - Saves only the trained LoRA adapter weights (lightweight, ~few hundred MB)
   - Can be loaded later with the base model
   - Useful for sharing or deploying with the original base model

2. **Merged Model** (`logical_reasoning_rocm_merged/`):
   - Merges LoRA adapters back into the base model
   - Creates a standalone model with all weights
   - Saved in 16-bit precision for better quality
   - Ready for immediate inference without loading adapters

Both formats include the tokenizer configuration. The merged model is production-ready and can be used directly for generating answers to logical reasoning questions.


```python
#fin
```


```python

```
