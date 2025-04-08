## This is the official repository of the paper [Know Me, Respond to Me: Benchmarking LLMs for Dynamic User Profiling and Personalized Responses at Scale](TODO)

<p align="center">
<img src=figures/advanced_artistic_illustration.png/>
</p>

We present <img src="figures/logo.png" alt="Logo" width="24"/> **PersonaMem**, a new personalization benchmark to assess how well language models can infer evolving user profiles and generate personalized responses across task scenarios. PersonaMem emphasizes **persona-oriented**, **multi-session** interactions between users and chatbots, facilitated by a synthetic dialog generation pipeline that simulates realistic and evolving conversational contexts.

<p align="center">
<img src=figures/benchmark_overview.png/>
</p>

As shown in the overview, each benchmark sample is a user persona with static (e.g., demographic info.) and dynamic attributes (e.g., evolving preferences). Users engage with a chatbot in multi-session interactions across a variety of topics such as food recommendation, travel planning, and therapy consultation. As the user’s preferences evolve over time, the benchmark offers annotated questions assessing whether models can track and incorporate the changes into their responses.

## 📊 Benchmark Data
We provide the benchmark data of <img src="figures/logo.png" alt="Logo" width="24"/> **PersonaMem** on [Google Drive](https://drive.google.com/drive/folders/1bUyh-JWB-U70iEvE70ZaXzRBw5KPWODO?usp=sharing), including question-answer pairs and their corresponding contexts. The dataset is available with three versions based on context token length:

- **32k tokens**
  - ```questions_32k.csv```
  - ```shared_contexts_32k.jsonl```
- **128k tokens**
  - ```questions_128k.csv```
  - ```shared_contexts_128k.jsonl```
- **1M tokens**
  - ```questions_1M.csv```
  - ```shared_contexts_1M.jsonl```


## 🔗 Dependencies
We use Python virtual environment. Please run the following commands to create a virtual environment and install all the requirements:
    
    python -m venv myenv
    source myenv/bin/activate
    pip install -r requirements.txt


## 🚀 Running Inference on Benchmark Data

**Before you begin**, create a new folder named [api_tokens/](api_tokens/) in the root directory. This folder will store your API keys required to run the models.

1. **Create API keys** from the respective providers if you haven't already.

2. Inside the [api_tokens/](api_tokens/) folder, create the following text files depending on which models you plan to use. Paste your API key as plain text into the corresponding file:

    - ```openai_key.txt``` – for OpenAI models
    - ```gemini_key.txt``` – for Google Gemini models
    - ```claude_key.txt``` – for Anthropic Claude models
    - ```lambda_key.txt``` – for models accessed via the [Lambda Cloud API](https://docs.lambda.ai/public-cloud/lambda-inference-api/?_gl=1*1yqhedk*_gcl_aw*R0NMLjE3NDQwOTAyNjIuQ2owS0NRanc3ODJfQmhEakFSSXNBQlR2X0pEWUpQRTRhLXJMY0xjeWZYYUZrRzE4Q196MG0zdjY0cmQtX09FYm5iRHlrek95QWVDVVZxVWFBbnhYRUFMd193Y0I.*_gcl_au*NTQ3OTExNDIzLjE3NDQwOTAyNjE.*_ga*MTA0MjYwNjUyMS4xNzQ0MDkwMjYy*_ga_43EZT1FM6Q*MTc0NDA5MDI2MS4xLjAuMTc0NDA5MDI2MS42MC4wLjY1NjAyNzc2NA..) (e.g., Llama, DeepSeek, etc.)


We provide ready-to-use **inference scripts** in the [scripts/](scripts/) directory for evaluating the following models:
- **[OpenAI Models](https://platform.openai.com/docs/models)**
  - GPT-4.5: ```inference_gpt_4p5_preview.sh```
  - o3-mini: ```inference_o3_mini.sh```
  - o1: ```inference_o1.sh```
  - o1-mini: ```inference_o1_mini.sh```
  - GPT-4o: ```inference_gpt_4o.sh```
  - GPT-4o-mini: ```inference_gpt_4o_mini.sh```
- **[Google Gemini Models](https://ai.google.dev/gemini-api/docs/models)**
  - Gemini-2.0-Flash: ```inference_gemini_2p0_flash.sh```
  - Gemini-2.0-Flash-Lite: ```inference_gemini_2p0_flash_lite.sh```
  - Gemini-1.5-Flash: ```inference_gemini_1p5_flash.sh```
- **[Anthropic Claude Models](https://docs.anthropic.com/en/docs/about-claude/models/all-models)**
  - Claude-3.7-Sonnet: ```inference_claude_3p7_sonnet.sh```
  - Claude-3.5-Haiku: ```inference_claude_3p5_haiku.sh```
- **[Meta Llama Models](https://docs.lambda.ai/public-cloud/lambda-inference-api/?_gl=1*1yqhedk*_gcl_aw*R0NMLjE3NDQwOTAyNjIuQ2owS0NRanc3ODJfQmhEakFSSXNBQlR2X0pEWUpQRTRhLXJMY0xjeWZYYUZrRzE4Q196MG0zdjY0cmQtX09FYm5iRHlrek95QWVDVVZxVWFBbnhYRUFMd193Y0I.*_gcl_au*NTQ3OTExNDIzLjE3NDQwOTAyNjE.*_ga*MTA0MjYwNjUyMS4xNzQ0MDkwMjYy*_ga_43EZT1FM6Q*MTc0NDA5MDI2MS4xLjAuMTc0NDA5MDI2MS42MC4wLjY1NjAyNzc2NA..)**
  - Llama-4-Maverick: ```inference_llama4_maverick.sh```
  - Llama-3.1-405B: ```inference_llama_3p1_405b.sh```
- **[DeepSeek Models](https://docs.lambda.ai/public-cloud/lambda-inference-api/?_gl=1*1yqhedk*_gcl_aw*R0NMLjE3NDQwOTAyNjIuQ2owS0NRanc3ODJfQmhEakFSSXNBQlR2X0pEWUpQRTRhLXJMY0xjeWZYYUZrRzE4Q196MG0zdjY0cmQtX09FYm5iRHlrek95QWVDVVZxVWFBbnhYRUFMd193Y0I.*_gcl_au*NTQ3OTExNDIzLjE3NDQwOTAyNjE.*_ga*MTA0MjYwNjUyMS4xNzQ0MDkwMjYy*_ga_43EZT1FM6Q*MTc0NDA5MDI2MS4xLjAuMTc0NDA5MDI2MS42MC4wLjY1NjAyNzc2NA..)**
  - DeepSeek-R1-607B: ```inference_deepseek_r1_671b.sh```

To run evaluation for a specific model, simply execute the corresponding script. For example:
  
    bash scripts/inference_gpt_4o.sh

Each script supports benchmarking at different **context window sizes**. If the model allows, you can modify the ```BENCHMARK_SIZE``` variable inside the script to ```32k```, ```128k```, or ```1M```. Currently, only Gemini models and Llama-4 support context windows up to **1 million tokens**.

**Evaluation results** will be automatically saved to the [data/results/](data/results/) directory.

If you would like to add support for **additional models**, refer to our implementation in [`inference.py`](inference.py) or [`inference_standalone_openai.py`](inference_standalone_openai.py) for guidance. You only need to update the `__init__` and `query_llm` methods of the `Evaluation` class.



## 💬 Building Persona-Oriented Multi-Session Conversation Data
Interested in how we built the conversation data? Keep reading!
<p align="center">
<img src=figures/generate_data.png/>
</p>



## To start the persona-aligned conversation generation

We allow command-line argparser for the following arguments: 

therapy, legal, datingConsultation, foodRecommendation, onlineShopping, studyConsultation, travelPlanning, writing

- ```--model``` **[str]** to select the LLM to generate the data
  - ```gpt-4o```
- ```--n_persona``` **[int]** to select the number of unique personas. *(This is the outer loop)*
- ```--context``` **[str]** to select the context of the conversation. To select a single context, use the format ```context1```. To select multiple contexts, use the format ```context1 context2 context3```. *(This is the middle loop)*
  - ```therapy```
  - ```legal```
  - ```datingConsultation```
  - ```foodRecommendation```
  - ```onlineShopping```
  - ```studyConsultation```
  - ```travelPlanning```
  - ```writing```
  - ```all```  to select all existing contexts under [./data/output/](./data/output/). Feel free to create a new empty folder with the new context name you want. Note that currently we have real-world seeding data for ```therapy```, ```legal```, and ```writing``` contexts only.
- ```--n_samples``` **[int]** to select the number of samples per context per persona. *(This is the inner loop)*
- ```--verbose``` **[store_true]** to print out all generated contents.

#### To generate conversations of a single context

    python prepare_data.py --model gpt-4o --context therapy --n_persona 10 --n_samples 1 --verbose

#### To generate conversations of multiple contexts, specify the names of the contexts and separate them by space, e.g.

    python prepare_data.py --model gpt-4o --context therapy travel food --n_persona 10 --n_samples 1 --verbose

#### To generate conversations of all contexts available

    python prepare_data.py --model gpt-4o --context all --n_persona 10 --n_samples 1 --verbose

The most common reason for generation failures is syntax errors in JSON formats. The LLM might occasionally produce strings that do not conform to the required JSON format. In such cases, we will output the data paths of all failed samples. You can then copy these paths into [./scripts/rerun_prepare_data.sh](./scripts/rerun_prepare_data.sh) and run the following command to process the failed samples again.
    
    bash scripts/rerun_prepare_data.sh

## To start the Q&A generation

The Q&As must be generated after the conversations. We allow command-line argparser for the following arguments:

- ```--model``` **[str]** to select the LLM to generate the q&a
  - ```gpt-4o```
- ```--action``` **[str]** to select the current action
    - ```view_graphs``` to display all linear graphs of knowledge updates up to the specified cut-off time (included). Not applicable for the ```writing``` context.
    - ```qa``` to generate question and answer pairs
- ```--data``` **[str]** to specify the data path of the conversation data. Note that the data path specifies the current context.
- ```--time``` **[str]** to specify the cut-off time (included) for the conversation data. Not applicable for the ```writing``` context.
    - ```init``` for the ```Initial Conversation``` block
    - ```next_week``` for the ```Conversation Next Week``` block
    - ```next_month``` for the ```Conversation Next Month``` block
    - ```next_year``` for the ```Conversation Next Year``` block
- ```--verbose``` **[store_true]** to print out all generated contents.

#### To visualize linear graphs of knowledge updates

    python prepare_qa.py --model gpt-4o --action view_graphs --data therapy_persona0_sample0 --time next_year --verbose

#### To generate Q&As for one given data file

    python prepare_qa.py --model gpt-4o --action qa --data therapy_persona0_sample0 --time next_year --verbose

#### To generate Q&As for a batch of data files

Specify the data files you want to process in [./scripts/run_all_prepare_qa.sh](./scripts/run_all_prepare_qa.sh) and run the following command.

    bash scripts/run_all_prepare_qa.sh

Similarly, the most common reason for generation failures is syntax errors in JSON formats, and we will output the data paths of all failed samples. You can copy these paths into [./scripts/rerun_prepare_qa.sh](./scripts/rerun_prepare_qa.sh) and run the following command to process the failed samples again.
    
    bash scripts/rerun_prepare_qa.sh

## To test block concatenations for enabling long context windows

This step is optional and intended for debugging purposes only. The actual block concatenation process will be performed in the next session.

The block concatenation must be performed after the Q&As are generated. We allow command-line argparser for the following arguments:

- ```--idx_persona``` **[int]** to select the index of the persona.
- ```--n_blocks``` **[int]** to select the number of conversation blocks to concatenate. We will randomly sample n_blocks from available data belonging to idx_persona.
- ```--format``` **[str]** to select the output conversation format.
  - ```string``` to select pure the string format for the concatenated conversations.
  - ```api_dict``` to select the API dictionary format for the concatenated conversations, such as 'user' and 'assistant'.
- ```--verbose``` **[store_true]** to print out all generated contents.

#### Example command

    python prepare_block.py --idx_persona 0 --n_blocks 5 --format string --verbose


## To evaluate LLMs on the generated data

This is the final step of the pipeline. You must have set up your API tokens under [api_tokens](api_tokens). We allow command-line argparser for the following arguments:

- ```--model``` **[str]** to select the LLM to evaluate (we currently support gpt-4o and gpt-4o-mini)
  - ```o1-preview```
  - ```o1-mini```
  - ```gpt-4o```
  - ```gpt-4o-mini```
  - ```gpt-4-turbo```
  - ```gpt-3.5-turbo```
  - ```gemini-1.5-flash-002```
  - ```gemini-1.5-pro-002```
  - ```gemini-1.0-pro```
  - ```meta-llama-3-70b-instruct```
  - ```meta-llama-3-8b-instruct```
  - ```claude-3-opus-20241022```
  - ```claude-3-5-sonnet-20241022```
- ```--idx_persona``` **[int]** to select the index of the persona.
- ```--format``` **[str]** to select the output conversation format.
  - ```string``` to select pure the string format for the concatenated conversations.
  - ```api_dict``` to select the API dictionary format for the concatenated conversations, such as ```user``` and ```assistant```.
- ```--n_blocks``` **[int]** to select the number of conversation blocks to concatenate. We will randomly sample n_blocks from available data belonging to idx_persona.
- ```--up_to``` **[store_true]** to evaluate on all the way from 1 up to n_blocks, not just n_blocks itself.
- ```--verbose``` **[store_true]** to print out all generated contents.

#### Example command

    python inference.py --model o1-preview --idx_persona 0 --format api_dict --n_blocks 5 --up_to --verbose

#### To run evaluations on multiple scenarios

Specify the evaluation hyperparameters in [./scripts/run_all_inference.sh](./scripts/run_all_inference.sh) and run the following command.

    bash scripts/run_all_inference.sh
