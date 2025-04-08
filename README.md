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

Google Gemini models have conflicting dependencies with OpenAI models related to `google-genai` and `httpx` packages. To run Gemini models, we therefore recommend creating a separate Conda environment:

    conda create -n persona_mem python=3.9
    conda activate persona_mem
    pip install -r requirements.txt
    pip install -q -U google-genai


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
  - Gemini-2.5-Pro: ```inference_gemini_2p5_pro.sh```
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
*Interested in how we built the conversation data? Keep reading!*
<p align="center">
<img src=figures/generate_data.png/>
</p>

### Step 1 - Generating User Personas and Conversations
**We provide a script to automatically generate persona-based multi-session conversations. To run it:**

```bash
bash scripts/run_all_prepare_data.sh
```
💡 **Tip:** If a data generation step fails, it's likely due to syntax issues in the LLM-generated response. Simply regenerate the data of that file.

> We also allow command-line argparser for the following arguments inside the script:
> - `--model` **[str]**: The LLM used for generation (e.g., `gpt-4o`).
> - `--topics` **[str]**: One or more conversation topics (space-separated for multiple).
> - `--n_persona` **[int]**: Total number of different personas to generate, specified by the `end_persona_id` variable in the script.
> - `--s_persona` **[int]**: The starting index of all personas to generate, specified by the `start_persona_id` variable in the script.
> - `--output_dir` **[str]**: Directory where generated data will be saved.
> - `--clean` **[store_true]** Remove existing data files and start clean.
> - `--verbose` **[store_true]**: Print all generated content to the console.
> 
> You only need to specify integer values for `end_persona_id` and `start_persona_id`. A total of `end_persona_id - start_persona_id` random personas will be created automatically. Data of different topics under the same `persona_id` will always share the same persona.
> 
> Example: Generate Conversations for a Single Topic
> ```bash
> python prepare_data.py --model gpt-4o --context therapy --output_dir data/output/ --verbose
> ```
> Example: Generate Conversations for Multiple Topics
> ```bash
> python prepare_data.py --model gpt-4o --topics therapy travelPlanning foodRecommendation --output_dir data/output/ --verbose
> ```
> 
> We currently include 18 diverse conversation topics: - `bookRecommendation`, `coding`, `datingConsultation`, `email`, `familyRelations`, `financialConsultation`, `foodRecommendation`, `homeDecoration`, `legalConsultation`, `medicalConsultation`, `movieRecommendation`, `musicRecommendation`, `onlineShopping`, `sportsRecommendation`, `studyConsultation`, `therapy`, `travelPlanning`, `writing`. Feel free to experiment by specifying a new topic name in the command line.


### Step 2 - Generating question-answering pairs
**We provide a script to continue to generate question-answering pairs. To run it:**

```bash
bash scripts/run_all_prepare_qa.sh
```

> We also allow command-line argparser for the following arguments inside the script:
> - `--model` **[str]**: The LLM used for generation (e.g., `gpt-4o`).
> - `--action` **[str]**: Default `qa` to generate question-answering pairs. `view_graphs` to visualize the event sequence of a persona.
> - `--topics` **[str]**: One or more conversation topics (space-separated for multiple).
> - `--n_persona` **[int]**: Total number of different personas to generate, specified by `end_persona_id` in the script.
> - `--s_persona` **[int]**: The starting index of all personas to generate, specified by `start_persona_id` in the script.
> - `--time` **[str]**: A list of time periods selected from `init`, `next_week`, `next_month`, and `next_year`, specified by the `time_periods` variable in the script.
> - `--clean` **[store_true]** Remove existing data files and start clean.
> - `--verbose` **[store_true]**: Print all generated content to the console.
> 
> Example: Generate Question-Answering Pairs for Multiple Topics
> ```bash
> python prepare_data.py --model gpt-4o --action qa --topics therapy travelPlanning foodRecommendation --time init --verbose
> ```

### Step 3 - Contructing long contexts in benchmark
*🧩 Now we have conversations and Q&A pairs for each conversation session. Let’s concatenate them to form the full interaction history.*

**We provide a script to continue to generate question-answering pairs. To run it, for example:**

```bash
bash scripts/run_generate_benchmark.sh large
```

The context length is determined by the argument you pass to the script:

- `small`  → up to **32k tokens**
- `medium` → up to **128k tokens**
- `large`  → up to **1M tokens**

> We also allow command-line argparser for the following arguments inside the script:
> - `--model` **[str]**: The LLM used for filtering low-quality questions (e.g., `gpt-4o-mini`).
> - `--step` **[str]**: Default `prepare` to generate benchmark contexts.
> - `--idx_persona` **[int]**: The index of the persona for which the context is constructed, specified by `start_persona_id` and `end_persona_id` in the script.
> - `--n_blocks` **[int]**: Total number of conversation sessions to concatenate. This is set automatically when using small, medium, or large.
> - `--n_variants` **[int]**: Number of different topological variants (orderings) of conversation sessions to concatenate.
> - `--filter_questions` **[store_true]**: Use an LLM to remove questions that can be answered directly without seeing context.
> - `--clean` **[store_true]** Remove existing data files and start clean.
> - `--verbose` **[store_true]**: Print all generated content to the console.
> 
> Example: Generate Full Context for One Persona
> ```bash
> python inference.py --step prepare --model gpt-4o-mini --idx_persona 0 --n_blocks 60 --n_variants 2 --filter_questions --clean --verbose
> ```
