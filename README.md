# This is the official implementation of the paper [MemoryBench: TODO](todo) in PyTorch.

<p align="center">
<img src=figures/artistic_illustration.jpeg/>
</p>

We create a persona-oriented memory benchmark for LLMs and their agents. We aim to evaluate not only if the LLMs can **remember the user's persona from long user-AI conversations**, but also **utilize this memory to generate persona-aligned recommendations and new contents** over diverse scenarios. 

## To start the persona-aligned conversation data generation

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


## To start the Q&A data generation

The Q&As must be generated after the conversations. We allow command-line argparser for the following arguments:

- ```--model``` **[str]** to select the LLM to generate the q&a
  - ```gpt-4o```
- ```--action``` **[str]** to select the current action
    - ```view_graphs``` to display all linear graphs of knowledge updates up to the specified cut-off time (included). Not applicable for ```writing``` context.
    - ```qa``` to generate question and answer pairs
    - ```batch_qa``` to generate question and answer pairs for all available data under [./data/output/](./data/output/) over all time periods.
- ```--data``` **[str]** to specify the data path of the conversation data. Not applicable for ```batch_qa``` action. Note that the data path also specifies the current context.
- ```--time``` **[str]** to specify the cut-off time (included) for the conversation data. Not applicable for ```batch_qa``` action or ```writing``` context.
    - ```init``` for the ```Initial Conversation``` block
    - ```next_week``` for the ```Conversation Next Week``` block
    - ```next_month``` for the ```Conversation Next Month``` block
    - ```next_year``` for the ```Conversation Next Year``` block
- ```--verbose``` **[store_true]** to print out all generated contents.

#### To visualize linear graphs of knowledge updates

    python prepare_qa.py --model gpt-4o --action view_graphs --data therapy_persona0_sample0 --time next_year --verbose

#### To generate Q&As for one given data file

    python prepare_qa.py --model gpt-4o --action qa --data therapy_persona0_sample0 --time next_year --verbose

#### To generate Q&As for all available data files

    python prepare_qa.py --model gpt-4o --action batch_qa --verbose


## To test the block concatenation

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

- ```--model``` **[str]** to select the LLM to evaluate
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
