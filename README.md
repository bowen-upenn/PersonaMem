## This is the official implementation of the paper [MemoryBench: TODO](todo) in PyTorch.

### To start the conversation data generation

We allow command-line argparser for the following arguments: 
    
- ```--model``` to select the LLM.
  - ```gpt-4o```
- ```--n_persona``` to select the number of unique personas. *(This is the outer loop)*
- ```--context``` to select the context of the conversation. To select a single context, use the format ```context1```. To select multiple contexts, use the format ```context1 context2 context3```. *(This is the middle loop)*
  - ```therapy```
- ```--n_samples``` to select the number of samples per context per persona. *(This is the inner loop)*
- ```--verbose``` to print out all generated contents.

#### To generate conversations of a single context

    python main.py --model gpt-4o --context therapy --n_persona 3 --n_samples 5 --verbose

#### To generate conversations of multiple contexts, specify the names of the contexts and separate them by space, e.g.

    python main.py --model gpt-4o --context therapy travel food --n_persona 3 --n_samples 5 --verbose


### To start the Q&A data generation

The Q&As must be generated after the conversations. We allow command-line argparser for the following arguments: 
    
- ```--model``` to select the LLM.
  - ```gpt-4o```
- ```--action``` to select the current action
    - ```view_graphs``` to display all linear graphs of knowledge updates up to the specified cut-off time (included)
    - ```qa``` to generate question and answer pairs
- ```--time``` to specify the cut-off time (included) for the conversation data.
    - ```init```
    - ```next_week```
    - ```next_month```
    - ```next_year```
- ```--data``` to specify the data path of the conversation data.
- ```--verbose``` to print out all generated contents.

#### To visualize linear graphs of knowledge updates

    python prepare_qa.py --model gpt-4o --action view_graphs --data therapy_persona0_sample0 --time next_year --verbose

#### To generate Q&As

    python prepare_qa.py --model gpt-4o --action qa --data therapy_persona0_sample0 --time next_year --verbose
