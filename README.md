## This is the official implementation of the paper [MemoryBench: TODO](todo) in PyTorch.

## Quick Start
We allow command-line argparser for the following arguments: 
    
- ```--model``` to select the LLM.
  - ```gpt-4-turbo```, ```gpt-4o```
- ```--n_persona``` to select the number of unique personas. *(This is the outer loop)*
- ```--context``` to select the context of the conversation. To select a single context, use the format ```"context1"```. To select multiple contexts, use the format ```"[context1,context2,context3]"```. *(This is the middle loop)*
  - ```therapy```
- ```--n_samples``` to select the number of samples per context per persona. *(This is the inner loop)*
- ```--verbose``` to print out all generated contents.
      
**To start the data generation**

To generate conversations of a single context

    python main.py --model gpt-4o --context therapy --n_persona 3 --n_samples 5 --verbose

To generate conversations of multiple contexts

    python main.py --model gpt-4o --context [therapy,travel,food] --n_persona 3 --n_samples 5 --verbose
