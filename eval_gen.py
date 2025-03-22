from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import numpy as np

# Prepare candidates
prompt = "I like "
candidate = "ice cream"
full_sequence = prompt + candidate  # e.g., "I like ice cream"
#llm = LLM(model="meta-llama/Llama-3.2-1B", dtype="half")
#llm = LLM(model="Qwen/Qwen2.5-1.5B-Instruct", dtype="float16")
llm = LLM(model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", dtype="half")


# Generate and get logprobs
params = SamplingParams(prompt_logprobs=1, max_tokens=10)  # max_tokens=0: no generation
output = llm.generate([full_sequence], params)[0]
token_logprobs = output.prompt_logprobs  # list of log-prob info for each prompt token
# params = SamplingParams(temperature=0.0, top_p=1.0, logprobs=1, max_tokens=len(candidate_tokens))
# output = llm.generate([prompt], params)[0]
# gen_text = output.outputs[0].text
#token_logprobs = output.outputs[0].logprobs  # log-prob for each generated token
# total_logprob = sum(token_logprobs)
print("Total log probability:", token_logprobs)


# # 1. Load the model once (outside the function).
# #    Replace "YourModelNameHere" with an actual model name you have locally or in HF Hub.
# model = LLM(model="meta-llama/Llama-2-7b-chat-hf")

# def choose_best_completion(context: str, choices: list[str]) -> str:
#     """
#     Given a context and four choices, returns the choice with the highest
#     joint probability p(context + choice).
#     """
#     best_choice = None
#     best_logprob_sum = float("-inf")
    
#     for choice in choices:
#         # 2. Compute the log-probs for `choice` given the `context`.
#         #    The .score() method returns a list (one entry per prompt),
#         #    each entry is a ScoreOutput with:
#         #       output.tokens   -> List of tokens in the continuation
#         #       output.logprobs -> List of log-probs (float) for each token
#         outputs = model.score(context, choice)  # returns List[ScoreOutput]
#         print(outputs)
        
#         # For a single prompt, outputs[0] holds the ScoreOutput.
#         # Sumiy the log-probs for the entire continuation to get the joint log-prob.
#         logprob_sum = sum(outputs[0].logprobs)
#         print(logprob_sum)
        
#         # Keep track of the choice with the highest sum of log-probs.
#         if logprob_sum > best_logprob_sum:
#             best_logprob_sum = logprob_sum
#             best_choice = choice
    
#     return best_choice

# # Example usage:
# if __name__ == "__main__":
#     context_str = "The capital of France is"
#     choice_list = [
#         "Paris",
#         "London",
#         "Berlin",
#         "Rome"
#     ]
    
#     best = choose_best_completion(context_str, choice_list)
#     print("Best choice:", best)

# model_name = "meta-llama/Llama-3.2-1B"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# llm = LLM(model=model_name, dtype="half")

# def get_best_choice_llama3(context, query, choices):
#     """
#     Given a context, a query, and four choices, this function evaluates the joint log probability 
#     of (query + choice) conditioned on the context for the Llama-3 8B model, and returns the choice 
#     with the highest probability.

#     Args:
#         context (str): The context text.
#         query (str): The query text.
#         choices (list of str): A list of four candidate choices.
#         model_name (str): The Hugging Face identifier for the Llama-3 8B model.

#     Returns:
#         best_choice (str): The choice with the highest joint probability.
#         best_log_prob (float): The summed log probability for the (query + choice) portion.
#     """
#     # 1. Initialize the Llama-3 8B model.
    
    
#     # 2. Build one prompt per choice, concatenating context, query, and each choice.
#     prompts = [f"{context}{query}{choice}" for choice in choices]
    
#     # 3. Use forced decoding (temperature=0) and request full token log probabilities.
#     sampling_params = SamplingParams(n=1, temperature=0.0, logprobs=20)
#     outputs = llm.generate(prompts, sampling_params)
    
#     # Tokenize the context to know how many tokens to skip.
#     context_tokens = llm.tokenizer(context)["input_ids"]
#     context_token_length = len(context_tokens)
    
#     best_choice = None
#     best_log_prob = -np.inf

#     # 4. For each output, extract the log probabilities corresponding only to the (query + choice) part.
#     for prompt, output, choice in zip(prompts, outputs, choices):
#         # Tokenize the full prompt to know its length.
#         full_prompt_ids = llm.tokenizer(prompt)["input_ids"]
#         prompt_token_count = len(full_prompt_ids)
        
#         # The generation uses forced decoding, so the last 'prompt_token_count' tokens in the generated
#         # sequence should match our full prompt.
#         generated_token_ids = output.outputs[0].token_ids
#         generated_log_probs = output.outputs[0].logprobs
        
#         # Align the tokens: take the last 'prompt_token_count' log probs.
#         aligned_log_probs = generated_log_probs[-prompt_token_count:]
        
#         # We want the log probabilities for the (query + choice) part, i.e. tokens after the context.
#         relevant_log_probs = aligned_log_probs[context_token_length:]
#         total_log_prob = float(np.sum(relevant_log_probs))
        
#         # Keep track of the best (highest) log probability.
#         if total_log_prob > best_log_prob:
#             best_log_prob = total_log_prob
#             best_choice = choice

#     return best_choice, best_log_prob

# # --------------------
# # Example usage:
# if __name__ == "__main__":
#     context_text = "In chess, the king can only move one square in any direction. "
#     query_text = "Which piece moves diagonally? "
#     choice_candidates = [
#         "Queen", 
#         "Bishop", 
#         "Knight", 
#         "Rook"
#     ]
    
#     best_choice, best_lp = get_best_choice_llama3(context_text, query_text, choice_candidates)
#     print(f"Best choice: '{best_choice}' with log probability: {best_lp:.4f}")