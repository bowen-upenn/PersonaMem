from vllm import LLM

# 1. Load the model once (outside the function).
#    Replace "YourModelNameHere" with an actual model name you have locally or in HF Hub.
model = LLM(model="meta-llama/Llama-2-7b-chat-hf")

def choose_best_completion(context: str, choices: list[str]) -> str:
    """
    Given a context and four choices, returns the choice with the highest
    joint probability p(context + choice).
    """
    best_choice = None
    best_logprob_sum = float("-inf")
    
    for choice in choices:
        # 2. Compute the log-probs for `choice` given the `context`.
        #    The .score() method returns a list (one entry per prompt),
        #    each entry is a ScoreOutput with:
        #       output.tokens   -> List of tokens in the continuation
        #       output.logprobs -> List of log-probs (float) for each token
        outputs = model.score(context, choice)  # returns List[ScoreOutput]
        print(outputs)
        
        # For a single prompt, outputs[0] holds the ScoreOutput.
        # Sumiy the log-probs for the entire continuation to get the joint log-prob.
        logprob_sum = sum(outputs[0].logprobs)
        print(logprob_sum)
        
        # Keep track of the choice with the highest sum of log-probs.
        if logprob_sum > best_logprob_sum:
            best_logprob_sum = logprob_sum
            best_choice = choice
    
    return best_choice

# Example usage:
if __name__ == "__main__":
    context_str = "The capital of France is"
    choice_list = [
        "Paris",
        "London",
        "Berlin",
        "Rome"
    ]
    
    best = choose_best_completion(context_str, choice_list)
    print("Best choice:", best)
