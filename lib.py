import re

OPENAI_MODELS = set([r'gpt', r'o1'])
MEM0_MODELS = OPENAI_MODELS | set([r'claude', r'gemini', r'mistral'])

def re_search_list(patterns, string):
    if isinstance(patterns, str):
        patterns = [patterns]
    for pattern in patterns:
        if re.search(pattern, string) is not None:
            return True
    return False

def get_provider(model):
    if re_search_list(OPENAI_MODELS, model):
        return 'openai'
    if re_search_list('mistral', model):
        return 'litellm'
    if re_search_list('claude', model):
        return 'anthropic'
    if re_search_list('gemini', model):
        return 'gemini'
    return None