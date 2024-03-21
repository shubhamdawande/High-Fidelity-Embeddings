import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel

### -------------------- load embedding model -------------------- ###

model_id = 'intfloat/e5-mistral-7b-instruct'
device = "cuda:0" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModel.from_pretrained(model_id)
tokenizer.add_eos_token = True

### -------------------- Utils -------------------- ###

def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'

def get_embedding(text):
    max_length = 4096
    batch_dict = tokenizer(text, max_length=max_length - 1, padding=True, truncation=True, return_tensors="pt")
    outputs = model(**batch_dict)
    text_emb = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
    text_emb = F.normalize(text_emb, p=2, dim=1)
    return text_emb

"""
Get Cosine similarity score between query and candidate text

Parameters:
- query (str)
- candidate (str)

Returns:
- score between 0-1
"""
def get_cosine_similarity(query, candidate):

    task_query = "Given is a query text." # task will vary according to domain
    query_instruct = get_detailed_instruct(task_query, query)
    query_emb = get_embedding(query_instruct)
    
    task_chunk = "Given is a text chunk from a pdf" # task will vary according to domain
    candidate_instruct = get_detailed_instruct(task_chunk, candidate)
    candidate_emb = get_embedding(candidate_instruct)
    
    # query_emb_tensor = torch.FloatTensor(query_emb)
    # candidate_emb_tensor = torch.FloatTensor(candidate_emb)

    candidate_emb = candidate_emb.T
    score = (query_emb @ candidate_emb) * 100

    return score

    
### Test
query = "What is the average salary of software engineer in europe?"
candidate = "The average person's salary in Europe ranged from €40,000 to €70,000 annually. However, this figure can be higher in cities with a higher cost of living or for specialized skills or extensive experience."
get_cosine_similarity(query, candidate)