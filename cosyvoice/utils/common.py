import random
import numpy as np
import torch

IGNORE_ID = -1

instruct_list = ["You are a helpful assistant. 请用广东话表达。<|endofprompt|>",
                 "You are a helpful assistant. 请用东北话表达。<|endofprompt|>",
                 "You are a helpful assistant. 请用甘肃话表达。<|endofprompt|>",
                 "You are a helpful assistant. 请用贵州话表达。<|endofprompt|>",
                 "You are a helpful assistant. 请用河南话表达。<|endofprompt|>",
                 "You are a helpful assistant. 请用湖北话表达。<|endofprompt|>",
                 "You are a helpful assistant. 请用湖南话表达。<|endofprompt|>",
                 "You are a helpful assistant. 请用江西话表达。<|endofprompt|>",
                 "You are a helpful assistant. 请用闽南话表达。<|endofprompt|>",
                 "You are a helpful assistant. 请用宁夏话表达。<|endofprompt|>",
                 "You are a helpful assistant. 请用山西话表达。<|endofprompt|>",
                 "You are a helpful assistant. 请用陕西话表达。<|endofprompt|>",
                 "You are a helpful assistant. 请用山东话表达。<|endofprompt|>",
                 "You are a helpful assistant. 请用上海话表达。<|endofprompt|>",
                 "You are a helpful assistant. 请用四川话表达。<|endofprompt|>",
                 "You are a helpful assistant. 请用天津话表达。<|endofprompt|>",
                 "You are a helpful assistant. 请用云南话表达。<|endofprompt|>",
                 "You are a helpful assistant. Please say a sentence as loudly as possible.<|endofprompt|>",
                 "You are a helpful assistant. Please say a sentence in a very soft voice.<|endofprompt|>",
                 "You are a helpful assistant. 请用尽可能慢地语速说一句话。<|endofprompt|>",
                 "You are a helpful assistant. 请用尽可能快地语速说一句话。<|endofprompt|>",
                 "You are a helpful assistant. 请非常开心地说一句话。<|endofprompt|>",
                 "You are a helpful assistant. 请非常伤心地说一句话。<|endofprompt|>",
                 "You are a helpful assistant. 请非常生气地说一句话。<|endofprompt|>",
                 "You are a helpful assistant. 我想体验一下小猪佩奇风格，可以吗？<|endofprompt|>",
                 "You are a helpful assistant. 你可以尝试用机器人的方式解答吗？<|endofprompt|>"]


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


# Repetition Aware Sampling in VALL-E 2
def ras_sampling(weighted_scores, decoded_tokens, sampling, top_p=0.8, top_k=25, win_size=10, tau_r=0.1):
    top_ids = nucleus_sampling(weighted_scores, top_p=top_p, top_k=top_k)
    rep_num = (torch.tensor(decoded_tokens[-win_size:]).to(weighted_scores.device) == top_ids).sum().item()
    if rep_num >= win_size * tau_r:
        weighted_scores[top_ids] = -float('inf')
        top_ids = random_sampling(weighted_scores, decoded_tokens, sampling)
    return top_ids


def nucleus_sampling(weighted_scores, top_p=0.8, top_k=25):
    prob, indices = [], []
    cum_prob = 0.0
    sorted_value, sorted_idx = weighted_scores.softmax(dim=0).sort(descending=True, stable=True)
    for i in range(len(sorted_idx)):
        # sampling both top-p and numbers.
        if cum_prob < top_p and len(prob) < top_k:
            cum_prob += sorted_value[i]
            prob.append(sorted_value[i])
            indices.append(sorted_idx[i])
        else:
            break
    prob = torch.tensor(prob).to(weighted_scores)
    indices = torch.tensor(indices, dtype=torch.long).to(weighted_scores.device)
    top_ids = indices[prob.multinomial(1, replacement=True)].item()
    return top_ids


def random_sampling(weighted_scores, decoded_tokens, sampling):
    top_ids = weighted_scores.softmax(dim=0).multinomial(1, replacement=True).item()
    return top_ids


def set_all_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)