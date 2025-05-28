import torch

def sample(gen_sample,rewards,n_samples,temperature,method="greedy"):
    
        
    if method=="multinomial":
        # probs = torch.softmax(temperature*rewards[:]/rewards.sum(),dim=0).reshape(-1,)
        probs = torch.softmax(temperature*rewards[:],dim=0).reshape(-1,)
                    
        select_ind = torch.multinomial(probs,n_samples,replacement=True) 
        # print(f"For the sampling the selected indexes are {select_ind}")
        # print(rewards)
        # print(probs)
        # print(select_ind)
        # print(select_ind)
        # print(gen_sample.shape)
        curr_samples = gen_sample[0, select_ind]
        return curr_samples
        # print(curr_samples.shape)
        
    else:
        select_ind = torch.max(rewards, dim=0)[1]
        curr_samples = torch.cat([x[select_ind[idx]].unsqueeze(0) for idx, x in enumerate(gen_sample)], dim=0) # TODO: Make it efficient
        return curr_samples.repeat(n_samples, 1, 1, 1)