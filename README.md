# The Secret Sharer: Evaluating and Testing Unintended Memorization in Neural Networks

University project about "The Secret Sharer: Evaluating and Testing
Unintended Memorization in Neural Networks" Paper. It focuses on Log Perplexities of canaries and the ranks of them and that is what is included the repo. Based on this you can calculate exposure what is not included in this repo. Everything explained in https://arxiv.org/pdf/1802.08232. You can add interpolation and extrapolation approximation of exposure like explained in the paper. They are the inventor of "Exposure"-Metric. In the CPU version the canary is inserted 10x in the training dataset. 

Should work fine for cuda GPU and x86 CPU but Macs MPS is still in progress. Edit 03.06.25: Deleted self.device = torch.device("mps") because Apple CPU works out of the box with same code as x86 CPU and no code change needed.

CPU Version should be fine for most reasons because the model is small. Requirements.txt worked for CPU well.

My setup should work fine: Python3.11.10 or higher on Linux. Unfortunately i do not have Windows tests.

All credits go to the authors of https://arxiv.org/pdf/1802.08232
