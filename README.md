
Multimodal Machine Learning Project for Fall'23

## Univl-dr
To run the experiments, go to the folder `UniVL-DR`. You will need to: 
- create an environment with `python==3.8` and the packages in `requirements.txt`
- add all the relevant files in `\data` (see the README of the submodule). These files can be downloaded with the links from the original repo of UniVL-DR.
- download and add the checkpoint of the pretrained CLIP model (cf. the path indicated in the default args of the files).
- try to run `experiments.py`, the default args should guide you to the missing files to download and where to put them.

Main contributions: 
- custom contrastive loss functions to fine-tune on hard-negatives
- experiments with different combinations of losses
- custom embedding generation and custom retrieval experiments
- using the obtained embeddings to generate answers (other branch of the main repo, see Avi's work)