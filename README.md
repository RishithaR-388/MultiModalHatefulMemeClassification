# Prompt-Based Hateful Meme Detection - `HateDetector`


Hateful meme classification is a challenging multimodal task that requires complex reasoning and contextual background knowledge. 
Ideally, we could leverage an explicit external knowledge base to supplement contextual and cultural information in hateful memes. 
However, there is no known explicit external knowledge base that could provide such hate speech contextual information. 
To address this gap, we propose PromptHate, a simple yet effective prompt-based model that prompts pre-trained language models (PLMs) for hateful meme
classification. Specifically, we construct simple prompts and provide a few in-context examples to exploit the implicit knowledge in the pre-
trained RoBERTa language model for hatefulmeme classification.


## Run HateDetector
Our code is built on [transformers](https://github.com/huggingface/transformers) and we use its version `4.35.0` version for pre-trained models such as Roberta and SqueezeBert. We also used PyTorch on an NVIDIA Tesla T4 GPU with 15 GB dedicated memory and CUDA 12.0 to train all the models. It takes up 15 GB dedicated memory for PromptHate training on FHM Dataset. 

To run use 
python main.py --SAVE_NUM 821 --SEED 1111 --LR_RATE 1.3e-5 and change it's arguments accordingly.




