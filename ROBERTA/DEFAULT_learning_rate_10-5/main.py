import torch
import numpy as np
import random
import sys

import config
import os
from train import train_for_epoch
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

if __name__=='__main__':
    opt=config.parse_opt()
    torch.cuda.set_device(0)
    set_seed(opt.SEED)
    log_file_path='/content/prompthate/PromptHate-Code/Log.rtf'
    with open(log_file_path, 'w') as f:
        sys.stdout = f
    
        # Create tokenizer
        tokenizer = RobertaTokenizer.from_pretrained('roberta-large')

        constructor='build_baseline'
        if opt.MODEL=='pbm':
            from dataset import Multimodal_Data
            import baseline
            train_set=Multimodal_Data(opt,tokenizer,opt.DATASET,'train',opt.SEED-1111)
            test_set=Multimodal_Data(opt,tokenizer,opt.DATASET,'test')
            label_list=[train_set.label_mapping_id[i] for i in train_set.label_mapping_word.keys()]
            model=getattr(baseline,constructor)(opt, label_list).cuda()
        else:
            from roberta_dataset import Roberta_Data
            import roberta_baseline
            train_set=Roberta_Data(opt,tokenizer,opt.DATASET,'train',opt.SEED-1111)
            test_set=Roberta_Data(opt,tokenizer,opt.DATASET,'test')
            model=getattr(roberta_baseline,constructor)(opt).cuda()
            
        train_loader=DataLoader(train_set,
                                opt.BATCH_SIZE,
                                shuffle=True,
                                num_workers=1)
        test_loader=DataLoader(test_set,
                              opt.BATCH_SIZE,
                              shuffle=False,
                              num_workers=1)
        train_for_epoch(opt,model,train_loader,test_loader)
        torch.save(model.state_dict(), 'model_weights.pth')

    # Reset stdout to the default (console)
    sys.stdout = sys.__stdout__

    exit(0)
    