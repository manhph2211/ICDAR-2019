import torch
from torch.nn import functional as F
import torch.nn as nn



def encode_batch(batch_text):
  encode_batch=[]
  len_batch=[]
  for text in batch_text:
    encode_batch.append(text)
    len_batch.append(len(text))
  return torch.cat(encode_batch),torch.IntTensor(len_batch)


def ctc_loss(text_batch, text_batch_logits,device='cuda'):
    text_batch_logps = F.log_softmax(text_batch_logits, 2) 
    text_batch_logps_lens = torch.full(size=(text_batch_logps.size(1),), 
                                       fill_value=text_batch_logps.size(0), 
                                       dtype=torch.int32).to(device) 
   
    text_batch,text_batch_targets_lens = encode_batch(text_batch)
    criterion = nn.CTCLoss().to(device)
    loss = criterion(text_batch_logps, text_batch, text_batch_logps_lens, text_batch_targets_lens)

    return loss