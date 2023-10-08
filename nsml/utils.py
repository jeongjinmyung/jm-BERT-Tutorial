import os
import numpy as np
import torch

class Args():
    seed = 42
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epochs = 4
    batch_size = 4
    sentence_max_len = 300
    learning_rate = 2e-5
    eps = 1e-8
    eval_step = 40
    num_classes = 7
    model_name = "beomi/kcbert-base"
    tokenizer_name = "beomi/kcbert-base"
    dataset_dir = os.path.join(os.getcwd(), 'data')
    result_dir = os.path.join(os.getcwd(), "results")
    
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)