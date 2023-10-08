#!/usr/bin/env python

import numpy as np
import pandas as pd
import os
import random
from tqdm import tqdm
import pickle

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from utils import Args, flat_accuracy
from mymodel import load_model_and_tokenizer
from mydata import MyDataset


# 1. 결과 폴더가 없는 경우
# 먼저 bert_tutorial.ipynb 파일을 통해 한국어 텍스트 데이터의 전처리를 완료하세요

# 2. 학습 파라미터를 설정합니다

# 실험 재현을 위해 시드를 고정해주세요

# 3. 모델과 토크나이저를 불러옵니다

# 4. 데이터를 불러와 데이터셋을 만듭니다

# 5. 학습에 사용할 옵티마이저를 선택하세요 

# 6. 학습 구성
def train():
    pass

if __name__ == '__main__':
# 학습 시작!
    train()