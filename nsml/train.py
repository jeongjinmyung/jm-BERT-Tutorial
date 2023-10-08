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
args = Args()

# 실험 재현을 위해 시드를 고정해주세요
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

# 3. 모델과 토크나이저를 불러옵니다
model, tokenizer = load_model_and_tokenizer(
    args.model_name, args.tokenizer_name, num_classes=7
)

# 4. 데이터를 불러와 데이터셋을 만듭니다
train_data = pd.read_csv(os.path.join(args.dataset_dir, "train_df.csv"))
valid_data = pd.read_csv(os.path.join(args.dataset_dir, "valid_df.csv"))
test_data = pd.read_csv(os.path.join(args.dataset_dir, "test_df.csv"))

train_dataset = MyDataset(
    df=train_data, tokenizer=tokenizer, max_length=args.sentence_max_len
)
valid_dataset = MyDataset(
    df=valid_data, tokenizer=tokenizer, max_length=args.sentence_max_len
)
test_dataset = MyDataset(
    df=test_data, tokenizer=tokenizer, max_length=args.sentence_max_len
)

train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size)
test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)

# 5. 학습에 사용할 옵티마이저를 선택하세요
optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=args.eps)
total_steps = len(train_dataloader) * args.epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=0, num_training_steps=total_steps
)


# 6. 학습 구성
def train():
    # 학습 결과 저장
    training_stats = []

    model.to(args.device)

    for epoch in range(args.epochs):
        # 실제 학습이 진행된 cuda 시간 측정
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()

        # 모델 학습 결과 저장
        total_train_loss = 0
        # 모델 학습
        model.train()
        for train_input, train_label in tqdm(train_dataloader):
            trian_input_ids = train_input["input_ids"].squeeze(1).to(args.device)
            train_input_mask = train_input["attention_mask"].to(args.device)
            train_label = train_label.to(args.device)

            optimizer.zero_grad()

            outputs = model(
                trian_input_ids,
                token_type_ids=None,
                attention_mask=train_input_mask,
                labels=train_label,
            )

            loss, logits = outputs["loss"], outputs["logits"]

            total_train_loss += loss.item()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

        avg_train_loss = total_train_loss / len(train_dataloader)

        # 평가
        model.eval()
        # 모델 평과 결과 저장
        total_eval_accuracy = 0
        total_eval_loss = 0

        for val_input, val_label in valid_dataloader:
            valid_input_ids = val_input["input_ids"].squeeze(1).to(args.device)
            valid_input_mask = val_input["attention_mask"].to(args.device)
            valid_label = val_label.to(args.device)

            with torch.no_grad():
                outputs = model(
                    valid_input_ids,
                    token_type_ids=None,
                    attention_mask=valid_input_mask,
                    labels=valid_label,
                )

                loss, logits = outputs["loss"], outputs["logits"]

            total_eval_loss += loss.item()

            logits = logits.detach().cpu().numpy()
            label_ids = valid_label.to("cpu").numpy()

            total_eval_accuracy += flat_accuracy(logits, label_ids)

        avg_val_loss = total_eval_loss / len(valid_dataloader)
        avg_val_accuracy = total_eval_accuracy / len(valid_dataloader)

        end_event.record()
        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event)

        training_stats.append(
            {
                "Epoch": epoch + 1,
                "Train_Loss": avg_train_loss,
                "Valid_Loss": avg_val_loss,
                "Valid_Accuracy": avg_val_accuracy,
                "Elapsed_Time": elapsed_time,
            }
        )

        print(
            f"Epochs: {epoch + 1} \
            | Train Loss: {avg_train_loss: .3f} \
            | Val Loss: {avg_val_loss: .3f} \
            | Val Accuracy: {avg_val_accuracy: .3f} \
            | Elapsed time: {elapsed_time: .3f} ms"
        )

        torch.save(
            model,
            os.path.join(args.result_dir, f"BERT_Classification_{epoch+1}epoch.pt"),
        )

    # 결과 저장
    with open(os.path.join(args.result_dir, "training_results.pickle"), "wb") as fw:
        pickle.dump(training_stats, fw)


if __name__ == "__main__":
    # 학습 시작!
    train()
