from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, df, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.labels = [label for label in df['label']]
        self.texts = [self.tokenizer.encode_plus(
                                text,                       # batch의 text를 받아 인코딩
                                padding = 'max_length',     # max_length까지 패딩
                                add_special_tokens = True,  #[CLS], [SEP]과 같은 special token 추가
                                max_length=self.max_length, # 최대 문장 길이 설정
                                truncation=True,            # 설정한 길이 이상의 토큰은 truncation
                                return_tensors="pt")        # pytorch tensor로 추출 
                                for text in df['text']]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]