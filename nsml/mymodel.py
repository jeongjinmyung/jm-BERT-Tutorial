from transformers import AutoModelForSequenceClassification, AutoTokenizer

def load_model_and_tokenizer(model_name, tokenizer_name, num_classes):
    model = AutoModelForSequenceClassification.from_pretrained(
                                            model_name, 
                                            num_labels = num_classes,
                                            output_attentions = False,
                                            output_hidden_states = False,
                                            )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    return model, tokenizer