import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from tqdm import tqdm

BATCH_SIZE = 64
LEARNING_RATE = 2e-5
EPOCHS = 15
MODEL_NAME = 'bert-large-uncased' # bert-base-uncased & bert-large-uncased

# Specify GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def one_test(percentage: int): 
    from sklearn.metrics import f1_score
    # Evaluate model
    def evaluate(model, dataloader):
        model.eval()
        total_correct = 0
        total_examples = 0
        
        all_labels = []
        all_predictions = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validating"):
                input_ids, attention_masks, labels = [tensor.to(device) for tensor in batch]
                outputs = model(input_ids, attention_mask=attention_masks)
                
                # Compute accuracy
                predictions = torch.argmax(outputs.logits, dim=1)
                correct = (predictions == labels).sum().item()
                total_correct += correct
                total_examples += labels.size(0)
                
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predictions.cpu().numpy())
                
        accuracy = total_correct / total_examples
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        return accuracy, f1

    print(f'Start Test on Sparsity: {percentage}%')

    # Load pre-trained model and tokenizer
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME)
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model.bert.embeddings.prune_embeddings(percentage)
    model = model.to(device)
    
    # Fine-tuning
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    # Read the tsv file
    df = pd.read_csv('dataset/CoLA/tokenized/in_domain_train.tsv', delimiter='\t', header=None)

    # Get the texts and labels
    texts = df[3].tolist()
    labels = df[1].tolist()
    tokens = [tokenizer.encode(text, add_special_tokens=True) for text in texts]
    max_len = max([len(token) for token in tokens])
    padded_tokens = [token + [0] * (max_len - len(token)) for token in tokens]
    attention_masks = [[1 if token != 0 else 0 for token in padded] for padded in padded_tokens]
    input_ids = torch.tensor(padded_tokens)
    attention_masks = torch.tensor(attention_masks)
    labels = torch.tensor(labels)
    dataset = TensorDataset(input_ids, attention_masks, labels)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)

    accuracy, f1 = evaluate(model, dataloader)
    print(f'Before Finetuning Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}')

    for epoch in range(EPOCHS):
        model.train()
        for batch in tqdm(dataloader, desc="Training"):
            input_ids, attention_masks, labels = [tensor.to(device) for tensor in batch]
            outputs = model(input_ids, attention_mask=attention_masks, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        accuracy, f1 = evaluate(model, dataloader)
        print(f'Epoch: {epoch}, Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}')
    
    print(f'Model Sparsity After Finetuning: {model.bert.embeddings.commit_sparsity_ratio() * 100:.2f}%')
    print('-' * 40)

def main():
    test_percentage_list = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    # test_percentage_list = [10]
    for percentage in test_percentage_list:
        one_test(percentage)

if __name__ == "__main__":
    main()