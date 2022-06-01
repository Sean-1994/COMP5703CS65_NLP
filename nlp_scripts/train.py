import torch
import argparse
from seqeval.metrics import classification_report
from torch.utils.data import Dataset, TensorDataset, DataLoader, random_split
from tqdm import tqdm
from utils import VOCAB, label2index, index2label
from model import BERTBiLSTMCRF
from transformers import AutoModel, AutoTokenizer, AdamW

# pre-trained model
bert_model = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"


class NERDataset(torch.utils.data.Dataset):
    """
    The Dataset class is mainly used for reading raw data or basic data processing.
    In this NLP task, the text needs to be converted into the corresponding dictionary ids,
    and this step is performed in the Dataset. Therefore, the NERDataset is a data processor
    that collects data, processes the data of each index, and finally outputs it.
    """

    def __init__(self, encodings, labels, maskings, original):
        """
        Read the original data and initialize the class
        :param encodings: the list of data tokenized
        :param labels: the labels list of data
        :param maskings: the list of masking data
        :param original: the original data
        """
        self.encodings = encodings
        self.labels = labels
        self.maskings = maskings
        self.original = original

    def __getitem__(self, idx):
        """
        Further processing of each data based on subscript
        :param idx: the index of data in the dataset
        :return: Elements which want to take out in the dataset through dataset[index]
        """
        item = {"input_ids": torch.tensor(self.encodings[idx])}
        item['labels'] = torch.tensor(self.labels[idx])
        item['maskings'] = torch.tensor(self.maskings[idx])
        # item['original'] = self.original[idx]
        return item

    def __len__(self):
        """
        :return: the number of datasets
        """
        return len(self.labels)


def load_dataset(data_file):
    """
    Load the input text and split the content and label
    :param data_file:  the input text
    :return: the segmented word list and label list
    """
    with open(data_file, 'r', encoding='UTF-8') as f:
        sentences = []
        tags = []
        lines = f.readlines()
        word = []
        label = []
    for line in lines:
        if line != "\n":
            w, l = line.split("\t")
            word.append(w.lower())
            label.append(l[:-1])
        else:
            if word:
                sentences.append(word)
                tags.append(label)
                word = []
                label = []
    return sentences, tags


def encode_tags(tags):
    """
    Convert 16 labels of raw data to numeric form, the range of digital is from 0 to 15
    :param tags: the raw data's tags
    :return: the corresponding numeric labels
    """
    tags_new = []
    for sent_tag in tags:
        temp = []
        for i in sent_tag:
            # print(i)
            temp.append(label2index[i])
        tags_new.append(temp)
    return tags_new


def bert_tokenize(sentences):
    """
    Use the built-in function to tokenize word provided by the
    pretrained model,which is fast in subsequent calls
    :param sentences: input word list
    :return: the converted list of tokenize id
    """
    Tokenizer = AutoTokenizer.from_pretrained(bert_model, add_special_tokens=False)
    sentences_converted = [Tokenizer.convert_tokens_to_ids(x) for x in sentences]
    return sentences_converted


def generate_NERdataset(sentences, sentences_converted, tags):
    """
    Divide the original data according to the ratio of training set:test set 4:1
    and encapsulate them into the corresponding dataset.
    :param sentences:  the original list of words
    :param sentences_converted: word tokenization approaches
    :param tags: the original label list
    :return: the packaged training dataset and test dataset
    """
    split_size = 0.8
    total_length = len(sentences_converted)
    split_length = int(split_size * total_length)

    train_sentences = sentences_converted[:split_length]
    test_sentences = sentences_converted[split_length:]

    train_labels = tags[:split_length]
    test_labels = tags[split_length:]

    train_masking = []
    test_masking = []

    train_original = sentences[:split_length]
    test_original = sentences[split_length:]

    # For train data
    # Padding of sentences,labels,masking,original according to desired input length
    for i in range(len(train_sentences)):
        if len(train_sentences[i]) >= 100:
            train_sentences[i] = train_sentences[i][:100]
            train_labels[i] = train_labels[i][:100]
            train_masking.append([1 for i in range(100)])
            train_original[i] = train_original[i][:100]
        else:
            train_sentences[i] = train_sentences[i] + [0 for i in range(100 - len(train_sentences[i]))]
            train_labels[i] = train_labels[i] + [1 for i in range(100 - len(train_labels[i]))]
            train_masking.append(
                [1 for i in range(len(train_sentences[i]))] + [0 for i in range(100 - len(train_sentences[i]))])
            train_original[i] = train_original[i] + [0 for i in range(100 - len(train_original[i]))]

    # For test data
    # Padding of sentences,labels,masking,original according to desired input length
    for i in range(len(test_sentences)):
        if len(test_sentences[i]) >= 100:
            test_sentences[i] = test_sentences[i][:100]
            test_labels[i] = test_labels[i][:100]
            test_masking.append([1 for i in range(100)])
            test_original[i] = test_original[i][:100]
        else:
            test_sentences[i] = test_sentences[i] + [0 for i in range(100 - len(test_sentences[i]))]
            test_labels[i] = test_labels[i] + [1 for i in range(100 - len(test_labels[i]))]
            test_masking.append(
                [1 for i in range(len(test_sentences[i]))] + [0 for i in range(100 - len(test_sentences[i]))])
            test_original[i] = test_original[i] + [0 for i in range(100 - len(test_original[i]))]

    train_dataset = NERDataset(train_sentences, train_labels, train_masking, train_original)
    test_dataset = NERDataset(test_sentences, test_labels, test_masking, test_original)

    return train_dataset, test_dataset


def generate_dataloader(batch_size, train_dataset, test_dataset):
    """
    In PyTorch, DataLoader only accepts torch.utils.data.Dataset class as incoming parameter.
    Extract a batch of data and encapsulate it into Dataloader for network training.
    :param batch_size: the size of each batch, in order to prevent memory overflow,
    DataLoder will divide the data set into many sub-data sets according to the set batch_size
    :param train_dataset: generator mechanism, the training set must be a dataset type
    :param test_dataset: generator mechanism, the test set must be a dataset type
    :return: the packaged train dataloader and test dataloader
    """
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_dataloader, test_dataloader


def train(device, train_dataloader, test_dataloader, epochs=20, learning_rate=1e-5):
    """
    Train the dataset and save the trained network model in the current local directory
    :param train_dataloader: the packaged train dataloader
    :param test_dataloader: the packaged test dataloader
    :param epochs: the time of iterations
    :param learning_rate: hyperparameter
    """
    model = BERTBiLSTMCRF(device, num_tags=16).to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    model.train()
    for epoch in range(epochs):
        print("EPOCH" + str(epoch))
        print()
        model.train()
        total_loss = 0.
        for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            optimizer.zero_grad()
            # Extract contents from each batch. They are of the size B*Seq_len
            sent = batch["input_ids"].to(device)
            lab = batch["labels"].to(device)
            mask = batch["maskings"].to(device)
            outputs = model(sent, lab, mask)
            loss = outputs[0]
            loss.backward()
            total_loss += loss.item()
            optimizer.step()
        print("loss is: ", total_loss)
        results = []
        tes_lab = []
        tes_sent = []

        # collect the predicted label and actual label for f1 score testing
        out_label = []
        true_label = []
        model.eval()
        with torch.no_grad():
            for step, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
                sent = batch["input_ids"].to(device)
                mask = batch["maskings"].to(device)
                tes_lab.append(batch['labels'])
                tes_sent.append(sent)
                # There is no "lables" in the test step
                outputs = model(sent, maskings=mask)
                results.append(outputs)
                # 
                out_label.extend(outputs)
                true_label.extend(batch['labels'].numpy().tolist())
        pred_label = []
        actual_label = []
        for i, j in zip(out_label, true_label):
            a1 = []
            p1 = []
            for x, y in zip(i, j):
                index_a = y
                index_p = x
                if index2label[index_p] != "[PAD]":
                    a1.append(index2label[index_a])
                    p1.append(index2label[index_p])
            pred_label.append(p1)
            actual_label.append(a1)
        print("Test classification report is shown as below:\n {}".format(
            classification_report(pred_label, actual_label, digits=4)))
        print("-" * 80)
        print()

    torch.save(model.state_dict(), 'data/NER_model.pt')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch", type=int, default=6)
    parser.add_argument("--epoch", type=int, default=20)
    parser.add_argument("--cuda", dest="cuda", action="store_true", default=False)
    parser.add_argument("--dataset", type=str, default="sample_data.txt")
    hp = parser.parse_args()

    device = torch.device('cpu')
    if hp.cuda:
        device = torch.device('cuda')

    sentences, tags = load_dataset(hp.dataset)
    train_dataset, test_dataset = generate_NERdataset(sentences, bert_tokenize(sentences), encode_tags(tags))
    train_dataloader, test_dataloader = generate_dataloader(hp.batch, train_dataset, test_dataset)
    train(device, train_dataloader, test_dataloader, hp.epoch, hp.lr)
