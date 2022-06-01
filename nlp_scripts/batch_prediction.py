from main import new_word_tokenize,predict_in_doc
from model import BERTBiLSTMCRF
from transformers import AutoTokenizer
from utils import post_process
from tqdm import tqdm
import argparse
import jsonlines
import torch
import json
import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
nltk.download('punkt')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", dest="cuda", action="store_true", default=False)
    parser.add_argument("--input_data", type=str, default="data/mayo_sentences.json")
    parser.add_argument("--output_data", type=str, default="data/output_data.json")
    parser.add_argument("--type", type=str, default="DISEASE")
    parser.add_argument("--model", type=str, default="NER_model.pt")
    hp = parser.parse_args()

    device = torch.device('cpu')
    if hp.cuda:
        device = torch.device('cuda')

    jsons_list = []
    with open(hp.input_data, "r+", encoding="utf8") as f:
        for item in jsonlines.Reader(f):
            jsons_list.append(item)
    text_list = []
    for idx, jsons in enumerate(jsons_list):
        text = jsons['text']
        title = jsons['name']
        text_list.append((title, hp.type, text))

    bert_model = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
    Tokenizer = AutoTokenizer.from_pretrained(bert_model,add_special_tokens=False)
    model = BERTBiLSTMCRF(device, num_tags = 16).to(device)
    model.load_state_dict(torch.load(hp.model,map_location=device))
    model.eval()

    rst_list = []
    for title, title_type, input_text in tqdm(text_list):
        # split txt into sentences
        sentence_lst = sent_tokenize(input_text)
        # remove last element
        sentence_lst = sentence_lst[:-1]
        sentences = new_word_tokenize(sentence_lst)
        test = predict_in_doc(model, sentences)
        rst = post_process(title, title_type, test)
        rst_list.append(rst)
        # print(rst)

    with open(hp.output_data, 'w') as f:
        f.write(json.dumps(rst_list))
