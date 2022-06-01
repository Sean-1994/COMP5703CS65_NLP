VOCAB = ( 'O', '[PAD]', 'B-DISEASE', 'I-DISEASE', 'B-SYMPTOM', 'I-SYMPTOM', 'B-CAUSE', 'I-CAUSE', 'B-POSITION', 'I-POSITION', 'B-TREATMENT', 'I-TREATMENT', 'B-DRUG', 'I-DRUG', 'B-EXAMINATION', 'I-EXAMINATION')
label2index = {tag: idx for idx, tag in enumerate(VOCAB)}
index2label = {idx: tag for idx, tag in enumerate(VOCAB)}
import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

def get_entities(text,result):
    """
        Extracting the predicted entities by original text and model prediction results.
        :param text: original text
        :param result: prediction result
        :return: A list of (entity_name,predicted tag)
    """
    entities = []
    curr_entity = ''
    curr_tag = ''
    for o,pred in zip(text,result):
        pp = index2label[pred]
        if pp.startswith('B'):
            curr_entity = o
            curr_tag = pp.split('-')[1]
        elif pp.startswith('I'):
            if curr_entity != '':
                curr_entity += ' '
                curr_entity += o
            # else:
                # print("ERROR: An I-label doesn't followed with a B-label")
        else:
            if curr_tag != '':
                entities.append((curr_entity,curr_tag))
            curr_entity = ''
            curr_tag = ''
    if curr_tag != '':
        entities.append((curr_entity,curr_tag))
    return entities

def get_relations(original,predicted):
    """
            Label conversion based on predicted values and document types.
            :param original: document types(DISEASE or SYMPTOM)
            :param predicted: predicted values
            :return: New tag names
        """
    if original == "DISEASE":
        if predicted == "DISEASE":
            return "DISEASE_RELATED_DISEASE"
        elif predicted == "SYMPTOM":
            return "DISEASE_HAS_SYMPTOM"
        elif predicted == "EXAMINATION":
            return "DISEASE_CORRESPONDING_EXAMINATION"
        elif predicted == "TREATMENT":
            return "DISEASE_CORRESPONDING_TREATMENT"
        elif predicted == "DRUG":
            return "DISEASE_CORRESPONDING_DRUG"
        elif predicted == "POSITION":
            return "DISEASE_CORRESPONDING_POSITION"
        elif predicted == "CAUSE":
            return "DISEASE_CORRESPONDING_CAUSE"
        else:
            return "UNKNOWN"
    elif original == "SYMPTOM":
        if predicted == "DISEASE":
            return "SYMPTOM_CORRESPONDING_DISEASE"
        elif predicted == "SYMPTOM":
            return "SYMPTOM_RELATED_SYMPTOM"
        elif predicted == "EXAMINATION":
            return "SYMPTOM_CORRESPONDING_EXAMINATION"
        elif predicted == "TREATMENT":
            return "SYMPTOM_CORRESPONDING_TREATMENT"
        elif predicted == "DRUG":
            return "SYMPTOM_CORRESPONDING_DRUG"
        elif predicted == "POSITION":
            return "SYMPTOM_CORRESPONDING_POSITION"
        elif predicted == "CAUSE":
            return "SYMPTOM_CORRESPONDING_CAUSE"
        else:
            return "UNKNOWN"
    else:
        return "UNKNOWN"

def remove_dup(title,result_list):
    """
        Removes the entity with the same title as the document from the result list.
        :param title: The title of the document
        :param result_list: The list of prediction result
        :return: new result list
    """
    new_res = []
    for item in result_list:
        entity,relation = item
        if entity.lower() == title.lower():
            continue
        else:
            if item not in new_res:
                new_res.append(item)

    return new_res

def remove_by_rule(result_list):
    """
        Removes some bad results by NLTK lemmatizer and rules.
        :param result_list: The list of prediction result
        :return: new result list
    """
    new_res = []
    lem_list = []
    lemmatizer = WordNetLemmatizer()
    for item in result_list:
        entity,relation = item
        lem_set = (lemmatizer.lemmatize(entity),relation)
        if lem_set not in lem_list:
            lem_list.append(lem_set)
            if not entity.endswith("of") or entity.endswith("the") or entity.startswith("of") or entity.startswith("and") or entity.endswith("and") or entity.startswith("`"):
                new_res.append(item)
        else:
            continue
    return new_res


def replace_relations(result_list,title_type):
    """
        Removes some bad results by NLTK lemmatizer and rules.
        :param result_list: The list of prediction result
        :return: new result list
    """
    new_res = []
    for item in result_list:
        entity,relation = item
        new_res.append((entity,get_relations(original = title_type, predicted = relation)))
    return new_res


def convert_into_dict(title, prediction_list, title_type="DISEASE"):
    """
        Convert the result into dict, in order to generate json file.
        :param title: The title of the document
        :param prediction_list: The list of prediction result
        :param title_type: The type of the document title(DISEASE or SYMPTOM)
        :return: result dict
    """
    json_dict = dict()
    json_dict['NAME'] = title
    json_dict['TITLE_TYPE'] = title_type
    for item in prediction_list:
        entity, tag = item
        try:
            json_dict[tag].append(entity)
        except:
            json_dict[tag] = [entity]

    return json_dict

def post_process(title,title_type,results):
    """
        Combination of all post-processing methods.
        :param title: The title of the document
        :param title_type: The type of the document title(DISEASE or SYMPTOM)
        :param results: The list of prediction result
        :return: result dict
    """
    prediction_list = replace_relations(remove_dup(title, remove_by_rule(results)), title_type)
    return convert_into_dict(title,prediction_list,title_type)