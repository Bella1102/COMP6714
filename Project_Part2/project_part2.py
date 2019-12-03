import numpy as np
import xgboost as xgb
import spacy
import math
from collections import defaultdict



def index_documents(documents):

    tf_tokens = defaultdict(dict, {})
    tf_entities = defaultdict(dict, {})
    idf_tokens = defaultdict(dict, {})
    idf_entities = defaultdict(dict, {})

    nlp = spacy.load("en_core_web_sm")

    # loop each document
    for id, doc in documents.items():
        nlp_texts = nlp(doc)

        # format tf_entities dict
        entity_list = []
        for i in nlp_texts.ents:
            entity_list.append(i.text)
        for ent in entity_list:
            dicts = {}
            dicts[id] = entity_list.count(ent)
            if ent not in tf_entities.keys():
                tf_entities[ent] = dicts
            else:
                tf_entities[ent][id] = dicts[id]

        # format tf_tokens dict
        token_list = []
        for i in nlp_texts:
            # remove stopword and punctiation
            if i.is_stop == False and i.is_punct == False:
                token_list.append(i.text)
        # 一个单词既是entity也是token:遍历entity,在entity中出现一次就在token中remove一次
        # single-token entities: only compute the TF-IDF index of the entities
        for i in entity_list:
            if i in token_list:
                token_list.remove(i)
        for tok in token_list:
            dicts = {}
            dicts[id] = token_list.count(tok)
            if tok not in tf_tokens.keys():
                tf_tokens[tok] = dicts
            else:
                tf_tokens[tok][id] = dicts[id]

    # format idf_entities dict
    for ent in tf_entities.keys():
        idf_entities[ent] = 1 + math.log(len(documents) / (len(tf_entities[ent]) + 1))
    # format idf_tokens dict
    for tok in tf_tokens.keys():
        idf_tokens[tok] = 1 + math.log(len(documents) / (len(tf_tokens[tok]) + 1))

    return tf_tokens, tf_entities, idf_tokens, idf_entities


def setFeatures(entity, value, train_doc, candidates, tf_tokens, tf_entities, idf_tokens, idf_entities, parsed_entity_pages):

    # store all features
    features_list = []

    # feature_1: tf-idf of each token
    feature_1 = 0
    for tuples in parsed_entity_pages[entity]:
        token = tuples[1]
        if train_doc in tf_tokens[token].keys():
            tf_token = 1 + math.log(tf_tokens[token][train_doc])
            idf_token = idf_tokens[token]
            feature_1 += tf_token * idf_token
    features_list.append(feature_1)

    # feature_2: tf-idf of each entity
    feature_2 = 0
    entity_words = entity.replace('_', ' ')
    if train_doc in tf_entities[entity_words].keys():
        tf_entity = 1 + math.log(1 + math.log(tf_entities[entity_words][train_doc]))
        idf_entity = idf_entities[entity_words]
        feature_2 += tf_entity * idf_entity
    features_list.append(feature_2)

    # feature_3: tf-idf of each word in mention's 'mention'
    feature_3 = 0
    mention_words = value['mention'].split()
    for word in mention_words:
        if train_doc in tf_tokens[word].keys():
            tf_mention_word = 1 + math.log(tf_tokens[word][train_doc])
            idf_mention_word = idf_tokens[word]
            feature_3 += tf_mention_word * idf_mention_word
    features_list.append(feature_3)


    # feature_4: counts same token in entity and mention
    feature_4 = 0
    # feature_5: is the tf-idf of the lower case of token
    feature_5 = 0
    for token in entity.lower().split('_'):
        if token in value['mention'].lower().split(' '):
            feature_4 += 1
        if train_doc in tf_tokens[token].keys():
            tf_token = 1 + math.log(tf_tokens[token][train_doc])
            idf_token = idf_tokens[token]
            feature_5 += tf_token * idf_token
    features_list.append(feature_4)
    features_list.append(feature_5)

    # feature_6: the number of tokens in each entity
    feature_6 = len(entity.split('_'))
    features_list.append(feature_6)

    # feature_7: the number of words in 'mention'
    feature_7 = len(mention_words)
    features_list.append(feature_7)

    # feature_8: the difference of entity and mention's length
    features_8 = len(entity) - len(value['mention'])
    features_list.append(features_8)

    return features_list


def disambiguate_mentions(train_mentions, train_labels, dev_mentions, men_docs, parsed_entity_pages):

    tf_tokens, tf_entities, idf_tokens, idf_entities = index_documents(men_docs)

    # for train
    train_data = []
    train_group = []
    train_label = []
    for key, value in train_mentions.items():
        train_group.append(len(value['candidate_entities']))
        train_doc = train_mentions[key]['doc_title']
        candidates = value['candidate_entities']
        ground_label = train_labels[key]['label']

        for entity in candidates:
            if entity == ground_label:
                train_label.append(1)
            else:
                train_label.append(0)
            one_train_data = setFeatures(entity, value, train_doc, candidates, tf_tokens, tf_entities, idf_tokens, idf_entities, parsed_entity_pages)
            train_data.append(one_train_data)

    #  for test
    dev_data = []
    dev_group = []
    for key, value in dev_mentions.items():
        dev_group.append(len(value['candidate_entities']))
        dev_doc = dev_mentions[key]['doc_title']
        candidates = value['candidate_entities']
        for entity in candidates:
            one_dev_data = setFeatures(entity, value, dev_doc, candidates, tf_tokens, tf_entities, idf_tokens, idf_entities, parsed_entity_pages)
            dev_data.append(one_dev_data)


    def transform_data(features, groups, labels=None):
        xgb_data = xgb.DMatrix(data=features, label=labels)
        xgb_data.set_group(groups)
        return xgb_data

    train_data = np.array(train_data)
    dev_data = np.array(dev_data)
    xgboost_train = transform_data(train_data, train_group, train_label)
    xgboost_dev = transform_data(dev_data, dev_group)
    param = {'max_depth': 8, 'eta': 0.05, 'silent': 1, 'objective': 'rank:pairwise', 'min_child_weight': 0.01, 'lambda':100}
    evallist = [ (xgboost_train, 'train') ]
    classifier = xgb.train(param, xgboost_train, num_boost_round=4900, evals=evallist, early_stopping_rounds=10)
    predicts = classifier.predict(xgboost_dev)

    idx = 0
    res = {}
    for i in range(len(dev_mentions)):
        key = list(dev_mentions)[i]
        value = dev_mentions[list(dev_mentions)[i]]
        entity_list = []
        for entity in value['candidate_entities']:
            entity_list.append(entity)
        res[key] = entity_list[np.argmax(predicts[idx: idx+dev_group[i]])]
        idx += dev_group[i]
    return res

