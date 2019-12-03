## Import Libraries and Modules here...
import spacy
import math
from collections import defaultdict


class InvertedIndex:
    # You can define some functions or variables in __init__ function.
    def __init__(self):
        ## You should use these variable to store the term frequencies for tokens and entities...
        self.tf_tokens = defaultdict(dict, {})
        self.tf_entities = defaultdict(dict, {})

        ## You should use these variable to store the inverse document frequencies for tokens and entities...
        self.idf_tokens = defaultdict(dict, {})
        self.idf_entities = defaultdict(dict, {})

        self.query_splits = {}

    ## Your implementation for indexing the documents...
    def index_documents(self, documents):


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
                if ent not in self.tf_entities.keys():
                    self.tf_entities[ent] = dicts
                else:
                    self.tf_entities[ent][id] = dicts[id]

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
                if tok not in self.tf_tokens.keys():
                    self.tf_tokens[tok] = dicts
                else:
                    self.tf_tokens[tok][id] = dicts[id]

        # format idf_entities dict
        for ent in self.tf_entities.keys():
            self.idf_entities[ent] = 1 + math.log(len(documents) / (len(self.tf_entities[ent]) + 1))
        # format idf_tokens dict
        for tok in self.tf_tokens.keys():
            self.idf_tokens[tok] = 1 + math.log(len(documents) / (len(self.tf_tokens[tok]) + 1))


    ## Your implementation to split the query to tokens and entities...
    def split_query(self, Q, DoE):

        eligible_entity = []
        subsets = [[]]
        valid_subsets = []

        # select eligible entity in DoE
        for DOE_value, DOE_key in DoE.items():
            flag = 1
            for ent in DOE_value.split():
                if ent not in Q:
                    flag = 0
                    break
            if flag == 1:
                eligible_entity.append(DOE_value)

        # get all subsets of eligible_entity
        for i in range(len(eligible_entity)):
            for j in range(len(subsets)):
                subsets.append(subsets[j] + [eligible_entity[i]])

        # select valid subsets
        for subset in subsets:
            tempQ = Q
            flag = 1
            for i in range(len(subset)):
                for item in subset[i].split():
                    if item in tempQ:
                        tempQ = tempQ.replace(item, "", 1)
                    else:
                        flag = 0
                        break
            if flag:
                valid_subsets.append(subset)

        # format query_splits dict
        n = 0
        for subset in valid_subsets:
            split_Q = Q.split()
            for item in subset:
                for i in item.split():
                    split_Q.pop(split_Q.index(i))
            self.query_splits[n] = {'tokens': split_Q, 'entities': subset}
            n += 1
        return self.query_splits


    ## Your implementation to return the max score among split_Q the query splits...
    def max_score_query(self, query_splits, doc_id):
        ## Output should be a tuple (max_score, {'tokens': [...], 'entities': [...]})

        score_list = []
        for query_id, query in query_splits.items():
            entities_score = 0
            tokens_score = 0
            entities = query['entities']
            tokens = query['tokens']

            if len(entities) == 0:
                entities_score = 0
            else:
                for ent in entities:
                    if ent in self.tf_entities.keys() and doc_id in self.tf_entities[ent].keys():
                        tf_ent = 1 + math.log(self.tf_entities[ent][doc_id])
                        idf_ent = self.idf_entities[ent]
                        entities_score += tf_ent * idf_ent

            for tok in tokens:
                if tok in self.tf_tokens.keys() and doc_id in self.tf_tokens[tok].keys():
                    tf_tok = 1 + math.log(1 + math.log(self.tf_tokens[tok][doc_id]))
                    idf_tok = self.idf_tokens[tok]
                    tokens_score += tf_tok * idf_tok

            combined_score = entities_score + 0.4 * tokens_score
            score_list.append([entities_score, tokens_score, combined_score, query])
            format_score = {'tokens_score': tokens_score, 'entities_score': entities_score, 'combined_score': combined_score}
            print("query =  ", query)
            print(f'{format_score}\n')

        sort_score = sorted(score_list, key=lambda x: x[2], reverse=True)
        max_score = (sort_score[0][2], sort_score[0][3])
        return max_score


# documents = {1: 'President Trump was on his way to new New York in New York City.',
#              2: 'New York Times mentioned an interesting story about Trump.',
#              3: 'I think it would be great if I can travel to New York this summer to see Trump.'}

# Q = 'New York Times Trump travel'
# DoE = {'New York Times': 0, 'New York': 1, 'New York City': 2}
# doc_id = 3

# if __name__ == '__main__':
#     index = InvertedIndex()
#     index.index_documents(documents)
#     query_splits = index.split_query(Q, DoE)
#     result = index.max_score_query(query_splits, doc_id)
#     print(result)

