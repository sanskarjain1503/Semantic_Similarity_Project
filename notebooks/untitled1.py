import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import pickle

model = SentenceTransformer('stsb-roberta-large')

sentence1 = 'hye my name is sankar jain'
sentence2 = 'i am in final year'

embedding1 = model.encode(sentence1, convert_to_tensor=True)
embedding2 = model.encode(sentence2, convert_to_tensor=True)

cosine_scores = util.pytorch_cos_sim(embedding1, embedding2)
print("Sentence 1:", sentence1)
print("Sentence 2:", sentence2)
print("Similarity score:", cosine_scores.item())

pickle.dump(model,open('model.pkl','wb'))
pickle.dump(cosine_scores,open('cosine_scores.pkl','wb'))
