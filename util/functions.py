from util import *
from tqdm import tqdm
import json
import string
from nltk.corpus import stopwords


def load_dataset():
    """
    Loads arXiv dataset and returns list.
    INPUTS:
		None
    OUTPUTS:
		dataset ([["word", "word", ...], ["word", "word", ...], ...])
    """
    dataset = []
    def get_metadata(docs_to_look_at):
        count = 0
        with open("data/arxiv-metadata-oai-snapshot.json", "r") as f:
            for line in f:
                count += 1
                yield line
                if count >= docs_to_look_at:
                    return
                
    metadata = get_metadata(docs_to_look_at)
    for paper in tqdm(metadata):
        doc = remove_stopwords(json.loads(paper)["abstract"].translate(str.maketrans("", 
                                                                                     "", 
                                                                                     string.punctuation)).lower().split())
        dataset.append(doc)
    
    return dataset
  
  
stop_words = set(stopwords.words("english"))
def remove_stopwords(sentence):
    """
    """
    return [w for w in sentence if not w in stop_words]