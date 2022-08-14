from util import *
from util.functions import *
from util.models import *
from util.plotting import *
import sys


dataset = load_dataset()

# Initialize Skip-gram model 
model = skipgram(dataset)
 
model.train()

print(model.predict("higgs", 5))

plot_points_and_vocab(model.U, 
                      model.vocabulary, 
                      embedding_dim, 
                      save_plots=False,
                      fig_show=True)