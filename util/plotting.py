import plotly.express as px
import numpy as np
from . import *
import pandas as pd
import sys
from pathlib import Path


def plot_points_and_vocab(embeddings, 
                          vocab, 
                          dimensions, 
                          save_plots,
                          fig_show,
                          name=""):
    """
    Plots learned embeddings. Only works for 2 dimensions.
    INPUTS:     Vectors, List of vocab, 
                Int indicating dimensions
    RETURNS:    None
    """
    df = pd.DataFrame(columns=["x", "y", "word"])
    
    for index, word in enumerate(vocab):
        x = np.array(embeddings[index][0])
        y = np.array(embeddings[index][1])
        df = pd.concat([df, 
                        pd.DataFrame([[x, y, word]], columns=["x", "y", "word"])], 
                        ignore_index = True, 
                        axis = 0)
        
    fig = px.scatter(df, x="x", y="y", text="word")
    fig.update_yaxes(scaleanchor = "x", scaleratio = 1)
    fig.update_traces(textposition='top center')
    
    if fig_show:
        fig.show()
    if save_plots:
        fig.write_image(fig_dir + str(name) + ".png")
