
#? PATHs
dataset_dir = "data/arxiv-metadata-oai-snapshot.json"
analogy_dataset_dir = "data/analogy/bats_3.0_combined.txt"
similarity_dataset_dir = "data/similarity/SimVerb-3500/SimVerb-3500.txt"
checkpoint_dir = "embeddings/"
fig_dir = "vis/"

#? Parameters
train_epochs = 500
learning_rate = 0.01
# Full dataset is 3360984
docs_to_look_at = 1
embedding_dim = 2
window_size = 1
save_iteration = 50

#? Parameters for plotting
colors = ["#FF0000", 
          "#00FFFF", 
          "#C0C0C0", 
          "#0000FF", 
          "#808080", 
          "#00008B", 
          "#000000", 
          "#ADD8E6", 
          "#FFA500", 
          "#800080", 
          "#A52A2A", 
          "#FFFF00", 
          "#800000", 
          "#00FF00", 
          "#008000", 
          "#FF00FF", 
          "#808000", 
          "#FFC0CB", 
          "#7FFD4"]
plot_iteration = 10