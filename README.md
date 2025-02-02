# NLP Coursework Task 1: Regression
Our team chose to tackle the first task, that of regressing to the mean funniness score.
We've split our approach into three parts, each with a corresponding notebook plus an extra notebook that contains code to replicate older LSTM experiments.

## Bi-LSTM approach
The bi_lstm_preliminary_experiments notebook contains code to replicate older experiments that required significantly different network architectures compared to the last experiment. The variable 'experiment_type' that takes the values 'cosine_distance', 'insert_after', 'original_representation', 'simple_difference' determines which experiment is going to be run. The 'freeze_embeddings' flag determines whether the embedding layer is going to be finetuned or frozen.

The bi_lstm_approach notebook contains code to replicate the results of the last experiment. To validate the results simply re-run all the cells in the notebook.

## Contextual Embeddings
The second notebook contains our work using BERT-style embeddings. A `model_type` flag, that takes the arguments `distilbert` and `roberta` controls whether a DistilBert or a RoBERTa-Large-MNLI model is used as the backbone. There's a separate section for our regression and our averaged classification model, which are described in the report. The hyper-parameter sets are also controlled by the `model_type` flag, and are the best we found during hyper-parameter tuning.

## Handcrafted Features

## 

```python
import nltk

nltk.download("averaged_perceptron_tagger")
nltk.download('wordnet')
```
Simply run the cells in order to replicate the results of report.


## Data exploration and Post-Training Analysis
The additional data exploration and the post-training analysis we detail in our report is a part of our BERT notebook. To replicate our data exploration, along with the awesome word clouds that didn't make the cut for the report, you should:
- change the data directories to point to where you locally store the dataset

To replicate our post-training experiments:
- set the `model_type` flag to `distilbert`
- Run the regression model
- Manually load the LSTM and handcrafted predictions as `.npy` files  
- Run the relevant section in the notebook
## Authors
Bouas, Nikos; Clive, Jordan; Siomos, Vasilis 
