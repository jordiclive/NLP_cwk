# NLP Coursework Task 1: Regression
Our team chose to tackle the first task, that of regressing to the mean funniness score.
We've split our approach into three parts, each with a corresponding notebook.

## Bi-LSTM approach

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install foobar.

```bash
pip install foobar
```

## 

```python
import foobar

foobar.pluralize('word') # returns 'words'
foobar.pluralize('goose') # returns 'geese'
foobar.singularize('phenomena') # returns 'phenomenon'
```

## Contextual Embeddings
The second notebook contains our work using BERT-style embeddings. A `model_type` flag, that takes the arguments `distilbert` and `roberta` controls whether a DistilBert or a RoBERTa-Large-MNLI model is used as the backbone. There's a separate section for our regression and our averaged classification model, which are described in the report. The hyper-parameter sets are also controlled by the `model_type` flag, and are the best we found during hyper-parameter tuning.

## Handcrafted Features

## 

```python
import nltk

nltk.download("averaged_perceptron_tagger")
nltk.download('wordnet')
```
Run the cells in order to replicate the results in the report of: feature extraction, and the LightGBM training procedure. 


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
