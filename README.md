NLP Project:
    Exploring the tweets of World Leaders ["@BorisJohnson", "@realDonaldTrump", "@LeoVaradkar"] in 2020:

Data:
* Novel Coronavirus (COVID-19) Cases, provided by JHU from https://github.com/CSSEGISandData/COVID-19.

Training dataset:
** Kaggle's sentiment140 dataset. (It contains 1,600,000 tweets).
* International Workshop on Semantic Evaluation (SemEval), a semantic evaluation forum previously known as SensEval tasks
* Kaggle airline sentiment
* manually labelled leader tweets

Model:
Train using my own simple Sentiment Analyser.

------------------------------------------------------------
Show:
i) Baseline model: Stats like accuracy
ii) Spot check examples
iii) Try Different Classifiers
iv) Try different building of corpus:
 - Bigrams & Trigrams
 - Lemmatizer instead of Stemmer
 - Spell Correction
 - Not removing stop words
v) Compare my model with that of:
 - nltk vader
 - Fasttext library from fb.

------------------------------------------------------------
Plots & correlation coefficients for each leader:

1. Average daily sentiment of leader tweets & retweets vs:
 a) # of country cases
 b) # of country deaths
 c) # of global cases
 d) # of global deaths
 e) With average sentiment of other leaders

2. Average daily sentiment of leader tweets in march 2020 vs nov 2019

3. Average Sentiment of corona tweets vs average sentiment of non-corona tweets over the same time period.
(contrast my own method vs nltk vader). Using labelled dataset from x.
|DATE |Boris |Trump |Leo |
|1 Mar|  2   |  -1  |  0 |
|2 Mar|  1   |  0   |  1 |


------------------------------------------------------------
Learn:
* Sparse to dense matrices.
* Covariance matrix
* Feature weighting to understand influences
* Switching to try a Lemmatizer rather than a Stemmer

