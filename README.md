NLP Project:
    Exploring the tweets of World Leaders ["@BorisJohnson", "@realDonaldTrump", "@LeoVaradkar"] in 2020:

Data:
* Covid Cases from European Center for Disease Control and Prevention (ECDC).
* Training dataset from Kaggle's sentiment140 dataset. (It contains 1,600,000 tweets).

Model:
Train using my own simple Sentimenter (re-watch video).

For each leader:
1. Number of tweets mentioning of 'corona' or 'covid' & when they started.
2. Plot of mentions over time (per date).
3. Average Sentiment of corona tweets vs average sentiment of non-corona tweets over the same time period.
(contrast my own method vs nltk vader). Using labelled dataset from x.
|DATE |Boris |Trump |Leo |
|1 Mar|  2   |  -1  |  0 |
|2 Mar|  1   |  0   |  1 |

Analyse:
Compare my home-rolled tweet sentiment score with that from the Fasttext library from fb.

Plots:
For each leader:
a) Covid vs Regular tweets per day.
b) Overall sentiment of tweets per day

Correlations Matrix of:
i) Sentiment between leaders across time.
ii) Sentiment of leader on day vs new cases on that day in country
iii) Sentiment of leader on day vs new deaths on that day in country


Learn:
* Sparse to dense matrices.
* Covariance matrix
* Feature weighting to understand influences
* Switching to try a Lemmatizer rather than a Stemmer
* Emoticons etc

