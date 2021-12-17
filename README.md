# Population's Attitudes Towards Trump

## Data Story Website
The webside of our story is https://wsssy.github.io/AttitudeToTrump/


## Abstract 📖

In recent years, Donald Trump is no doubt one of the most
controversial public figures around the world. From winning the 2016
presidential election as a dark horse to losing the election by a
narrow margin in late 2020, Trump has made a significant impact on
the United States and the world during the four years of his presidency.

We are going to discover what has changed for quotes on Trump and the reason behind these changes. At the beginning, we first discover the change of Trump's social circle by analysing who talked most about him at different times. Then we investigate the changes over people's words when they talked about Trump. After that, we go one step further to analyse the semantics of quotes rather than analysing on the level of words. Finally, we try to identify the potential factors that would have caused these changes.

## **Research Questions** ❓

- **Question 1**. Back in 2016, it was a big surprise that Trump was elected as the president despite his lack of political experience, which we believe should draw the attention of many, who would have never talked about Trump, to the newly elected president. Therefore, we would like to find out whether the people who talk about Trump have changed.

- **Question 2**. We are interested in what words people usually use to talk about Trump and also how these words and their frequency changes before and after the election.

- **Question 3**. Apart from changes in people's use of vocabulary, we would also want to get a higher level view of what people say, which leads us to analyse the semantics of people’s words based on one feature (i.e. region).

- **Question 4**. Having a general view of semantics of people’s word, we would like to combine multiple features and do the clustering of speakers, and then analyse if there are different words contents among different clusters.


##  **Methods** 🔨

### 1. **Quotebank Data Cleaning**

Since we mainly focus on words on Trump, we first need to extract the quotations mentioning Donald Trump and America. We filter out quotations whose speaker is “None”, keep the quotation containing the keywords related to Donald Trump or
America and only keep the quotation whose phase is ‘E’. After that,
we learn about speakers’ personal information from wikidata through
**‘qid’**s.

### 2. **Sentiment Analysis**

By applying **vaderSentiment** package, we could get the negtive, positive or neural of each quote with probability, an example of the result for each is like:

```{neg: 0.0, pos: 0.0, neu: 0.5, compound: 0.0}```

### 3. **Word Frequencies Counting**

To get a basic idea of the vocabulary people use, we count the occurrences of words for each year after applying the usual text cleaning pipeline, i.e., tokenization and removal of stop words.

### 4. **Word Embeddings Using BERT**

We use BERT to get a vector representation of the semantics of the
quotes so that we can analyse what different people are interested in
under different situations (e.g., before and after certain events).

### 5. **Wikidata Cleaning and processing**

We aggregate all the categorical fields of a speaker into one array
with the label of the fields prepended, which we then convert to an
indicator vector whose columns represent one category. The categorical
fields we consider are as follows:

 - nationality
 - gender
 - party
 - academic degree
 - candidacy
 - religion

We also calculate the ages of the speakers from their dates of
birth. After that, we plot the distribution of the ages of the
speakers and filter out people aged over 120, which we consider
irrelevant to our analysis. We make use of the semantic links in the Wikidata database in order to get fewer categorical values with broader semantics.

### 6. **Clustering and Feature Selection**

By clustering the samples using k-prototype, our target speakers will
be placed into different groups according to their attributes from
Wikidata. We also categorise people by analysing graphs built using
the semantic data from Wikidata.

### 7. **Analysis of how geographic location influence people’s words**  

We investigate people’s words by their geographic information learnt from Wikidata and study the different styles in which people from different parts of the world speak. We classify the quotes according to their sentiments (either positive or negative) and further uncover the difference by looking at the word frequencies.

### 8. **Visualizaiton with PCA**
In order to have a clear view on our BERT vectors, we use PCA to reduce BERT vector from 1024 dimentions to 2-dimention for different situations and analyses.

## Outside DataSet
We use Wiki Data to extract speaker's information. The link: https://www.wikidata.org/wiki/Wikidata:Main_Page

## **Structure of the Codebase**

milestone3.ipynb
lib/feature_semantic.py
lib/__init__.py

We place some helper functions under the module `lib`.

## **Organization within the team** 👨‍👩‍👧‍👦

### Milestone 2

**Jinyi Xian**: initial data analysis, data wrangling, sentiment
analysis

**Sijia Du**: initial data analysis, data wrangling, sentiment
analysis

**Huan Yang**: initial data analysis, write readme file, come up with
research questions and possible methods

**Siyi Wang**: initial data analysis, write readme file, come up with
and improve proposal

### Milestone 3

**Jinyi Xian**: Cluster analysis, visualization, research question1,
research question 3, write data story.

**Sijia Du**: Visualization, research question1, research question2,
write data story.

**Huan Yang**: Cluster analysis, visualization, research question2,
research question 3, write data story.

**Siyi Wang**: Visualization, research question1, research question 3,
correlation analysis, write data story.



