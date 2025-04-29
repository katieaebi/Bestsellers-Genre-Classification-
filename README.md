# Genre Classification with NYT Bestsellers 
## About the Project 
This project was originally created as a part of my Data Science Portfolio class at Belmont University. It utilizes some NYT Bestseller datasets from the year 2023, which I built in a [previous project](https://github.com/katieaebi/Unifying-Data-in-Literary-Publishing/blob/main/README.md). For this project, my primary goal was to build a classification model to predict a book's genre based off of its description. My secondary goal was to fill some of the gaps in the datasets, as well as add some additional features. This is an ongoing project, so there will be future updates from time to time. I'm currently finished with my classification model, but since my gap-filling ambitions are still ongoing, I'd like to retry classification with the full datasets in the future. Thus, my analysis will likely be subject to change in the future as well. 
## Preparation
### Step 1: download the datasets 
Navigate to [my previous project](https://github.com/katieaebi/Unifying-Data-in-Literary-Publishing/blob/main/README.md), and download all of the datasets I used in the form of .txt files. Each dataset is its own NYT Bestseller category (ex: hardcover fiction, paperback nonfiction, etc.). One slightly funky thing about these datasets is that each row represents a certain rank from a certain week in 2023. This is because the NYTBooks API returns the bestseller rankings for a specified week. So, for example, below is pictured a few columns from the first four rows from the hardcover fiction dataset. 
photo
After downloading the datasets onto your local machine, put all the files in the same directory as your project and load them into your code using the line of code below.
```
data = {} #initializing a dictionary to access all the categories from the same place  
for i, category in enumerate(os.listdir(os.path.join('.', '2023_bestsellers'))): data[category[:-4]] = pd.read_csv(os.path.join('.', '2023_bestsellers', category))
```
### Step 2: obtain an API key from ISBNDB
Since the original datasets don't have a genre feature, I used the ISBNDB API for getting genre labels for each title. These will be our base truth for classification. ISBNDB does require a paid subscription to access its database, but I used the basic plan, which is $14.95 a month and allows up to 5000 requests to the API per day, and that was more than enough for this project. They also offer a free 7-day trial for the basic plan, if you have absolutely no other uses for it than this. To create an ISBNDB account, navigate [here](https://isbndb.com) and click "register."

### Step 3: Start pulling genre labels
To use the API, you'll want to create a header varriable with your authorization key. In the example below, you'll replace the underscores with the API key you obtained from step 2. The variable r will be the object for making requests and accessing the data returned. The string of numbers at the end of the URL specifies the ISBN of the book you're requests data for. In the example below, I was making a test request with the first ISBN from the hardcover fiction category (Lessons in Chemistry, if you're curious). In the actual loop I wrote to iterate over each title, I used an f-string to replace this value for each iteration.
photo
Naturally, some requests will return a 404 status code, or might not have the genres feature. To keep track of which ISBNs from each category had an issue, I created a "problems" dictionary to save a list of the problematic ISBNs for each category. My plan was to later use these ISBNs for webscraping, but that step is still in progress. 

### Step 4: Pull descriptions 
Most of the datasets did not have any gaps in the description column, but there were three categories that were missing descriptions for the entire dataset. Hence, I wanted to fill those before moving on to classification. I essentially repeated the same process as pulling the genre labels, except iterating over just the three categories and grabbing descriptions instead of subjects. 

### Step 5: Preprocess data
In order to use the genre labels as my target, I needed to convert the labels to something my classifier could process and predict. I decided to convert the genre labels into a binary dataframe so that each row could represent a book, each column would represent a unique genre, and each value would represent if that genre applied (1) or didn't apply (2) to that book. Then, to filter out any genre labels that were over or under present in the data, I removed any columns/genres whose sum was greater than 90% of the samples or less than 1% of the samples.  
Then I moved on to the descriptions. I concatenated all descriptions from across the categories into one object. Then I iterated over each description in this object to remove stopwords, stem words, and convert to tf-idf vectors. (If you're unfamiliar with tf-idf vectors, it stands for Term Frequency-Inverse Document Frequency. They're basically vector representations of text that weight term frequencies based off the principle that the rarest words in a text are the most important to the meaning.) 

### Step 6: Classification 
Since each book belonged to multiple genre labels, I used sklearn's [OneVsRestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsRestClassifier.html) for my classification model with a [Support Vector Machine](https://scikit-learn.org/stable/modules/svm.html) as my estimator. I used Kfold cross validation to split the data 3 times, and I created an ROC curve for each split, which are pictured below. The mean accuracy across the splits was 69.96%. 
photos 
The accuracy score was better than I expected it to be, although it was only moderate and still significantlly lower than the AUC would suggest. I suspect that the difference between the metrics comes from the fact that this is one-vs-rest classification, so the model might do a good job of knowing what genres don't apply to a book but struggle to know which genres do apply. Additionally, these curves are micro-averaging, which means that each label is considered a binary prediction (see [here](https://scikit-learn.org/0.15/auto_examples/plot_roc.html#:~:text=In%20order%20to%20extend%20ROC,prediction%20(micro%2Daveraging)) for more information). In the future, I'd like to try some hyperparameter tuning on my classifier, as well as possibly one-vs-one classification. 

### Webscraping 
I am currently still in the process of web scraping to fill holes in the data, but if you would like to take a look at how I've gona about navigating with the webdriver, then you can find my code thus far here. 
I used Selenium to try to web scrap data off of the popular website Goodreads. If you have a Goodreads account and would like to recreate my progress so far, replace the underscores in the code for your personal account login information. 
