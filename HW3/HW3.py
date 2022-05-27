import sys
import csv
import random
import numpy
import scipy
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction import stop_words
from nltk.tokenize import TreebankWordTokenizer


# Global parameters
exclude_stop_words = True  # If True, then exclude stop words from counts/frequencies
use_tfidf = True  # If True, then use word frequencies; else use word counts
ranking_limit = 50  # Limit to number of movies in final rankings

# Dictionaries to hold data read in from files
movie_title = {}  # dict of movie title by movieId
movie_year = {}  # dict of movie year by movieId
movie_genres = {}  # dict of genres, list of genre keywords, by movieId
movie_plot = {}  # dict of movie plot by movieId
movie_imdb_rating = {}  # dict movie IMDb rating by movieId
user_ratings = {}  # dict of ratings, list of (movieId,rating,timestamp), by userId

# Word vectors
genres_vect = []
titles_count_vect = None
titles_tfidf_transformer = None
plots_count_vect = None
plots_tfidf_transformer = None

# Global variables to hold training data
X = None  # will be a sparse matrix
y = []  # list of target classes
w = []  # list of sample weights


# ----- Data Processing -----

def read_data():
    global movie_title, movie_year, movie_genres, movie_plot, movie_imdb_rating, user_ratings
    # read movie titles, years, and genres
    with open('../HW3/movies.csv') as csv_file:
        csv_reader = csv.reader(csv_file)
        line_num = 0
        for row in csv_reader:
            if line_num > 0:  # skip header
                movieId = int(row[0])
                title_year = row[1]
                genres = row[2]
                movie_title[movieId] = title_year[:-7]
                movie_year[movieId] = int(title_year[-5:-1])
                if genres == "(no genres listed)":
                    movie_genres[movieId] = []
                else:
                    movie_genres[movieId] = genres.split('|')
            line_num += 1
    # read movie plots
    with open('../HW3/plots-imdb.csv') as csv_file:
        csv_reader = csv.reader(csv_file)
        line_num = 0
        for row in csv_reader:
            if line_num > 0:  # skip header
                movieId = int(row[0])
                plot = row[1]
                movie_plot[movieId] = plot
            line_num += 1
    # read movie IMDb ratings
    with open('../HW3/ratings-imdb.csv') as csv_file:
        csv_reader = csv.reader(csv_file)
        line_num = 0
        for row in csv_reader:
            if line_num > 0:  # skip header
                movieId = int(row[0])
                rating = float(row[1])
                movie_imdb_rating[movieId] = rating
            line_num += 1
    # read user ratings of movies
    with open('../HW3/ratings.csv') as csv_file:
        csv_reader = csv.reader(csv_file)
        line_num = 0
        for row in csv_reader:
            if line_num > 0:  # skip header
                userId = int(row[0])
                movieId = int(row[1])
                rating = float(row[2])
                timestamp = int(row[3])
                user_rating = (movieId, rating, timestamp)
                if userId in user_ratings:
                    user_ratings[userId].append(user_rating)
                else:
                    user_ratings[userId] = [user_rating]
            line_num += 1


def generate_feature_vector(example):
    # Returns feature vector for example, where example is of the form
    # (userId,movieId1,movieId2,rating_diff). The feature vector consists
    # of the year, genre bag of words, title bag of words, and plot bag of
    # words for both movies. Assumes proper word vectorization processing
    # has already been done (extract_text_features() already called).
    movie1_fv = movie_feature_vector(example[1])
    movie2_fv = movie_feature_vector(example[2])
    return movie1_fv + movie2_fv


def movie_feature_vector(movieId):
    global movie_year, movie_title, movie_plot, use_tfidf
    global titles_count_vect, titles_tfidf_transformer, plots_count_vect, plots_tfidf_transformer
    movie_year_fv = [movie_year[movieId]]
    genre_fv = genre_feature_vector(movieId)
    title_fv = titles_count_vect.transform([movie_title[movieId]])
    if use_tfidf:
        title_fv = titles_tfidf_transformer.transform(title_fv)
    plot_fv = plots_count_vect.transform([movie_plot[movieId]])
    if use_tfidf:
        plot_fv = plots_tfidf_transformer.transform(plot_fv)
    return movie_year_fv + genre_fv + list(title_fv.toarray()[0]) + list(plot_fv.toarray()[0])


def genre_feature_vector(movieId):
    # Return Boolean vector indicating which genre keywords associated with given movie.
    global movie_genres
    genre_fv = []
    movie_genre_words = movie_genres[movieId]
    for genre_word in genres_vect:
        if genre_word in movie_genre_words:
            genre_fv.append(1.0)
        else:
            genre_fv.append(0.0)
    return genre_fv


def extract_text_features(dataset):
    # Compute word vectors for movie genres, titles, and plots, for movies in dataset.
    # Genre keywords are fixed, but the titles and plots are free-form text,
    # so we use NLTK to tokenize words, and then SciKit Learn's text-based
    # feature extraction tools to generate the word vectors.
    # Assume read_data() has already been called.
    global movie_genres, movie_title, movie_plot, genres_vect, titles_count_vect, plots_count_vect
    global titles_tfidf_transformer, plots_tfidf_transformer
    global exclude_stop_words, use_tfidf
    # Get movieIds mentioned in dataset
    movieIds = []
    for example in dataset:
        movieId1 = example[1]
        movieId2 = example[2]
        if movieId1 not in movieIds:
            movieIds.append(movieId1)
        if movieId2 not in movieIds:
            movieIds.append(movieId2)
    # Movie genres
    word_set = set()
    for movieId in movieIds:
        genre_words = movie_genres[movieId]  # genres already converted to list of words
        word_set = word_set.union(set(genre_words))
    genres_vect = list(word_set)
    # Movie titles
    tokenizer = TreebankWordTokenizer()
    titles_count_vect = CountVectorizer()
    titles_count_vect.set_params(tokenizer=tokenizer.tokenize)
    if exclude_stop_words:
        titles_count_vect.set_params(stop_words='english')
    # titles_count_vect.set_params(ngram_range=(1,2)) # include 1-grams and 2-grams
    titles_count_vect.set_params(max_df=0.5)  # ignore terms that appear in >50% of the titles
    titles_count_vect.set_params(min_df=2)  # ignore terms that appear in only 1 title
    titles = list(map(lambda k: movie_title[k], movieIds))
    titles_count_vect.fit(titles)
    if use_tfidf:
        title_counts = titles_count_vect.transform(titles)
        titles_tfidf_transformer = TfidfTransformer()
        titles_tfidf_transformer.fit(title_counts)
    # Movie plots
    plots_count_vect = CountVectorizer()
    plots_count_vect.set_params(tokenizer=tokenizer.tokenize)
    if exclude_stop_words:
        plots_count_vect.set_params(stop_words='english')
    # plots_count_vect.set_params(ngram_range=(1,2)) # include 1-grams and 2-grams
    plots_count_vect.set_params(max_df=0.5)  # ignore terms that appear in >50% of the plots
    plots_count_vect.set_params(min_df=2)  # ignore terms that appear in only 1 plot
    plots = list(map(lambda k: movie_plot[k], movieIds))
    plots_count_vect.fit(plots)
    if use_tfidf:
        plot_counts = plots_count_vect.transform(plots)
        plots_tfidf_transformer = TfidfTransformer()
        plots_tfidf_transformer.fit(plot_counts)


def build_dataset(movie_id_limit=0):
    global user_ratings  # dict of user ratings, list of (movieId,rating,timestamp), by userId
    # Returns dataset consisting of (userId,movieId1,movieId2,rating_diff), where rating_diff
    # is (user_rating(movie1) - user_rating(movie2)), assuming the difference is not zero (no ties).
    # If movie_id_limit > 0, then examples restricted to movieId's <= movie_id_limit.
    dataset = []
    posegs = 0
    negegs = 0
    for userId in user_ratings:
        ratings = remove_duplicate_ratings(user_ratings[userId])
        for rating1 in ratings:
            movieId1 = rating1[0]
            r1 = rating1[1]
            if (movie_id_limit == 0) or (movieId1 <= movie_id_limit):
                for rating2 in ratings:
                    movieId2 = rating2[0]
                    r2 = rating2[1]
                    if (movie_id_limit == 0) or (movieId2 <= movie_id_limit):
                        if (movieId1 != movieId2) and (r1 != r2):
                            dataset.append((userId, movieId1, movieId2, r1 - r2))
    print("  " + str(len(dataset)) + " egs, movie id limit = " + str(movie_id_limit))
    return dataset


def remove_duplicate_ratings(ratings):
    # If ratings contains multiple ratings for the same movie, then remove all but the most recent rating.
    # Each rating is a tuple (movieId,rating,timestamp).
    unique_ratings = []
    for rating1 in ratings:
        movieId1 = rating1[0]
        timestamp1 = rating1[2]
        best_rating = True
        for rating2 in ratings:
            movieId2 = rating2[0]
            timestamp2 = rating2[2]
            if (movieId1 == movieId2) and (timestamp1 < timestamp2):
                best_rating = False
                break
        if best_rating:
            unique_ratings.append(rating1)
    return unique_ratings


def build_training_set(dataset):
    # Construct sparse matrix X of feature vectors for each example in dataset.
    # Construct target classes y and sample weights w for each example in dataset.
    # Dataset entries are (userId,movieId1,movieId2,rating_diff).
    # Example features are based on movie1+movie2. Since most feature
    # values will be 0.0, and we have a lot of features, need to use
    # a sparse matrix. The targets are y=-1 if rating_diff < 0; otherwise y=1.
    # Sample weights are based on abs(rating_diff).
    global X, y, w
    X_data = []
    X_row = []
    X_col = []
    row = 0
    num_egs = len(dataset)
    for example in dataset:
        # if (row % 100) == 0:
        #    print("  processing example " + str(row) + " of " + str(num_egs))
        fvec = generate_feature_vector(example)
        col = 0
        for fval in fvec:
            if fval != 0.0:
                X_data.append(fval)
                X_row.append(row)
                X_col.append(col)
            col += 1
        row += 1
        rating_diff = example[3]
        if rating_diff < 0:
            y.append(-1)  # movie1 rated lower than movie2 by user
        else:
            y.append(1)  # movie1 rated higher than movie2 by user
        w.append(abs(rating_diff))  # cost of misclassification proportional to difference in ratings
    X_data_arr = numpy.array(X_data)
    X_row_arr = numpy.array(X_row)
    X_col_arr = numpy.array(X_col)
    X = scipy.sparse.csr_matrix((X_data_arr, (X_row_arr, X_col_arr)), shape=(num_egs, len(fvec)))


# ----- Ranking Processing -----

def get_imdb_ranking(ranking):
    # Returns IMDb's ranking of movies in given ranking. Returned ranking is a list
    # of (movieId,imdb_rating) pairs, sorted in decreasing order by imdb_rating.
    global movie_imdb_rating
    imdb_ratings = list(map(lambda pair: (pair[0], movie_imdb_rating[pair[0]]), ranking))
    imdb_ranking = sorted(imdb_ratings, key=lambda pair: pair[1], reverse=True)
    return imdb_ranking


def print_ranking(ranking):
    # Print ranking, which is a list of (movieId,score) pairs, sorted in decreasing order by score.
    rank = 1
    for movie_score_pair in ranking:
        movieId = movie_score_pair[0]
        score = movie_score_pair[1]
        title = movie_title[movieId]
        print(str(rank) + ". (" + str(score) + ") " + title + " (id=" + str(movieId) + ")")
        rank += 1


def compare_rankings(ranking1, ranking2):
    # Return evaluation of comparing given rankings (ordered lists of (movieId,score) pairs).
    # Specifically, compute the average distance between each movie's ranking.
    # This is the Kemeny distance measure. The function assumes the same
    # movieIds are in both rankings.

    global movie_imdb_rating
    movieIds1 = list(map(lambda pair: pair[0], ranking1))
    movieIds2 = list(map(lambda pair: pair[0], ranking2))
    distance = 0
    rank1 = 0
    for movieId in movieIds1:
        rank2 = movieIds2.index(movieId)
        distance += abs(rank1 - rank2)
        rank1 += 1
    return float(distance) / float(len(ranking1))


# ----- Naive Ranking -----

def naive_rank_train(classifier):
    # Assumes build_training_set already called to build X and y.
    return classifier.fit(X, y)


def naive_rank_test(dataset, classifier):
    # Return list of (movieId, score) pairs sorted in decreasing order by score.
    # The classifier is used to predict preference between each pair of movies.
    # initialize movie scores dictionary
    scores_dict = {}
    for example in dataset:
        movieId1 = example[1]
        movieId2 = example[2]
        if movieId1 not in scores_dict:
            scores_dict[movieId1] = 0
        if movieId2 not in scores_dict:
            scores_dict[movieId2] = 0
    # update movie scores based on all possible pairs of movies
    for movieId1 in scores_dict:
        for movieId2 in scores_dict:
            if movieId1 != movieId2:
                dummy_example = (0, movieId1, movieId2, 0)
                fv = generate_feature_vector(dummy_example)
                y = classifier.predict([fv])[0]
                scores_dict[movieId1] += y
                scores_dict[movieId2] -= y
    # sort movies by score
    sorted_scores = sorted(scores_dict.items(), key=lambda kv: kv[1], reverse=True)
    return sorted_scores


# ----- Ranking v2 -----

def rank_train_2(classifier):
    # Assumes build_training_set already called to build X, y, and w.
    # The use of w is the main difference between rank_train_2 and naive_rank_train.
    global X, y, w
    return classifier.fit(X, y, sample_weight=w)


def rank_test_2(dataset, classifier):
    # Return list of (movieId, score) pairs sorted in decreasing order by score.
    # The classifier is used to predict preference between each pair of movies.
    # initialize movie scores dictionary
    movieIds = []
    for example in dataset:
        movieId1 = example[1]
        movieId2 = example[2]
        if movieId1 not in movieIds:
            movieIds.append(movieId1)
        if movieId2 not in movieIds:
            movieIds.append(movieId2)
    ranked_movieIds = rank_test_2_quicksort(movieIds, classifier)
    # Rank Test v2 doesn't produce movie ranking scores, so just set each movie's score to 0
    return list(map(lambda id: (id, 0), ranked_movieIds))


def rank_test_2_quicksort(movieIds, classifier):
    if len(movieIds) < 2:
        return movieIds
    else:
        pivot = random.choice(movieIds)
        left = []
        right = []
        for movieId in movieIds:
            if movieId != pivot:
                dummy_example = (0, movieId, pivot, 0)
                fv = generate_feature_vector(dummy_example)
                class_1_index = numpy.where(classifier.classes_ == 1)[0]
                y_prob = classifier.predict_proba([fv])[0][class_1_index]  # probability movie preferred to pivot
                if numpy.random.uniform(0, 1) < y_prob:
                    left.append(movieId)
                else:
                    right.append(movieId)
        left = rank_test_2_quicksort(left, classifier)
        right = rank_test_2_quicksort(right, classifier)
        return left + [pivot] + right


# ----- Main -----

def main():
    global movie_title, ranking_limit
    # print("Reading data...", flush=True)
    read_data()
    # print("Building dataset...", flush=True)
    dataset = build_dataset(200)  # parameter is number of movies
    # print("Extracting text features...", flush=True)
    extract_text_features(dataset)
    # print("Building training set...", flush=True)
    build_training_set(dataset)

    # Sample for testing
    example = dataset[0]
    # print("  Example: " + str(example), flush=True)
    fv = generate_feature_vector(example)
    # print("  Feature vector (#features = " + str(len(fv)) + "):", flush=True)
    # print("  " + str(fv))

    # Naive Ranking
    print("\nNaive Ranking: training classifier...", flush=True)
    classifier = MultinomialNB()
    classifier = naive_rank_train(classifier)
    naive_ranking = naive_rank_test(dataset, classifier)[:ranking_limit]
    imdb_ranking = get_imdb_ranking(naive_ranking)
    dist = compare_rankings(naive_ranking, imdb_ranking)
    print("\nNaive Ranking (distance to IMDb ranking = " + str(dist) + "):", flush=True)
    print_ranking(naive_ranking)
    print("IMDb Ranking:", flush=True)
    print_ranking(imdb_ranking)

    # Ranking v2
    print("\nRanking v2: training classifier...", flush=True)
    classifier = MultinomialNB()
    classifier = rank_train_2(classifier)
    ranking2 = rank_test_2(dataset, classifier)[:ranking_limit]
    imdb_ranking2 = get_imdb_ranking(ranking2)
    dist = compare_rankings(ranking2, imdb_ranking2)
    print("\nRanking v2 (distance to IMDb ranking = " + str(dist) + "):", flush=True)
    print_ranking(ranking2)
    print("IMDb Ranking:", flush=True)
    print_ranking(imdb_ranking2)


main()