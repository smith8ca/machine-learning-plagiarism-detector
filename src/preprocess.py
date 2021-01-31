import pandas as pd
import numpy as np
import os
import helpers


csv_file = 'data/file_information.csv'
plagiarism_df = pd.read_csv(csv_file)


# TESTING: Print out the first few rows of data info
# print(plagiarism_df.head())

# Create new `transformed_df`
transformed_df = helpers.numerical_dataframe(csv_file)

# TESTING: Print out the transformed dataframe
# print('\nExample data: ')
# print(transformed_df.head())

# Create a text column
text_df = helpers.create_text_column(transformed_df)

# Check out the processed text for a single file, by row index
row_idx = 0
sample_text = text_df.iloc[0]['Text']

# TESTING: Print out the sample processed text
# print('Sample processed text:\n', sample_text)


# STRATIFIED SAMPLING
random_seed = 1

# Create new dataframe with Datatype (train, test, orig) column
# Pass `text_df` from above to create a complete dataframe, with all the information you need
complete_df = helpers.train_test_dataframe(text_df, random_seed=random_seed)

# TESTING: Check results of the complete dataframe
# print(complete_df.head(10))

# CONTAINMENT CALCULATION
n = 3   # Select a value for n
test_indices = range(5)   # Indices for first few files

# Iterate through files and calculate containment
category_vals = []
containment_vals = []
for i in test_indices:
    # Get level of plagiarism for a given file index
    category_vals.append(complete_df.loc[i, 'Category'])

    # Calculate containment for given file and n
    filename = complete_df.loc[i, 'File']
    c = helpers.calculate_containment(complete_df, n, filename)
    containment_vals.append(c)

# TESTING: Print out result
# print('Original category values: \n', category_vals)
# print()
# print(str(n)+'-gram containment values: \n', containment_vals)


A = "i think pagerank is a link analysis algorithm used by google that uses a system of weights attached to each element of a hyperlinked set of documents"
S = "pagerank is a link analysis algorithm used by the google internet search engine that assigns a numerical weighting to each element of a hyperlinked set of documents"

# Calculate LCS
lcs = helpers.lcs_norm_word(A, S)

# TESTING: Expected value test
# print('LCS = ', lcs)
# assert lcs==20/27., "Incorrect LCS value, expected about 0.7408, got " + str(lcs)
# print('Test passed!')


test_indices = range(5)   # look at first few files

category_vals = []
lcs_norm_vals = []

# Iterate through first few docs and calculate LCS
for i in test_indices:
    category_vals.append(complete_df.loc[i, 'Category'])

    # get texts to compare
    answer_text = complete_df.loc[i, 'Text']
    task = complete_df.loc[i, 'Task']

    # we know that source texts have Class = -1
    orig_rows = complete_df[(complete_df['Class'] == -1)]
    orig_row = orig_rows[(orig_rows['Task'] == task)]
    source_text = orig_row['Text'].values[0]

    # calculate lcs
    lcs_val = helpers.lcs_norm_word(answer_text, source_text)
    lcs_norm_vals.append(lcs_val)


# TESTING: print out result
# print('Original category values: \n', category_vals)
# print()
# print('Normalized LCS values: \n', lcs_norm_vals)


# CREATING LCS FEATURES

# Create a features DataFrame by selecting an ngram_range
ngram_range = range(1, 7)   # Define an ngram range
features_list = []

# Create features in a features_df
all_features = np.zeros((len(ngram_range)+1, len(complete_df)))

# Calculate features for containment for ngrams in range
i = 0
for n in ngram_range:
    column_name = 'c_'+str(n)
    features_list.append(column_name)

    # create containment features
    all_features[i] = np.squeeze(
        helpers.create_containment_features(complete_df, n))
    i += 1


# Calculate features for LCS_Norm Words
features_list.append('lcs_word')
all_features[i] = np.squeeze(helpers.create_lcs_features(complete_df))

# create a features dataframe
features_df = pd.DataFrame(np.transpose(all_features), columns=features_list)

# TESTING: Print all features/columns
# print()
# print('Features: ', features_list)
# print()
# print (features_df.head(10))


# CORRELATED FEATURES
# Create correlation matrix for just Features to determine different models to test
corr_matrix = features_df.corr().abs().round(2)

# TESTING: shows all of a dataframe
# print (corr_matrix)


# CREATE SELECTED TRAIN/TEST/DATA
test_selection = list(features_df)[:2]  # first couple columns as a test

# test that the correct train/test data is created
(train_x, train_y), (test_x, test_y) = helpers.train_test_data(
    complete_df, features_df, test_selection)

# Select your list of features, this should be column names from features_df
selected_features = ['c_1', 'c_5', 'lcs_word']

(train_x, train_y), (test_x, test_y) = helpers.train_test_data(
    complete_df, features_df, selected_features)

# TESTING: Check that division of samples seems correct; These should add up to 95 (100 - 5 original files)
# print('Training size: ', len(train_x))
# print('Test size: ', len(test_x))
# print()
# print('Training df sample: \n', train_x[:10])


# CREATE FINAL DATA FILES


# Test cells
fake_x = [[0.39814815, 0.0001, 0.19178082],
          [0.86936937, 0.44954128, 0.84649123],
          [0.44086022, 0., 0.22395833]]

fake_y = [0, 1, 1]

helpers.make_csv(fake_x, fake_y, filename='to_delete.csv', data_dir='test_csv')

# Read in and test dimensions
fake_df = pd.read_csv('test_csv/to_delete.csv', header=None)

# Check shape
assert fake_df.shape == (3, 4), \
    'The file should have as many rows as data_points and as many columns as features+1 (for indices).'

# TESTING: Check that first column = labels
# assert np.all(fake_df.iloc[:,0].values==fake_y), 'First column is not equal to the labels, fake_y.'
# print('Tests passed!')

data_dir = 'models'
helpers.make_csv(train_x, train_y, filename='train.csv', data_dir=data_dir)
helpers.make_csv(test_x, test_y, filename='test.csv', data_dir=data_dir)
