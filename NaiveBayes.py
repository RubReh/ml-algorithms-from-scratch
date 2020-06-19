import pandas as pd



class NaiveBayes:

    def __init__(self, data_with_labels: pd.DataFrame) -> None:

        self._data_with_labels = data_with_labels

    '''
    Returns the original input dataframe
    '''
    def _get_data_with_labels(self):

        return self._data_with_labels

    '''
    Creates a dataframe with each data point along 
    with the bayseian statistic for success and non success
    '''
    def train(self) -> pd.DataFrame:

        prepared_data = self._prepare_data(self._get_data_with_labels())
        return self._calculate_bayseian_statistic(prepared_data)

    '''
    Prepares the data with for training
    '''
    def _prepare_data(self, data_with_labels: pd.DataFrame)->pd.DataFrame:

        data_with_labels.columns = ['input_data','labels']
        df = self._get_dataframe_with_total_counts(data_with_labels)
        df = self._get_dataframe_of_counts_for_success_and_non_success(df)
        df = self._compensate_data_frame_label_for_zero_probability(df)

        return df

    '''
    Makes a dataframe containing the counts for each word
    '''
    def _get_dataframe_with_total_counts(self, df:pd.DataFrame) -> pd.DataFrame:

        # Count some totals of each word
        total_counts = pd.DataFrame(df['input_data'].value_counts().reset_index())
        total_counts.columns = ['input_data', 'total_counts']

        return pd.merge(df, total_counts, on='input_data')

    '''
    Makes a dataframe containing the counts of both success 
    and non success for each data input
    '''
    def _get_dataframe_of_counts_for_success_and_non_success(self, df:pd.DataFrame) -> pd.DataFrame:

        # Count spams and non spams for each word
        spam_words = df.loc[df['labels'] == 1]  # Filter out the success cases
        spam_word_counts = spam_words['input_data'].value_counts()
        spam_word_counts = pd.DataFrame(spam_word_counts.reset_index())
        spam_word_counts.columns = ['input_data', 'success_case_count']
        merge = pd.merge(df, spam_word_counts, on='input_data') # Join the data with original dataframe

        merge['non_success_count'] = merge['total_counts'] - merge['success_case_count']
        merge = merge[['input_data', 'total_counts', 'success_case_count', 'non_success_count']].drop_duplicates()

        return merge

    '''
    Handles the issue where there might be 0 successes or non successes 
    for a certain input data which breaks the bayseian approach. 
    Solves this by adding one entry for all success and non success cases
    '''
    @staticmethod
    def _compensate_data_frame_label_for_zero_probability(df: pd.DataFrame)->pd.DataFrame:

        df[['non_success_count', 'success_case_count']] = df[['non_success_count', 'success_case_count']] + 1
        df['total_counts'] = df['total_counts'] + 2

        return df

    '''
    Calculates and returns the prior probabilites of success and non success 
    and the total counts of success and non success
    '''
    def _get_priors_and_counts(self, df:pd.DataFrame) -> (int, int, int, int):

        sums = df.sum(axis=0)
        total_success_count = sums['success_case_count']
        total_non_success_count = sums['non_success_count']
        total = total_success_count + total_non_success_count

        # P(success) & P(not success)
        p_success = total_success_count / total
        p_non_success = total_non_success_count / total

        return p_success, p_non_success, total_success_count, total_non_success_count

    '''
    Use bayes' theorem to mark each label with the probablity of success and non success
    '''
    def _calculate_bayseian_statistic(self, df:pd.DataFrame) -> pd.DataFrame:
        p_success, p_non_success, total_success_count, total_non_success_count = self._get_priors_and_counts(df)

        # Get P( Word | Spam) and P(Word | Not spam)
        df['input_data|success'] = df['success_case_count'] / total_success_count
        df['input_data| not success'] = df['non_success_count'] / total_non_success_count
        df = df[['input_data', 'input_data|success', 'input_data| not success']]

        df['succ_baysian'] = df['input_data|success'] * p_success
        df['non_succ_baysian'] = df['input_data| not success'] * p_non_success

        return df[['input_data','succ_baysian','non_succ_baysian']]



#Implementation
df = pd.read_csv('./files/bayes_data.csv')
test = NaiveBayes(df)








