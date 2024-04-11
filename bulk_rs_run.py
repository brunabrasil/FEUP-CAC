import json
from collections import defaultdict
import pandas as pd
from surprise import Dataset, Reader, accuracy, NormalPredictor, KNNBasic, SVD

from utils import load_filtered_data

city = "Springfield"
sna_models = ['categories', 'business_reviews', 'business_tips', 'categories_and_reviews', 'combined', 'friendships', 'priority_combined', 'threshold_categories']
#sna_models = ['priority_combined']

n_recommendations = 1
pos_rating = 2
remove_users_with_less_than_x_reviews = 3


# RS Algorithms
random_algo = NormalPredictor()
ubcf_algo = KNNBasic(sim_options={'user_based': True})
ibcf_algo = KNNBasic(sim_options={'user_based': False})
svd_algo = SVD()

rs_algorithms = [random_algo, ubcf_algo, ibcf_algo, svd_algo]


# Load city data
city_data = load_filtered_data(city)
reviews = city_data['review'][['review_id', 'business_id', 'user_id', 'stars']]
reviews = reviews.groupby(['user_id', 'business_id'])['stars'].mean().reset_index()
reviews.columns = ['user_id', 'business_id', 'rating']


def recommend_top_n(all_recommendations):
    '''
    Recommend top n items for each user
    '''
    recommendations = {}
    for uid, user_recs in all_recommendations.items():
        user_recs_filtered = [(iid, est) for iid, est in user_recs if est > pos_rating] # Dont recommend bellow pos rating
        user_recs_filtered.sort(key=lambda x: x[1], reverse=True)  # Sort by estimated rating in descending order
        top_n_items = [iid for iid, _ in user_recs_filtered[:n_recommendations]]
        recommendations[uid] = top_n_items

    return recommendations


def evaluate_algorithm(algo, trainset, testset):
    '''
    Evaluate the algorithm using RMSE
    '''
    if (len(trainset.all_users()) == 0) or (len(trainset.all_items()) == 0):
        return None

    algo.fit(trainset)
    predictions = algo.test(testset)

    return predictions


results = []

for connection in sna_models:
    for algo in rs_algorithms:
        with open(f'communities/{city}_{connection}_communities.json') as f:
            data = json.load(f)

        community_matrices = {}
        for i, community in enumerate(data['communities']):
            community_matrices[i] = reviews[reviews['user_id'].isin(community)]
            community_matrices[i] = community_matrices[i].pivot_table(index='user_id', columns='business_id',
                                                                      values='rating')

        initial_len = len(community_matrices)

        # remove users with less than x reviews
        community_matrices = {k: v for k, v in community_matrices.items() if v.shape[0] >= remove_users_with_less_than_x_reviews}

        # remove communities with 0-1 users
        community_matrices = {k: v for k, v in community_matrices.items() if v.shape[0] > 1}

        final_len = len(community_matrices)

        sparsity_sum = 0
        sparsity_len = 0
        for community_id, matrix in community_matrices.items():
            sparsity_sum += matrix.isnull().sum().sum()
            sparsity_len += matrix.shape[0] * matrix.shape[1]
        sparsity = sparsity_sum / sparsity_len

        community_trainsets = {}
        community_testsets = {}
        filtered_matrices = {}
        for community_id, matrix in community_matrices.items():

            user_review_counts = matrix.apply(lambda row: row.count(), axis=1)
            users_with_min_reviews = user_review_counts[user_review_counts >= 3].index.tolist()
            matrix_filtered = matrix.loc[users_with_min_reviews]
            df = matrix_filtered.stack().reset_index()
            df.columns = ['user_id', 'business_id', 'rating']

            trainset = defaultdict(list)
            testset = defaultdict(list)
            for user_id, group in df.groupby('user_id'):
                num_reviews = len(group)
                train_size = int(num_reviews * 0.8)
                train_reviews = group[:train_size]
                test_reviews = group[train_size:]
                trainset[community_id].extend(train_reviews.values.tolist())
                testset[community_id].extend(test_reviews.values.tolist())

            reader = Reader(rating_scale=(1, 5))
            train_data = Dataset.load_from_df(
                pd.DataFrame(trainset[community_id], columns=['user_id', 'business_id', 'rating']), reader)
            test_data = Dataset.load_from_df(
                pd.DataFrame(testset[community_id], columns=['user_id', 'business_id', 'rating']), reader)
            
            community_trainsets[community_id] = train_data.build_full_trainset()
            community_testsets[community_id] = test_data.build_full_trainset().build_testset()

        # ------ BINARY METRICS ------
        total_rmse = 0
        total_mae = 0
        count_rmse_mae = 0
        rights = 0
        total_recommendations = 0
        total_good_items = 0
        all_recommendations = {}
        for community_id, trainset in community_trainsets.items():
            testset = community_testsets[community_id]
            predictions = evaluate_algorithm(algo, trainset, testset)
            all_recommendations[community_id] = {}

            if predictions is not None:
                count_rmse_mae += 1
                total_rmse += accuracy.rmse(predictions)
                total_mae += accuracy.mae(predictions)
                for uid, iid, true_r, est, _ in predictions:

                    if uid not in all_recommendations:
                        all_recommendations[community_id][uid] = []

                    all_recommendations[community_id][uid].append((iid, est))

                    if(est > pos_rating):
                        total_recommendations += 1 # count the number of recommendations made

                    if (true_r > pos_rating): # if true rating is greater than pos rating, it is a good item
                        total_good_items += 1 # count the number of good/relevant items 

                        if (est > pos_rating): # if rating estimated also greater than pos_rating, the recommendation is right
                            rights += 1
                
        total_rmse /= count_rmse_mae
        total_mae /= count_rmse_mae

        if total_recommendations == 0:
            precision = 0
        else:
            precision = rights / total_recommendations

        if total_good_items == 0:
            recall = 0
        else:
            recall = rights / total_good_items
        
        if precision + recall == 0:
            f1 = 0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)

        community_test_dfs = {}
        for community_id, testset in community_testsets.items():
            community_test_dfs[community_id] = pd.DataFrame(testset, columns=['user_id', 'item_id', 'rating'])
        
        # ------ RANK METRICS ------

        avg_precision = 0 # sum of avg precision of each user
        n_user = 0 # number of users being recommended

        for community_id in community_trainsets.keys():
            test_df = community_test_dfs[community_id]
            if not test_df.empty:
                if len(community_trainsets[community_id].all_users()) == 0:
                    continue

                best_recommendations = recommend_top_n(all_recommendations[community_id]) # list of best recommendations ordered
                
                for user_id, recs in best_recommendations.items():
                    n_user += 1

                    user_test_df = test_df[test_df['user_id'] == user_id] # testset of user

                    ranked_test = user_test_df[user_test_df['rating'] > pos_rating].sort_values(by='rating', ascending=False) # order test by rating and consider that the user only likes if user greater than pos rating
                    
                    test_top_items = ranked_test['item_id'].tolist()

                    position = 1
                    sum_precision = 0
                    hits = 0
                    if len(test_top_items) > 0:
                        for item in test_top_items:
                            if position <= len(recs):
                                if item == recs[position-1]:
                                    hits += 1 # successful prediction
                                    sum_precision += (hits / position)
                            position += 1
                        avg_precision += sum_precision / len(test_top_items)

        final_avg_precision = avg_precision / n_user # mean of users' avg_precision
            
        if (hasattr(algo, 'sim_options')):
            if algo != svd_algo and algo != random_algo:  # Exclude SVD and NormalPredictor
                boolean_user_based = ' user_based' if algo.sim_options['user_based'] else ' item_based'
            else:
                boolean_user_based = '' 
                
        results.append({
            'city': city,
            'connection': connection,
            'algo': algo.__class__.__name__ +  boolean_user_based,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'avg_precision': final_avg_precision,
            'avg rmse': total_rmse,
            'avg mae': total_mae,
            'sparsity': sparsity,
            'communities initially': initial_len,
            'communities finally': final_len
        })


results_df = pd.DataFrame(results)
results_df.to_csv('results.csv', index=False)