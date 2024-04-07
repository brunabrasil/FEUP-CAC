import json
from collections import defaultdict
import pandas as pd
from surprise import Dataset, Reader, accuracy, NormalPredictor, KNNBasic, SVD

from utils import load_filtered_data

city = "Springfield"
sna_models = ['categories', 'business_reviews', 'business_tips', 'categories_and_reviews', 'combined', 'friendships', 'priority_combined', 'threshold_categories']
#sna_models = ['priority_combined']

n_recommendations = 5
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


def recommend_top_n(algo, trainset, testset, n=3):
    '''
    Recommend top n items for a user using a recommender model
    '''

    algo.fit(trainset)
    predictions = algo.test(testset)
    recommendations = {}

    for uid, iid, true_r, est, _ in predictions:
        if uid not in recommendations:
            recommendations[uid] = []
        recommendations[uid].append((iid, est, true_r))

    for uid, user_recs in recommendations.items():
        user_recs.sort(key=lambda x: x[1], reverse=True)  # Sort by estimated rating in descending order
        top_n_recs = [(iid, est, true_r) for iid, est, true_r in user_recs[:n]]  # Get the top n items
        recommendations[uid] = top_n_recs

    return recommendations


def evaluate_algorithm(algo, trainset, testset):
    '''
    Evaluate the algorithm using RMSE
    '''
    if (len(trainset.all_users()) == 0) or (len(trainset.all_items()) == 0):
        print('No data.')
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

        community_matrices = {k: v[v >= 3].dropna(axis=0, how='all') for k, v in community_matrices.items()}

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
            # matrix_filtered = matrix[matrix >= 3].dropna(axis=0, how='all')
            # filtered_matrices[community_id] = matrix_filtered

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

        total_rmse = 0
        total_mae = 0
        count_rmse_mae = 0
        for community_id, trainset in community_trainsets.items():
            testset = community_testsets[community_id]
            predictions = evaluate_algorithm(algo, trainset, testset)
            if predictions is not None:
                count_rmse_mae += 1
                total_rmse += accuracy.rmse(predictions)
                total_mae += accuracy.mae(predictions)

        total_rmse /= count_rmse_mae
        total_mae /= count_rmse_mae

        community_test_dfs = {}
        for community_id, testset in community_testsets.items():
            community_test_dfs[community_id] = pd.DataFrame(testset, columns=['user_id', 'item_id', 'rating'])
        

        hits = 0
        total = 0
        total_gt = 0
        for community_id in community_trainsets.keys():
            test_df = community_test_dfs[community_id]
            if not test_df.empty:
                recommendations = {}
                if len(community_trainsets[community_id].all_users()) == 0:
                    continue
                recommendations = recommend_top_n(algo, community_trainsets[community_id], community_testsets[community_id], n_recommendations)

                for user_id, recs in recommendations.items():
                    print(recs)
                    gt = test_df[(test_df['user_id'] == user_id) & (test_df['rating'] > pos_rating)].item_id.to_list()
                    if len(recs) > 0 and len(gt) > 0:
                        for item_id, estimated_rating, true_rating in recs:
                            if true_rating == round(estimated_rating, 1):
                                hits += 1
                        total += len(recs)  
                    total_gt += len(gt)

        if total == 0:
            precision = 0
        else:
            precision = hits / total
        if total_gt == 0:
            recall = 0
        else:
            recall = hits / total_gt
        if precision + recall == 0:
            f1 = 0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
        

        if (hasattr(algo, 'sim_options')):
            if algo.sim_options['user_based'] is None:
                boolean_user_based = ' '
            elif algo.sim_options['user_based']:
                boolean_user_based = ' user_based'
            elif algo.sim_options['user_based'] == False:
                boolean_user_based = ' item_based'
                
        results.append({
            'city': city,
            'connection': connection,
            'algo': algo.__class__.__name__ +  boolean_user_based,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'avg rmse': total_rmse,
            'avg mae': total_mae,
            'sparsity': sparsity,
            'communities initially': initial_len,
            'communities finally': final_len
        })


results_df = pd.DataFrame(results)
results_df.to_csv('results.csv', index=False)