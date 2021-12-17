import json
import re
import numpy as np
import random
from sympy import nextprime
import pandas as pd
from difflib import SequenceMatcher
import sys
import copy
from scipy.cluster import hierarchy

# Always print the complete output matrices
#np.set_printoptions(threshold=np.inf)

################################### PRE-SELECTION ###################################

########### STEP 0 - DATA CLEANING ###########
# Done manually

########### STEP 1 - CREATING BINARY VECTORS ###########
def binary_vectors_Hartveld(titles, KVPs):
    n_descriptions = len(titles)

    # Find all model words
    model_words_title = []
    regex_title = re.compile(r'([a-zA-Z0-9]*(([0-9]+[ˆ0-9, ]+)|([ˆ0-9, ]+[0-9]+))[a-zA-Z0-9]*)')
    model_words_KVP = []
    regex_KVP = re.compile(r'(ˆ\d+(\.\d+)?[a-zA-Z]+$|ˆ\d+(\.\d+)?$)')

    for i in range(0, n_descriptions):
        title_i = titles[i]
        model_words_title_i = regex_title.findall(title_i)

        n_mw_title_i = len(model_words_title_i)
        if n_mw_title_i > 0:
            for j in range(0, n_mw_title_i):
                if model_words_title_i[j][0] not in model_words_title:
                    model_words_title.append(model_words_title_i[j][0])

        # Find all model words in the key-value pairs
        KVP_i = list(KVPs[i].items())
        n_KVP_i = len(KVP_i)

        for k in range(0, n_KVP_i):
            value_ik = KVP_i[k][1]
            model_words_value_ik = regex_title.findall(value_ik)

            n_mw_value_ik = len(model_words_value_ik)
            if n_mw_value_ik > 0:
                for l in range(0, n_mw_value_ik):
                    if model_words_value_ik[l][0] not in model_words_KVP:
                        model_words_KVP.append(model_words_value_ik[l][0])

    # Create the binary vectors
    n_model_words_title = len(model_words_title)
    n_model_words_KVP = len(model_words_KVP)
    n_model_words = n_model_words_title + n_model_words_KVP

    vectors_Hartveld = np.zeros((n_model_words, n_descriptions))

    for i in range(0, n_descriptions):
        title_i = titles[i]

        for j in range(0, n_model_words_title):
            if model_words_title[j] in title_i:
                vectors_Hartveld[j, i] = 1

        KVP_i = list(KVPs[i].items())
        n_KVP_i = len(KVP_i)

        for k in range(0, n_KVP_i):
            value_ik = KVP_i[k][1]

            for l in range(0, n_model_words_KVP):
                if model_words_KVP[l] in value_ik:
                    vectors_Hartveld[l+n_model_words_title, i] = 1

    return vectors_Hartveld

def binary_vectors_Rigutto(titles, KVPs):
    n_descriptions = len(titles)

    # Construct a list of unique model IDs mentioned in the product description titles
    ids_in_titles = []
    for i in range(0, n_descriptions):
        # Regex searches for strings that are a combination of capital letters and digits (i.e. ID codes)
        id_in_title = re.findall('([A-Z0-9]*(([0-9]+[ˆ0-9,-]+)|([ˆ0-9,]+[0-9]+))[A-Z0-9]{6,})|$', titles[i])[0]
        # Add valid codes to the list
        if len(id_in_title[0]) > 4 and id_in_title[0] not in ids_in_titles:
            ids_in_titles.append(id_in_title[0])

    # Construct a list of unique other IDs found in the key-value pairs
    unique_UPC = []
    unique_provider = []
    unique_ASIN1 = []
    unique_ASIN2 = []
    unique_man_part_number = []
    for i in range(0, n_descriptions):
        KVP_i = list(KVPs[i].items())
        n_KVP_i = len(KVP_i)

        for j in range(0, n_KVP_i):
            key_j = KVP_i[j][0]
            value_j = KVP_i[j][1]

            if key_j == "UPC" and value_j not in unique_UPC:
                unique_UPC.append(value_j)
            elif key_j == "Provider Product ID" and value_j not in unique_provider:
                unique_provider.append(value_j)
            elif key_j == "ASIN:" and value_j not in unique_ASIN1:
                unique_ASIN1.append(value_j)
            elif key_j == "ASIN" and value_j not in unique_ASIN2:
                unique_ASIN2.append(value_j)
            elif key_j == "Manufacturer Part Number:" and value_j not in unique_man_part_number:
                unique_man_part_number.append(value_j)

    # Create the binary vectors
    n_id_codes_titles = len(ids_in_titles)
    n_id_codes_KVPs = len(unique_UPC) + len(unique_provider) + len(unique_ASIN1) + len(unique_ASIN2) + len(unique_man_part_number)

    vectors_Rigutto = np.zeros(((3*n_id_codes_titles+n_id_codes_KVPs), n_descriptions))

    # Check for each description title whether it contains a model ID mentioned in another description title
    for i in range(0, n_descriptions):
        for j in range(0, n_id_codes_titles):
            if ids_in_titles[j] in titles[i]:
                vectors_Rigutto[3*j, i] = 1
                vectors_Rigutto[3*j + 1, i] = 1
                vectors_Rigutto[3*j + 2, i] = 1

        KVP_i = list(KVPs[i].items())
        n_KVP_i = len(KVP_i)

        for k in range(0, n_KVP_i):
            key_ik = KVP_i[k][0]
            value_ik = KVP_i[k][1]

            if key_ik == "UPC":
                #print("We're now checking product", i, 'which has a key "UPC" with value', value_ik)
                for c in range(0, len(unique_UPC)):
                    if unique_UPC[c] == value_ik:
                        vectors_Rigutto[3*n_id_codes_titles + c, i] = 1
                        #print(value_ik, 'matches with the', c, 'th item in the list, namely:', unique_UPC[c])

            if key_ik == "Provider Product ID":
                for c in range(0, len(unique_provider)):
                    if unique_provider[c] == value_ik:
                        vectors_Rigutto[3*n_id_codes_titles + len(unique_UPC) + c, i] = 1

            if key_ik == "ASIN:":
                for c in range(0, len(unique_ASIN1)):
                    if unique_ASIN1[c] == value_ik:
                        vectors_Rigutto[3*n_id_codes_titles + len(unique_UPC) + len(unique_provider) + c, i] = 1

            if key_ik == "ASIN":
                for c in range(0, len(unique_ASIN2)):
                    if unique_ASIN2[c] == value_ik:
                        vectors_Rigutto[3*n_id_codes_titles + len(unique_UPC) + len(unique_provider) + len(unique_ASIN1) + c, i] = 1

            if key_ik == "Manufacturer Part Number:":
                for c in range(0, len(unique_man_part_number)):
                    if unique_man_part_number[c] == value_ik:
                        vectors_Rigutto[3*n_id_codes_titles + len(unique_UPC) + len(unique_provider) + len(unique_ASIN1) + len(unique_ASIN2) + c, i] = 1

    return vectors_Rigutto


########### STEP 2 - MIN-HASHING ###########
def pickRandomCoeffs(k):
    # Create a list of 'k' random values.
    randList = []

    while k > 0:
        # Get a random shingle ID.
        randIndex = random.randint(0, 5109)

        # Ensure that each random number is unique.
        while randIndex in randList:
            randIndex = random.randint(0, 5109)

            # Add the random number to the list.
        randList.append(randIndex)
        k = k - 1

    return randList


def min_hash(binary_vectors):
    n_hashes = 5000
    n_model_words = len(binary_vectors)
    n_descriptions = len(binary_vectors[0])

    print('The vectors from this bootstrap sample have dimensions', n_model_words, 'by', n_descriptions)
    k = nextprime(n_model_words - 1)
    signature_matrix = np.full((n_hashes, n_descriptions), np.inf)

    # Loop over all products
    for p in range(0, len(binary_vectors[0])):
        hash_evaluations = np.empty((n_hashes, 1))
        for r in range(0, len(binary_vectors)):
            # Evaluate n hash functions on every row index of product p
            coeffA = pickRandomCoeffs(n_hashes)
            coeffB = pickRandomCoeffs(n_hashes)
            for i in range(0, n_hashes):
                hash_evaluations[i, 0] = (coeffA[i] + coeffB[i]*r) % k

            # For each hash function, add its minimum outcome across all rows to
            # the signature matrix if the binary vector at that row equals one
            min_hash = 2**32
            for i in range(0, n_hashes):
                if binary_vectors[r, p] == 1 and hash_evaluations[i] < min_hash:
                    signature_matrix[i, p] = hash_evaluations[i]

    return signature_matrix


########### STEP 3 - LOCALITY-SENSITIVE HASHING ###########
def LSH(signature_matrix, threshold):
    n_hashes = len(signature_matrix)
    n_descriptions = len(signature_matrix[0])

    # Find the r and b that correspond most closely with the threshold
    best = 1
    for r in range(1, n_hashes + 1):
        for b in range(1, n_hashes + 1):
            if r*b == n_hashes:
                approx = (1/b) ** (1/r)
                if abs(approx - threshold) < abs(best - threshold):
                    best = approx
                    r_best = r
                    b_best = b


    # Perform the locality-sensitive hashing
    hashes = np.zeros((b_best, n_descriptions))
    candidates = np.zeros((n_descriptions, n_descriptions))
    for b in range(0, b_best):
        # For every column, aggregate the numbers within the band and hash them
        first_row = r_best*b
        last_row = first_row + r_best

        for p in range(0, n_descriptions):
            band_b = signature_matrix[first_row:last_row, p]
            band_value = ""
            for element in range(0, len(band_b)):
                if element > 0 and band_b[element] < 0:
                    band_b[element] = 0
                band_value += band_b[element].astype(str)


            band_value = int(band_value)
            hashes[b, p] = band_value % 2**32

        # Assign a product description p to a bucket if it is still empty
        # If there are multiple product descriptions within one bucket, they are a candidate pair
        buckets = dict()
        for p in range(0, n_descriptions):
            if hashes[b, p] in buckets:
                for i in buckets[hashes[b, p]]:
                    candidates[i, p] = 1
                    candidates[p, i] = 1
            else:
                buckets[hashes[b, p]] = [p]

    return candidates


################################### CLUSTERING ###################################
def MSM(LSH_candidates, sample_titles, sample_KVPs, sample_shops, sample_brands, sample_duplicates, threshold):
    # Initialize the dissimilarity matrix
    n_descriptions = len(sample_titles)
    dissimilarity = np.zeros((n_descriptions, n_descriptions))

    # Check every candidate pair
    for i in range(0, len(LSH_candidates)):
        #print(i / len(LSH_candidates) * 100, 'percent')
        id1 = LSH_candidates[i][0]
        id2 = LSH_candidates[i][1]

        # If two product descriptions are from the same shop, they are surely not duplicates
        if sample_shops[id1] == sample_shops[id2]:
            dissimilarity[id1, id2] = sys.maxsize
            dissimilarity[id2, id1] = sys.maxsize
        # If two products are from different brands, they are surely not duplicates
        elif sample_brands[id1] is not None and sample_brands[id2] is not None and sample_brands[id1] == sample_brands[
            id2]:
            dissimilarity[id1, id2] = sys.maxsize
            dissimilarity[id2, id1] = sys.maxsize
        # Otherwise, compute the dissimilarities based on three components:
        # avgSim, mwPerc and titleSim as in Hartveld et al. (2018)
        else:
            # Firstly, compute the average similarities of the key-value pairs
            KVP_id1 = list(sample_KVPs[id1].items())
            n_KVP_id1 = len(KVP_id1)
            KVP_id2 = list(sample_KVPs[id2].items())
            n_KVP_id2 = len(KVP_id2)

            sim = 0
            avgSim = 0
            m = 0  # Number of matches
            w = 0  # Weight of the matches
            nmk_i = copy.deepcopy(KVP_id1)
            nmk_j = copy.deepcopy(KVP_id2)

            # First check the similarity of the keys of the key-value pairs
            for r in range(0, n_KVP_id1):
                for q in range(0, n_KVP_id2):
                    key_1k = KVP_id1[r][0]
                    key_2l = KVP_id2[q][0]
                    s = SequenceMatcher(None, key_1k, key_2l)
                    keySim = s.ratio()

                    # If the similarity of the keys is sufficiently high
                    if keySim > 0.8:
                        # Update the weights
                        value_1k = KVP_id1[r][1]
                        value_2l = KVP_id2[q][1]
                        s = SequenceMatcher(None, value_1k, value_2l)
                        valueSim = s.ratio()
                        weight = keySim
                        sim = sim + weight * valueSim
                        m = m + 1
                        w = w + weight

                        # Remove the key-value pair from the list
                        nmk_i[r] = None
                        nmk_j[q] = None

            nmk_i = list(filter((None).__ne__, nmk_i))
            nmk_j = list(filter((None).__ne__, nmk_j))

            if w > 0:
                avgSim = sim / w

            # Secondly, compute the percentage of common model words in the remaining key-value pairs
            model_words_KVPs_i = []
            model_words_KVPs_j = []
            regex_title = re.compile(r'([a-zA-Z0-9]*(([0-9]+[ˆ0-9, ]+)|([ˆ0-9, ]+[0-9]+))[a-zA-Z0-9]*)')
            for m in range(0, len(nmk_i)):
                value = nmk_i[m][1]
                model_words_value = regex_title.findall(value)

                n_mw_value = len(model_words_value)
                if n_mw_value > 0:
                    for l in range(0, n_mw_value):
                        if model_words_value[l][0] not in model_words_KVPs_i:
                            model_words_KVPs_i.append(model_words_value[l][0])

            for m in range(0, len(nmk_j)):
                value = nmk_j[m][1]
                model_words_value = regex_title.findall(value)

                n_mw_value = len(model_words_value)
                if n_mw_value > 0:
                    for l in range(0, n_mw_value):
                        if model_words_value[l][0] not in model_words_KVPs_j:
                            model_words_KVPs_j.append(model_words_value[l][0])

            intersection = set(model_words_KVPs_i).intersection(model_words_KVPs_j)
            union = set(model_words_KVPs_i).union(model_words_KVPs_j)
            percentage = len(intersection) / len(union)

            # Thirdly, compute the similarities of the titles
            s = SequenceMatcher(None, sample_titles[id1], sample_titles[id2])
            titleSim = s.ratio()

            # Combine the three metrics
            minFeatures = min(n_KVP_id1, n_KVP_id2)
            if titleSim < 0.7:
                theta1 = m / minFeatures
                theta2 = 1 - theta1
                hSim = theta1 * avgSim + theta2 * percentage
            else:
                mu = 0.650
                theta1 = (1 - mu) * (m / minFeatures)
                theta2 = 1 - mu - theta1
                hSim = theta1 * avgSim + theta2 * percentage + mu * titleSim

            dissimilarity[id1, id2] = 1 - hSim
            dissimilarity[id2, id1] = 1 - hSim

    # Perform the hierarchical clustering
    linkage = hierarchy.linkage(dissimilarity, method="single")
    clusters = hierarchy.fcluster(linkage, threshold, criterion="distance")

    print(max(clusters), 'clusters were formed')

    # Evaluate the performance
    TP = 0
    FN = 0
    FP = 0
    for i in range(0, n_descriptions):
        for j in range(0, n_descriptions):
            if sample_duplicates[i, j] == 1:
                if clusters[i] == clusters[j]:
                    TP += 1
                else:
                    FN += 1
            else:
                if clusters[i] == clusters[j]:
                    FP += 1

    F1_score = TP / (TP + 0.5 * (FP + FN))

    return F1_score


################################### PERFORMANCE EVALUATION ###################################
def bootstrap(all_description_ids, all_titles, all_KVPs, all_shops, all_brands):
    # Generate the indices of the product descriptions to be included in the bootstrap sample
    n_descriptions = len(all_description_ids)
    random.seed(1)
    bootstrap_indices = np.random.randint(0, n_descriptions - 1, n_descriptions)

    # Retrieve the corresponding model ID, title and key-value pairs
    sample_model_ids = []
    sample_titles = []
    sample_KVPs = []
    sample_shops = []
    sample_brands = []
    for i in range(0, n_descriptions):
        sample_model_ids.append(all_description_ids[bootstrap_indices[i]])
        sample_titles.append(all_titles[bootstrap_indices[i]])
        sample_KVPs.append(all_KVPs[bootstrap_indices[i]])
        sample_shops.append(all_shops[bootstrap_indices[i]])
        sample_brands.append(all_brands[bootstrap_indices[i]])

    # Construct a matrix of duplicates, where element [i,j] = 1 if product ID i matches product ID j
    sample_duplicates = np.zeros((n_descriptions, n_descriptions))
    for i in range(0, n_descriptions):
        for j in range(0, n_descriptions):
            if i != j and sample_model_ids[i] == sample_model_ids[j]:
                sample_duplicates[i, j] = 1

    return sample_model_ids, sample_titles, sample_KVPs, sample_shops, sample_brands, sample_duplicates


def main(all_data):
    # Retrieve the model ID, title and key-value pairs of each unique product description
    description_ids = []
    titles = []
    KVPs = []
    shops = []
    for i in data:
        for j in data[i]:
            modelID = j['modelID']
            title = j['title']
            KVP = j['featuresMap']
            shop = j['shop']
            description_ids.append(modelID)
            titles.append(title)
            KVPs.append(KVP)
            shops.append(shop)

    n_descriptions = len(description_ids)

    brands = [None] * n_descriptions
    for i in range(0, n_descriptions):
        KVP_i = list(KVPs[i].items())
        n_KVP_i = len(KVP_i)
        for k in range(0, n_KVP_i):
            if KVP_i[k][0] == "Brand":
                brand_i = KVP_i[k][1]
                brands[i] = brand_i

    percentages = range(5, 30, 5)
    thresholds = np.zeros((len(percentages), 1))
    for t in range(0, len(percentages)):
        thresholds[t, 0] = percentages[t]/100

    n_thresholds = len(percentages)

    # Create an array to store the 5 evaluation metrics (fraction of comparisons, PC, PQ, F1* and later F1) for each bootstrap
    n_bootstraps = 5
    statistics_per_threshold = np.zeros((10, n_thresholds))
    for t in range(0, n_thresholds):
        threshold = thresholds[t, 0]
        statistics_per_boostrap = np.zeros((10, n_bootstraps))
        for i in range(0, n_bootstraps):
            print('Evaluating threshold', threshold, 'for bootstrap', i)
            # Generate the bootstrap sample
            bootstrap_description_ids, bootstrap_titles, bootstrap_KVPs, bootstrap_shops, bootstrap_brands, bootstrap_duplicates = bootstrap(description_ids, titles, KVPs, shops, brands)
            # Obtain the binary vectors for the bootstrap sample
            bootstrap_vectors_Hartveld = binary_vectors_Hartveld(bootstrap_titles, bootstrap_KVPs)
            bootstrap_vectors_Rigutto0 = binary_vectors_Rigutto(bootstrap_titles, bootstrap_KVPs)
            bootstrap_vectors_Rigutto = np.vstack((bootstrap_vectors_Rigutto0, bootstrap_vectors_Hartveld))
            # Construct the corresponding minhash matrix
            bootstrap_minhash_Hartveld = min_hash(bootstrap_vectors_Hartveld)
            bootstrap_minhash_Rigutto = min_hash(bootstrap_vectors_Rigutto)
            # Perform LSH
            bootstrap_candidates_Hartveld = LSH(bootstrap_minhash_Hartveld, threshold)
            bootstrap_candidates_Rigutto = LSH(bootstrap_minhash_Rigutto, threshold)

            # Retrieve the candidate pairs from the LSH candidates matrix
            candidate_pairs_Hartveld = []
            candidate_pairs_Rigutto = []
            for c in range(0, n_descriptions):
                for d in range(0, n_descriptions):
                    if bootstrap_candidates_Hartveld[c, d] == 1 and [d, c] not in candidate_pairs_Hartveld:
                        candidate_pairs_Hartveld.append([c, d])
                    if bootstrap_candidates_Rigutto[c, d] == 1 and [d, c] not in candidate_pairs_Rigutto:
                        candidate_pairs_Rigutto.append([c, d])

            if i == 4:
                F1_clustering_Hartveld = MSM(candidate_pairs_Hartveld,  bootstrap_titles, bootstrap_KVPs, bootstrap_shops, bootstrap_brands, bootstrap_duplicates, threshold)
                F1_clustering_Rigutto = MSM(candidate_pairs_Rigutto,  bootstrap_titles, bootstrap_KVPs, bootstrap_shops, bootstrap_brands, bootstrap_duplicates, threshold)

            # Compute fraction of comparisons, PC, PQ and F1* for each bootstrap
            total_comparisons = 1624*1623/2
            total_duplicates = bootstrap_duplicates.sum()/2
            n_comparisons_Hartveld = bootstrap_candidates_Hartveld.sum()/2
            n_comparisons_Rigutto = bootstrap_candidates_Rigutto.sum()/2
            frac_comp_Hartveld = n_comparisons_Hartveld/total_comparisons
            frac_comp_Rigutto = n_comparisons_Rigutto/total_comparisons

            n_correct_Hartveld = 0
            n_correct_Rigutto = 0
            for j in range(0, n_descriptions):
                for k in range(0, n_descriptions):
                    if bootstrap_candidates_Hartveld[j, k] == 1 and bootstrap_duplicates[j, k] == 1:
                        n_correct_Hartveld += 1
                    if bootstrap_candidates_Rigutto[j, k] == 1 and bootstrap_duplicates[j, k] == 1:
                        n_correct_Rigutto += 1

            PQ_Hartveld = (n_correct_Hartveld/2)/n_comparisons_Hartveld
            PQ_Rigutto = (n_correct_Rigutto/2)/n_comparisons_Rigutto
            PC_Hartveld = (n_correct_Hartveld/2)/total_duplicates
            PC_Rigutto = (n_correct_Rigutto/2)/total_duplicates
            F1_Hartveld = 2*PQ_Hartveld*PC_Hartveld/(PQ_Hartveld+PC_Hartveld)
            F1_Rigutto = 2*PQ_Rigutto*PC_Rigutto/(PC_Rigutto+PC_Rigutto)

            # Save the metrics into the statistics matrix
            statistics_per_boostrap[0, i] = frac_comp_Hartveld
            statistics_per_boostrap[1, i] = PQ_Hartveld
            statistics_per_boostrap[2, i] = PC_Hartveld
            statistics_per_boostrap[3, i] = F1_Hartveld
            statistics_per_boostrap[4, i] = 0
            statistics_per_boostrap[5, i] = frac_comp_Rigutto
            statistics_per_boostrap[6, i] = PQ_Rigutto
            statistics_per_boostrap[7, i] = PC_Rigutto
            statistics_per_boostrap[8, i] = F1_Rigutto
            statistics_per_boostrap[9, i] = 0

        for i in range(0, 10):
            statistics_per_threshold[i, t] = 0
            for j in range(0, n_bootstraps):
                statistics_per_threshold[i, t] += (statistics_per_boostrap[i, j])/n_bootstraps

        statistics_per_threshold[4, t] = F1_clustering_Hartveld
        statistics_per_threshold[9, t] = F1_clustering_Rigutto
        print(statistics_per_threshold)

    df = pd.DataFrame(statistics_per_threshold)
    writer = pd.ExcelWriter('Statistics.xlsx', engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Welcome', index=False)
    writer.save()


# Import the product descriptions BASED ON THE CLEANED DATASET
f = open('TVs-all-merged3.json')
data = json.load(f)
main(data)