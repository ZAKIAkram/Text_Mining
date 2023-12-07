import scipy.io as sio
import os
import numpy as np
import json

def load_data(filename):
    data = sio.loadmat(filename)
    return data

def to_json(filename):
    """
    function that converts MED.REL file to json format
    """
    with open(filename, "r") as f:
        lines = f.readlines()

    json_dict = {}
    for line in lines:
        line = line.split()
        if line[0] not in json_dict:
            json_dict[line[0]] = []
        json_dict[line[0]].append(int(line[2]))
    # Write the entire dictionary to a single line with indentation for better readability
    with open(filename[:-4] + ".json", "w") as f:
        json.dump(json_dict, f, indent=4)
    return json_dict

def load_json(filename):
    """
    function that loads a json file
    """
    with open(filename, "r") as f:
        json_dict = json.load(f)
    return json_dict

def vector_space_cos(a_j, q):
    return q.T @ a_j / (np.linalg.norm(q) * np.linalg.norm(a_j))

def retrieve_docs(query_distances, tol):
    """
    retrieve relevant docs for a given query using the distance matrix
    """
    retrieved_docs = []
    for i in range(len(query_distances)):
        if query_distances[i] > tol:
            retrieved_docs.append(i + 1)
    return retrieved_docs


def precision_recall(json_dict, query_id, retrieved_docs):
    """
    function that computes the precision and recall for a given query
    """
    relevant_docs = json_dict[query_id]
    relevant_retrieved_docs = set(retrieved_docs).intersection(set(relevant_docs))
    D_r = len(relevant_retrieved_docs)
    D_t = len(retrieved_docs)
    N_r = len(relevant_docs)
    try:
        precision = D_r / D_t
        recall = D_r / N_r
    except ZeroDivisionError:
        return 1, 0

    return precision, recall



def main():
    to_json(r"C:\Users\akram\Desktop\etudes_liu\Matrix_AI\git_project\text_mining\data\med\MED.REL")

if __name__ == "__main__":
    main()