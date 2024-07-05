import sys
import json
import pytest
from pprint import pprint
from collections import defaultdict, Counter
import concurrent.futures
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib
import time

sys.path.insert(0, './')  # プロジェクトのルートディレクトリをパスに追加
from uot_modules.llm_utils import get_response_util, pydantic_to_dict

# サンプルデータの定義
question_sample = "外出の準備ですか?"
chunk_sample = ["腹痛がある", "不明", "靴下を探している", "服を探している", "不安", "怒り", "デイサービスの準備", "歯磨きをする", "ごみを捨てに行く", "外に行きたくない"]
history_str = """
Q1: お手伝いできますか? -> A: 'yes'
Q2: 体調は悪くないですか? -> A: 'yes'
"""

# 期待されるレスポンス
expected_response = {
    "question": question_sample,
    "items": [
        {"name": "腹痛がある", "y_prob" : 0.2},
        {"name": "不明", "y_prob" : 0.5},
        {"name": "靴下を探している", "y_prob" : 0.9},
        {"name": "服を探している", "y_prob" : 0.9},
        {"name": "不安", "y_prob" : 0.4},
        {"name": "怒り", "y_prob" : 0.4},
        {"name": "デイサービスの準備", "y_prob" : 0.9},
        {"name": "歯磨きをする", "y_prob" : 0.6},
        {"name": "ごみを捨てに行く", "y_prob" : 0.6},
        {"name": "外に行きたくない", "y_prob" : 0.1}
    ]
}

def remove_thought_field(data):
    if isinstance(data, dict):
        if 'thought' in data:
            del data['thought']
        for key, value in data.items():
            remove_thought_field(value)
    elif isinstance(data, list):
        for item in data:
            remove_thought_field(item)
    return data

def evaluate_response(result_dict, expected_dict):
    correct_labels = []
    predicted_labels = []
    results = []

    for expected_item, result_item in zip(expected_dict['items'], result_dict['items']):
        correct_labels.append(expected_item['name'])
        if 'name' in result_item and 'y_prob' in result_item:
            predicted_labels.append(result_item['name'])
            results.append((result_item['name'], expected_item['name'], result_item['y_prob'], expected_item['y_prob']))
        else:
            print(f"Missing label or probabilities in result_item: {result_item}")

    return correct_labels, predicted_labels, results

def classify_chunk():
    classify_chunk_chain = get_response_util("evaluate_probabilities_of_chunk")
    try:
        start_time = time.time()
        result = classify_chunk_chain.invoke({"item_name_list": chunk_sample, "question": question_sample, "history": history_str})
        end_time = time.time()
        result_dict = pydantic_to_dict(result)
        return result_dict, end_time - start_time
    except Exception as e:
        print(f"Parse error: {e}")
        return {"error": str(e)}, 0

def test_classify_chunk():
    label_counts = defaultdict(lambda: defaultdict(int))
    parse_errors = 0
    all_correct_labels = []
    all_predicted_labels = []
    all_results = []
    api_call_times = []
    
    start_time = time.time()
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(classify_chunk) for _ in range(1)]
        
        for future in concurrent.futures.as_completed(futures):
            result_dict, api_call_time = future.result()
            api_call_times.append(api_call_time)
            if "error" in result_dict:
                parse_errors += 1
            else:
                correct_labels, predicted_labels, results = evaluate_response(result_dict, expected_response)
                all_correct_labels.extend(correct_labels)
                all_predicted_labels.extend(predicted_labels)
                all_results.extend(results)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print_results(all_results, api_call_times, total_time)

def print_results(results, api_call_times, total_time):
    print("\nResults:")
    for i, result in enumerate(results):
        print(f"This is result: {result}")
        correct = float(expected_response['items'][i]['y_prob'])
        probability = float(result[2])
        if abs(probability - correct) <= 0.15:
            print(f"Name: {result[0]}, Correct Probability: {correct}, Predicted Probability: {probability}")
        else:
            print(f"Name: {result[0]}, Correct Probability: {correct}, Predicted Probability: {probability} (Out of range)")
    
    print(f"\nTotal test execution time: {total_time:.2f} seconds")
    print(f"Number of API calls: {len(api_call_times)}")
    print(f"Average API call time: {sum(api_call_times) / len(api_call_times):.2f} seconds")
    print(f"Minimum API call time: {min(api_call_times):.2f} seconds")
    print(f"Maximum API call time: {max(api_call_times):.2f} seconds")

if __name__ == "__main__":
    test_classify_chunk()