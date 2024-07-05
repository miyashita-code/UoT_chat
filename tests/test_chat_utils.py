import pytest

import sys
sys.path.insert(0, './')  # プロジェクトのルートディレクトリをパスに追加
from uot_modules.chat_utils import generate_questions_and_estimate_probability, format_history, _simulate_and_estimate_chunk
from uot_modules.uot import Item

def test_generate_questions_and_estimate_probability():
   # サンプルデータの定義
   item_names = ["腹痛がある", "不明", "靴下を探している", "服を探している", "不安", "怒り", "デイサービスの準備", "歯磨きをする", "ごみを捨てに行く", "外に行きたくない"]
   items = [Item(name, "", 1/len(item_names)) for name in item_names]
   ques_num = 4
   history = [
       {"q": "お手伝いできますか?", "a": "yes"},
       {"q": "体調は悪くないですか?", "a": "yes"}
   ]
   additional_context = ["外出の準備をしているようです"]

   # generate_questions_and_estimate_probability()関数をテスト
   results = generate_questions_and_estimate_probability(items, ques_num, history, additional_context)

   print(f"results: {results}")

   # 結果の検証
   assert len(results) == ques_num
   for result in results:
       assert "question" in result
       assert "evaluated_items" in result
       assert len(result["evaluated_items"]) == len(items)
       for item in result["evaluated_items"]:
           assert "p_yes_given_item" in item
           assert "p_no_given_item" in item
           assert "name" in item
           assert "description" in item
           assert 0 <= item["p_yes_given_item"] <= 1
           assert 0 <= item["p_no_given_item"] <= 1

def test_format_history():
   # サンプルデータの定義
   history = [
       {"q": "お手伝いできますか?", "a": "yes"},
       {"q": "体調は悪くないですか?", "a": "yes"}
   ]

   # format_history()関数をテスト 
   history_str = format_history(history)
   assert history_str == "Question 1: お手伝いできますか? -> Answer 1 : yes\nQuestion 2: 体調は悪くないですか? -> Answer 2 : yes"

def test_simulate_and_estimate_chunk():
   # サンプルデータの定義
   item_names = ["腹痛がある", "不明", "靴下を探している", "服を探している", "不安"]
   items = [Item(name, "", 1/len(item_names)) for name in item_names]
   chunk = items[:5]
   question = "外出の準備をしていますか?"
   history = [
       {"q": "お手伝いできますか?", "a": "yes"},
       {"q": "体調は悪くないですか?", "a": "yes"}
   ]
   history_str = format_history(history)

   # _simulate_and_estimate_chunk()関数をテスト
   response = _simulate_and_estimate_chunk(chunk, question, history_str)
   assert "items" in response
   assert len(response["items"]) == len(chunk)

if __name__ == "__main__":
   pytest.main()