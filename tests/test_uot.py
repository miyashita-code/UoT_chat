import asyncio
import pytest
import logging
import time
from typing import List, Tuple
import sys
import os
import re
import concurrent.futures

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from uot_modules.uot import UoT
from uot_modules.item import Item
from uot_modules.uot_node import UoTNode

# Configure logging
logging.basicConfig(filename='test_uot_detailed.log', level=logging.DEBUG, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

def setup_uot() -> UoT:
    """Initialize a UoT object with predefined items."""
    item_names = ["腹痛がある", "不明(その他)", "靴下を探している", "服を探している", "不安", 
                  "怒り", "デイサービスの準備", "歯磨きをする", "ごみを捨てに行く", "外に行きたくない"]
    items = [Item(name, "", 1/len(item_names)) for name in item_names]

    return UoT(initial_items=items, n_extend_layers=3, n_question_candidates=4, n_max_pruning=3, lambda_=2, is_debug=False, unknown_reward_prob_ratio=0.1)

def blocking_input(prompt: str) -> str:
    return input(prompt)

async def async_input(prompt: str) -> str:
    loop = asyncio.get_running_loop()
    with concurrent.futures.ThreadPoolExecutor() as pool:
        return await loop.run_in_executor(pool, blocking_input, prompt)

async def get_user_input() -> Tuple[str, float]:
    """Get user input asynchronously."""
    while True:
        input_str = await async_input("回答と確信度を入力してください (yes/no, 確信度 0 < p < 1): ")
        match = re.match(r"^(yes|no),\s*(0\.\d+|1\.0*)$", input_str.strip(), re.IGNORECASE)
        if match:
            a_str, p_y_str = match.groups()
            p_y = float(p_y_str)
            return a_str.lower(), p_y
        print("無効な入力です。'yes'または'no'、および0と1の間の確信度をカンマで区切って入力してください。")


'''
@pytest.mark.asyncio
async def test_uot_initialization():
    """Test the initialization of the UoT object."""
    uot = setup_uot()
    logger.info(f"UoT initialized with {len(uot.root.items)} items")
    logger.debug(f"Initial items: {[item.get_name() for item in uot.root.items]}")

    print(f"initial items : {[item.get_name() for item in uot.root.items]}")
    
    assert isinstance(uot.root, UoTNode)
    assert len(uot.root.items) == 10
    assert all(isinstance(item, Item) for item in uot.root.items)
    assert sum(item.p_s for item in uot.root.items) == pytest.approx(1.0)

@pytest.mark.asyncio
async def test_uot_extend():
    """Test the tree extension functionality."""
    uot = setup_uot()
    
    # 最初の拡張（1層目）を待機
    await uot.extend()
    
    print("First layer extension complete")
    print(f"Current tree depth: {uot.root.current_extended_depth}")
    
    # バックグラウンド拡張の開始を確認
    assert uot.extension_task is not None
    
    # バックグラウンド拡張の進行を少し待つ
    await asyncio.sleep(20)
    
    print(f"Tree depth after waiting: {uot.root.current_extended_depth}")
    uot.print_detail_of_tree_DFS()
    
    # 拡張が進行していることを確認
    assert uot.root.current_extended_depth > 0
    assert len(uot.root.children) > 0
    
    # バックグラウンド拡張の完了を待たずにテスト終了
    print("Test completed without waiting for full extension")



# failed
@pytest.mark.asyncio
async def test_uot_get_question():
    """Test the question generation functionality."""
    uot = setup_uot()
    question = await uot.get_question()
    
    logger.info(f"Generated question: {question}")
    
    print(f"generated question: {question}")
    assert isinstance(question, str)
    assert len(question) > 0
    assert uot.extension_task is not None

@pytest.mark.asyncio
async def test_uot_answer():
    """Test the answer processing functionality."""

    print("\n****** setup ******")
    uot = setup_uot()

    print("****** extend ******")
    await uot.extend()
    initial_root = uot.root
    
    print("****** print_detail_of_tree_DFS ******")
    uot.print_detail_of_tree_DFS()
    
    print("****** answer ******")
    await uot.answer(0.7)
    
    logger.info("Answer processed")
    logger.debug(f"New root depth: {uot.root.depth}")

    print(f"new root depth: {uot.root.depth}")
    print(f"new root current_extended_depth: {uot.root.current_extended_depth}")

    print("****** print_detail_of_tree_DFS ******")
    uot.print_detail_of_tree_DFS()
    
    assert uot.root != initial_root
    assert uot.root.parent is None
    assert uot.root.depth == 0
    assert uot.extension_task is not None
'''

@pytest.mark.asyncio
async def test_uot_full_cycle():
    """Comprehensive test of the UoT algorithm simulating real-world usage."""
    uot = setup_uot()
    start_time = time.time()
    
    for i in range(5):
        print(f"\n===== ループ {i+1} 開始 =====")
        logger.info(f"Starting loop {i+1}")
        loop_start_time = time.time()

        print("****** print_detail_of_tree_DFS ******")
        uot.print_detail_of_tree_DFS()
        
        if uot.root.current_extended_depth == 0:
            # Initial question generation only once
            print(f"層 {uot.root.current_extended_depth + 1} の生成を開始")
            question_gen_start = time.time()
            await uot.extend()
            question_gen_time = time.time() - question_gen_start
            logger.info(f"Initial question generation time: {question_gen_time:.4f} seconds")
            print(f"層 {uot.root.current_extended_depth} の生成完了 (所要時間: {question_gen_time:.4f}秒)")
        
        # Best question selection
        print("最適な質問の選択中...")
        question = await uot.get_question()
        logger.info(f"Selected question: {question}")
        print(f"選択された質問: {question}")
        
        # User input
        a_str, p_y = await get_user_input()
        logger.info(f"User answer: {a_str}, p_y: {p_y}")
        
        # Tree update
        print("ツリーの更新中...")
        update_start = time.time()
        await uot.answer(p_y)
        update_time = time.time() - update_start
        logger.info(f"Tree update time: {update_time:.4f} seconds")
        print(f"ツリー更新完了 (所要時間: {update_time:.4f}秒)")
        
        # Check if another extension is needed
        if uot.root.children:
            print(f"層 {uot.root.current_extended_depth + 1} の生成を開始")
            question_gen_start = time.time()
            await uot.extend()
            question_gen_time = time.time() - question_gen_start
            logger.info(f"Question generation time: {question_gen_time:.4f} seconds")
            print(f"層 {uot.root.current_extended_depth} の生成完了 (所要時間: {question_gen_time:.4f}秒)")
        
        # Log probability distribution
        print("現在の確率分布:")
        probabilities = [f"{item.get_name()}: {item.p_s:.4f}" for item in uot.root.items]
        for prob in probabilities:
            print(prob)
        logger.info(f"Updated probability distribution:\n" + "\n".join(probabilities))
        
        loop_time = time.time() - loop_start_time
        logger.info(f"Loop {i+1} completed (duration: {loop_time:.4f} seconds)")
        print(f"===== ループ {i+1} 終了 (所要時間: {loop_time:.4f}秒) =====")
    
    total_time = time.time() - start_time
    logger.info(f"Total execution time: {total_time:.4f} seconds")
    print(f"\n全体の実行時間: {total_time:.4f}秒")
    
    # Assertions
    assert sum(item.p_s for item in uot.root.items) == pytest.approx(1.0)
    assert all(0 <= item.p_s <= 1 for item in uot.root.items)
    assert len(uot.root.children) <= uot.n_max_pruning * 2

'''
@pytest.mark.asyncio
async def test_uot_edge_cases():
    """Test edge cases and boundary conditions."""
    # Case 1: Single item
    single_item = [Item("Single Item", "", 1.0)]
    uot_single = UoT(initial_items=single_item, n_extend_layers=3, n_question_candidates=4, n_max_pruning=2, lambda_=0.1, is_debug=True)
    question = await uot_single.get_question()
    logger.info(f"Single item test - Question: {question}")
    assert question is None
    
    # Case 2: Two items
    two_items = [Item("Item 1", "", 0.5), Item("Item 2", "", 0.5)]
    uot_two = UoT(initial_items=two_items, n_extend_layers=3, n_question_candidates=4, n_max_pruning=2, lambda_=0.1, is_debug=True)
    question = await uot_two.get_question()
    logger.info(f"Two items test - Question: {question}")
    assert isinstance(question, str)
    
    # Case 3: Maximum depth
    uot = setup_uot()
    for _ in range(uot.n_extend_layers):
        await uot.extend()
    logger.info(f"Maximum depth test - Current depth: {uot.root.current_extended_depth}")
    assert uot.root.current_extended_depth == uot.n_extend_layers - 1

@pytest.mark.asyncio
async def test_uot_long_running():
    """Test long-running execution of the UoT algorithm."""
    uot = setup_uot()
    start_time = time.time()
    
    for i in range(50):  # Increased number of iterations
        print(f"\n===== 長時間実行テスト - イテレーション {i+1} =====")
        logger.info(f"Long-running test - Iteration {i+1}")
        question = await uot.get_question()
        print(f"質問: {question}")
        a_str, p_y = await get_user_input()
        await uot.answer(p_y)
        print("現在の確率分布:")
        for item in uot.root.items:
            print(f"{item.get_name()}: {item.p_s:.4f}")
    
    total_time = time.time() - start_time
    logger.info(f"Long-running test completed in {total_time:.4f} seconds")
    print(f"\n長時間実行テスト完了 (所要時間: {total_time:.4f}秒)")
    
    # Check final state
    assert sum(item.p_s for item in uot.root.items) == pytest.approx(1.0)
    assert max(item.p_s for item in uot.root.items) > 0.5  # Expect some convergence

@pytest.mark.asyncio
async def test_uot_extension_complete():
    """Test that extension stops at maximum depth and restarts after answer."""
    uot = setup_uot()
    
    # Extend to maximum depth
    for _ in range(uot.n_extend_layers):
        await uot.extend()
    
    assert uot.extension_complete == True
    
    # Try to extend again, should not increase depth
    initial_depth = uot.root.current_extended_depth
    await uot.extend()
    assert uot.root.current_extended_depth == initial_depth
    
    # Answer and check if extension restarts
    await uot.answer(0.7)
    assert uot.extension_complete == False
    
    # Extend again, should increase depth
    await uot.extend()
    assert uot.root.current_extended_depth > initial_depth
'''
if __name__ == "__main__":
    pytest.main(["-vs", __file__])