import asyncio
import concurrent.futures
from typing import List, Dict, Tuple, Optional, Union
import re
from uot_modules.item import Item
from uot_modules.uot_node import UoTNode

def blocking_input(prompt: str) -> str:
    return input(prompt)

async def async_input(prompt: str) -> str:
    loop = asyncio.get_running_loop()
    with concurrent.futures.ThreadPoolExecutor() as pool:
        return await loop.run_in_executor(pool, blocking_input, prompt)

async def get_user_input() -> Tuple[str, float]:
    while True:
        input_str = await async_input("回答と確信度を入力してください (yes/no, 確信度 0 < p < 1): ")
        match = re.match(r"^(yes|no),\s*(0\.\d+|1\.0*)$", input_str.strip(), re.IGNORECASE)
        if match:
            a_str, p_y_str = match.groups()
            p_y = float(p_y_str)
            return a_str.lower(), p_y
        print("無効な入力です。'yes'または'no'、および0と1の間の確信度をカンマで区切って入力してください。")

class UoT:
    def __init__(self, initial_items: List[Item], n_extend_layers: int,
                 n_question_candidates: int, n_max_pruning: int, lambda_: float,
                 is_debug: bool = False, unknown_reward_prob_ratio: float = 0.1):
        self.root = UoTNode("root", initial_items, is_debug=is_debug)
        self.n_extend_layers = n_extend_layers
        self.n_question_candidates = n_question_candidates
        self.n_max_pruning = n_max_pruning
        self.lambda_ = lambda_
        self.is_debug = is_debug
        self.extension_stop_flag = asyncio.Event()
        self.lock = asyncio.Lock()
        self.unknown_reward_prob_ratio = unknown_reward_prob_ratio
        self.best_question = None
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.extension_task = None

        self.root.configure_node(
            n_extend_layers=n_extend_layers,
            accumulation=True,
            expected_method='avg',
            n_max_pruning=n_max_pruning,
            lambda_=lambda_,
            n_question_candidates=n_question_candidates
        )
        
        self.debug_print(f"UoT initialized with {len(initial_items)} items")

    async def extend(self) -> None:
        self.debug_print("Starting tree extension")
        if not self.root.children:
            self.debug_print("Generating first layer")
            await self.extend_single_layer(self.root, 1)
            self.best_question = self.root.get_best_question()
            self.debug_print(f"First layer generated, best question: {self.best_question}")
        
        if not self.extension_task:
            self.extension_task = asyncio.create_task(self._background_extend())
            self.debug_print("Background extension task created")

    async def extend_single_layer(self, node: UoTNode, depth: int) -> None:
        self.debug_print(f"Extending layer at depth {depth}")
        if node.is_terminal or depth > self.n_extend_layers:
            self.debug_print(f"Reached terminal node or max depth at {depth}")
            return

        if not node.children:
            await node.generate_children()
            self.debug_print(f"Generated children for node at depth {depth}")
        else:
            extension_tasks = [self.extend_single_layer(child, depth+1) for child in node.children]
            await asyncio.gather(*extension_tasks)
        
        await node.calculate_rewards()
        node.prune_children()
        node.current_extended_depth = max(node.current_extended_depth, depth)
        self.debug_print(f"Layer extended, new depth: {node.current_extended_depth}")

    async def _background_extend(self) -> None:
        self.debug_print("Starting background extension")
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(self.executor, self._extend_in_thread)
        except Exception as e:
            self.debug_print(f"Background extension error: {e}")
        finally:
            self.extension_task = None
            self.debug_print("Background extension complete")

    def _extend_in_thread(self) -> None:
        current_depth = self.root.current_extended_depth + 1
        while current_depth < self.n_extend_layers and not self.extension_stop_flag.is_set():
            self.debug_print(f"Extending layer {current_depth} in background")
            asyncio.run(self.extend_single_layer(self.root, current_depth))
            new_best_question = self.root.get_best_question()
            if new_best_question != self.best_question:
                self.debug_print(f"Updating best question: {new_best_question}")
                self.best_question = new_best_question
            current_depth += 1

    async def get_question(self) -> Optional[str]:
        if not self.root.children:
            self.debug_print("No children, extending first layer")
            await self.extend()
        return self.best_question

    def stop_extension(self) -> None:
        self.debug_print("Stopping extension")
        self.extension_stop_flag.set()
        if self.extension_task:
            self.extension_task.cancel()

    async def answer(self, p_prime_y: float) -> None:
        self.debug_print(f"Processing answer with p_prime_y={p_prime_y}")
        
        self.stop_extension()
        
        async with self.lock:
            best_question = self.root.get_best_question()
            yes_node, no_node = self.root.get_yes_no_nodes(best_question)

            if p_prime_y == 0.5:
                self.root = await self.root.handle_equal_probability(yes_node, no_node, p_prime_y)
            else:
                self.root = yes_node if p_prime_y > 0.5 else no_node

            await self.root.update_probabilities(p_prime_y, is_root=True)
            self.root.parent = None
            self.root.p_reply = p_prime_y
            await self.root.recalculate_rewards()
            await self.root.elevate_layer()
            self._reward_unknown_prob()
        
        self.extension_stop_flag.clear()
        self.best_question = self.root.get_best_question()
        self.debug_print(f"New best question after answer: {self.best_question}")
        await self.extend()

    def _reward_unknown_prob(self):
        unknown_item = next((item for item in self.root.items if item.get_name() == "不明(その他)"), None)
        if unknown_item:
            max_prob = max(item.p_s for item in self.root.items)
            unknown_item.p_s += max_prob * self.unknown_reward_prob_ratio
        else:
            unknown_item = Item("不明(その他)", "その他のアイテム", 1/(len(self.root.items)+1))
            self.root.items.append(unknown_item)
        
        # 正規化を確実に行う
        total_prob = sum(item.p_s for item in self.root.items)
        for item in self.root.items:
            item.p_s /= total_prob
        
        # 正規化後の確認
        self.debug_print("Probabilities after normalization:")
        for item in self.root.items:
            self.debug_print(f"{item.get_name()}: {item.p_s:.4f}")
        
        # 合計が1になることを確認
        total_after = sum(item.p_s for item in self.root.items)
        self.debug_print(f"Total probability after normalization: {total_after:.4f}")

    async def get_current_probabilities(self) -> List[Tuple[str, float]]:
        return [(item.get_name(), item.p_s) for item in self.root.items]

    async def run(self) -> None:
        while True:
            question = await self.get_question()
            if question is None:
                self.debug_print("No more questions available.")
                break
            self.debug_print(f"Question: {question}")
            
            a_str, p_y = await get_user_input()
            await self.answer(p_y if a_str == 'yes' else 1 - p_y)
            
            probabilities = await self.get_current_probabilities()
            self.debug_print("Current probabilities:")
            for item, prob in probabilities:
                self.debug_print(f"{item}: {prob:.4f}")
            
            if self.is_debug:
                self.display_results()
                self.save_results_to_file("uot_results.json")

    def analyze_node(self, node: UoTNode, indent: str = "") -> None:
        self.debug_print(f"{indent}Node Analysis:")
        self.debug_print(f"{indent}  Question: {node.question}")
        self.debug_print(f"{indent}  Depth: {node.depth}")
        self.debug_print(f"{indent}  Is Terminal: {node.is_terminal}")
        self.debug_print(f"{indent}  Reward: {node.reward:.4f}")
        self.debug_print(f"{indent}  Information Gain: {node.information_gain:.4f}")
        self.debug_print(f"{indent}  Number of Items: {len(node.items)}")
        self.debug_print(f"{indent}  Number of Children: {len(node.children)}")
        self.debug_print(f"{indent}  API Stats:")
        for line in node.get_api_stats().split('\n'):
            self.debug_print(f"{indent}    {line}")
        
        if node.children:
            self.debug_print(f"{indent}  Child Nodes:")
            for child in node.children:
                self.analyze_node(child, indent + "    ")

    def store_results(self) -> Dict:
        return {
            "n_extend_layers": self.n_extend_layers,
            "n_question_candidates": self.n_question_candidates,
            "n_max_pruning": self.n_max_pruning,
            "lambda": self.lambda_,
            "root": self.root.store_results()
        }

    def save_results_to_file(self, filename: str) -> None:
        import json
        results = self.store_results()
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        self.debug_print(f"Results saved to {filename}")

    def display_results(self) -> None:
        self.debug_print("UoT Results:")
        self.debug_print(f"n_extend_layers: {self.n_extend_layers}")
        self.debug_print(f"n_question_candidates: {self.n_question_candidates}")
        self.debug_print(f"n_max_pruning: {self.n_max_pruning}")
        self.debug_print(f"lambda: {self.lambda_}")
        self.debug_print("Root Node:")
        self.analyze_node(self.root, "  ")

    @classmethod
    def load_results_from_file(cls, filename: str) -> 'UoT':
        import json
        with open(filename, 'r') as f:
            data = json.load(f)
        uot = cls(
            initial_items=[],  # This will be overwritten
            n_extend_layers=data['n_extend_layers'],
            n_question_candidates=data['n_question_candidates'],
            n_max_pruning=data['n_max_pruning'],
            lambda_=data['lambda'],
            is_debug=True
        )
        uot.root = UoTNode.create_from_data(data['root'])
        return uot

    def print_detail_of_tree_DFS(self) -> None:
        def _dfs_print(node: UoTNode, depth: int = 0, yes_no: str = "") -> None:
            indent = "  " * depth
            self.debug_print(f"{indent}{yes_no}{node.print_node_info(depth)}")
            print(f"{indent}{yes_no}{node.print_node_info(depth)}")
            
            if node.children:
                self.debug_print(f"{indent}子ノード:")
                print(f"{indent}子ノード:")
                for i, child in enumerate(node.children):
                    _dfs_print(child, depth + 1, f"{'Yes' if child.reply else 'No'}: ")

        self.debug_print("UoT木の詳細情報(DFS順):")
        print("UoT木の詳細情報(DFS順):")
        _dfs_print(self.root)

        self.print_overview_of_tree()

    def print_overview_of_tree(self) -> None:
        self.debug_print("UoT木の概要:")
        print("UoT木の概要:")

        depth_dict = {}
        def _dfs_count(node: UoTNode, depth: int = 0) -> None:
            if depth not in depth_dict:
                depth_dict[depth] = 0
            depth_dict[depth] += 1
            for child in node.children:
                _dfs_count(child, depth + 1)
        _dfs_count(self.root)
        self.debug_print(str(depth_dict))
        print(str(depth_dict))

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        self.stop_extension()
        self.executor.shutdown(wait=True)

    def debug_print(self, message: str) -> None:
        if self.is_debug:
            print(message)

# 使用例
async def main():
    async with UoT(initial_items, n_extend_layers, n_question_candidates, n_max_pruning, lambda_, is_debug=True) as uot:
        await uot.run()

if __name__ == "__main__":
    asyncio.run(main())