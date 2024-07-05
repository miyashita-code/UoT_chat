import asyncio
import numpy as np
import time
from typing import List, Dict, Tuple, Optional, Union
from uot_modules.chat_utils import generate_questions_and_estimate_probability
from uot_modules.item import Item

class UoTNode:
    api_stats = {
        "generate_questions": {"attempts": 0, "successes": 0, "total_time": 0},
        "estimate_probability": {"attempts": 0, "successes": 0, "total_time": 0},
        "update_probabilities": {"attempts": 0, "successes": 0, "total_time": 0},
        "custom_bayesian_update": {"attempts": 0, "successes": 0, "total_time": 0}
    }

    def __init__(self, question: str, items: List[Item], parent: Optional['UoTNode'] = None, 
             reply: Optional[bool] = None, p_reply: Optional[float] = None, 
             history: Optional[List[Dict[str, Union[str, bool, float]]]] = None,
             is_debug: bool = False, is_terminal: bool = False, max_item_size_at_once: int = 3,
             generated_info: Optional[List[Dict]] = None):
        self.question = question
        self.items = items
        self.parent = parent
        self.reply = reply
        self.p_reply = p_reply
        self.history = history.copy() if history else []
        self.children = []
        self.generated_info = generated_info
        self.depth = self.parent.depth + 1 if self.parent else 0
        self.n_extend_layers = -1
        self.expected_method = 'avg'
        self.accumulation = True
        self.information_gain = 0.0
        self.reward = 0.0
        self.best_question = None
        self.n_max_pruning = 0
        self.n_question_candidates = 0
        self.lambda_ = 0.0
        self.is_debug = is_debug
        self.is_terminal = is_terminal
        self.current_extended_depth = 0
        self.max_item_size_at_once = max_item_size_at_once

        if question != "root":
            self.history.append({"q": question, "a": reply, "p": p_reply})

        self.debug_print("__init__", f"Creating new node: question={self.question}, items={len(self.items)}, depth={self.depth}, is_terminal={self.is_terminal}")

    def debug_print(self, method: str, message: str) -> None:
        if self.is_debug:
            print(f"[DEBUG] UoTNode.{method}: {message}")

    @classmethod
    def log_api_call(cls, api_name: str, success: bool, duration: float) -> None:
        cls.api_stats[api_name]["attempts"] += 1
        if success:
            cls.api_stats[api_name]["successes"] += 1
        cls.api_stats[api_name]["total_time"] += duration

    @classmethod
    def get_api_stats(cls) -> str:
        stats = []
        for api_name, data in cls.api_stats.items():
            attempts = data["attempts"]
            successes = data["successes"]
            avg_time = data["total_time"] / attempts if attempts > 0 else 0
            stats.append(f"{api_name}: Attempts: {attempts}, Successes: {successes}, Avg Time: {avg_time:.4f}s")
        return "\n".join(stats)

    def configure_node(self, n_extend_layers: int, accumulation: bool, expected_method: str, n_max_pruning: int, lambda_: float, n_question_candidates: int) -> None:
        self.debug_print("configure_node", f"Configuring node: n_extend_layers={n_extend_layers}, accumulation={accumulation}, expected_method={expected_method}, n_max_pruning={n_max_pruning}, lambda_={lambda_}")
        
        self.n_extend_layers = n_extend_layers
        self.accumulation = accumulation
        self.expected_method = expected_method
        self.n_max_pruning = n_max_pruning
        self.n_question_candidates = n_question_candidates
        self.lambda_ = lambda_


    
    def _calculate_posterior_probabilities(self, item_probabilities: List[Dict], p_yes: float, p_no: float) -> Tuple[List[Item], List[Item], List[float], List[float]]:
        self.debug_print("_calculate_posterior_probabilities", f"Calculating posterior probabilities: p_yes={p_yes}, p_no={p_no}")

        omega_yes = []
        omega_no = []
        p_yes_given_items = []
        p_no_given_items = []
        for item_prob, item in zip(item_probabilities, self.items):
            p_item = item.p_s
            p_yes_given_item = item_prob["p_yes_given_item"]
            p_no_given_item = 1 - p_yes_given_item
            
            p_item_given_yes = p_yes_given_item * p_item / p_yes if p_yes > 0 else 0
            p_item_given_no = p_no_given_item * p_item / p_no if p_no > 0 else 0
            
            omega_yes.append(Item(item.get_name(), item.description, p_item_given_yes))
            omega_no.append(Item(item.get_name(), item.description, p_item_given_no))
            p_yes_given_items.append(p_yes_given_item)
            p_no_given_items.append(p_no_given_item)

        self.debug_print("_calculate_posterior_probabilities", f"Posterior probabilities calculated: omega_yes={len(omega_yes)}, omega_no={len(omega_no)}")
        
        return omega_yes, omega_no, p_yes_given_items, p_no_given_items

    async def extend_single_layer(self) -> None:
        print(f"Extending layer at depth {self.depth}")
        if self.is_terminal or self.depth >= self.n_extend_layers - 1:
            print(f"Reached terminal node or max depth at {self.depth}")
            return

        if not self.children:
            await self.generate_children()
            print(f"Generated {len(self.children)} children at depth {self.depth}")
        else:
            extension_tasks = [child.extend_single_layer() for child in self.children]
            await asyncio.gather(*extension_tasks)
        
        await self.calculate_rewards()
        self.prune_children()
        self.current_extended_depth += 1
        print(f"Layer extended, new depth: {self.current_extended_depth}")

    async def recalculate_rewards(self) -> None:
        self.debug_print("recalculate_rewards", "Recalculating rewards")
        
        self.reward = await self.calculate_reward(self)
        self.information_gain = await self.calculate_information_gain(self)

        if not self.is_terminal and self.children:
            child_recalculations = [child.recalculate_rewards() for child in self.children]
            await asyncio.gather(*child_recalculations)

    async def elevate_layer(self) -> None:
        self.debug_print("elevate_layer", "Elevating layer and updating is_terminal status")
        if self.parent is not None:
            self.depth = self.parent.depth + 1
        else:
            self.depth = 0
        
        self.is_terminal = self.depth >= self.n_extend_layers - 1 or len(self.items) <= 2

        elevate_tasks = [child.elevate_layer() for child in self.children]
        await asyncio.gather(*elevate_tasks)

        self.debug_print("elevate_layer", f"Layer elevated, new depth: {self.depth}, is_terminal: {self.is_terminal}")

    async def calculate_rewards(self) -> None:
        self.debug_print("calculate_rewards", "Calculating rewards for all children")
        for child in self.children:
            child.reward = await self.calculate_reward(child)
            child.information_gain = await self.calculate_information_gain(child)

    async def calculate_reward(self, child: 'UoTNode') -> float:
        self.debug_print("calculate_reward", "Calculating reward")

        if not child.children:
            child.reward = child.information_gain
            self.debug_print("calculate_reward", f"Leaf node, reward=information_gain={child.reward}")
            return child.reward

        child_rewards = [await self.calculate_reward(grandchild) for grandchild in child.children]
        expected_child_reward = sum(child_rewards) / len(child_rewards)
        
        child.reward = child.information_gain + expected_child_reward

        self.debug_print("calculate_reward", f"Non-leaf node: information_gain={child.information_gain}, expected_child_reward={expected_child_reward}, total reward={child.reward}")

        return child.reward

    async def calculate_information_gain(self, child: 'UoTNode') -> float:
        self.debug_print("calculate_information_gain", f"Calculating information gain for child: {child.question}")

        initial_entropy = self.calculate_entropy
        child_entropy = child.calculate_entropy

        information_gain = initial_entropy - child_entropy

        self.debug_print("calculate_information_gain", f"Information gain calculated: initial_entropy={initial_entropy}, child_entropy={child_entropy}, information_gain={information_gain}")

        return information_gain

    @property
    def calculate_entropy(self) -> float:
        self.debug_print("calculate_entropy", "Calculating entropy")

        entropy = -sum(item.p_s * np.log2(item.p_s) for item in self.items if item.p_s > 0)

        self.debug_print("calculate_entropy", f"Entropy calculated: {entropy}")

        return entropy

    def _calculate_information_gain(self, omega_yes: List[Item], omega_no: List[Item], p_yes: float, p_no: float) -> float:
        self.debug_print("_calculate_information_gain", f"Calculating information gain: p_yes={p_yes}, p_no={p_no}")

        initial_entropy = self.calculate_entropy
        entropy_yes = -sum(item.p_s * np.log2(item.p_s) for item in omega_yes if item.p_s > 0)
        entropy_no = -sum(item.p_s * np.log2(item.p_s) for item in omega_no if item.p_s > 0)
        information_gain = initial_entropy - (p_yes * entropy_yes + p_no * entropy_no)

        self.debug_print("_calculate_information_gain", f"Information gain calculated: initial_entropy={initial_entropy}, entropy_yes={entropy_yes}, entropy_no={entropy_no}, information_gain={information_gain}")

        return information_gain

    def propagate_rewards(self) -> None:
        self.debug_print("propagate_rewards", "Propagating rewards")

        if self.parent:
            self.parent.propagate_rewards()

        self.debug_print("propagate_rewards", f"Rewards propagated, current reward={self.reward}")

    def set_best_question(self) -> None:
        self.debug_print("set_best_question", "Setting best question")

        self.best_question = self.get_best_question()

        self.debug_print("set_best_question", f"Best question set: {self.best_question}")

    def prune_children(self) -> None:
        self.debug_print("prune_children", f"Pruning children: Current number of children={len(self.children)}")

        if not self.children:
            self.debug_print("prune_children", "No children to prune")
            return
        
        paired_children = list(zip(self.children[::2], self.children[1::2]))
        paired_children.sort(key=lambda x: max(x[0].reward, x[1].reward), reverse=True)
        self.children = [child for pair in paired_children[:self.n_max_pruning] for child in pair]

        self.debug_print("prune_children", f"Children pruned: New number of children={len(self.children)}")

    def get_top_5_items(self) -> dict[str, float]:
        """
        Sort self.items and retrieve the top 5 items based on their probability values.
        If there are items with the same probability, include all of them even if it exceeds 5 items.
        In other words, sort the probability values and append all items with the same value.
        Stop when the index exceeds 5 for the first time.
        Return these top 5 items as a dictionary.
        """

        sorted_items = sorted(self.items, key=lambda x: x.p_s, reverse=True)
        top_5_items = {}
        for item in sorted_items:
            if len(top_5_items) < 5 or item.p_s == list(top_5_items.values())[-1]:
                top_5_items[item.get_name()] = item.p_s
            if len(top_5_items) > 5 and item.p_s != list(top_5_items.values())[-1]:
                break
        return top_5_items
        
    async def generate_children(self) -> None:
        self.debug_print("generate_children", f"Generating child nodes: n_question_candidates={self.n_question_candidates}")

        if self.is_terminal:
            self.debug_print("generate_children", "Terminal node, no children generated")
            return

        start_time = time.time()
        try:
            top_5_items = self.get_top_5_items()
            raw_questions_and_probabilities = await generate_questions_and_estimate_probability(self.items, self.n_question_candidates, self.history, max_item_size_at_once=self.max_item_size_at_once, top_5_items=top_5_items)
            success = True
        except Exception as e:
            self.debug_print("generate_children", f"API call failed: {str(e)}")
            success = False
            raw_questions_and_probabilities = None
        finally:
            end_time = time.time()
            duration = end_time - start_time
            self.log_api_call("generate_questions", success, duration / 2)
            self.log_api_call("estimate_probability", success, duration / 2)

        if self.is_debug:
            self.debug_print("generate_children", self.get_api_stats())

        if not raw_questions_and_probabilities:
            self.debug_print("generate_children", "No questions generated")
            return

        self.generated_info = []
        for question_data in raw_questions_and_probabilities:
            question = question_data["question"]
            item_probabilities = question_data["evaluated_items"]

            p_yes = sum(item_prob["p_yes_given_item"] * item.p_s for item_prob, item in zip(item_probabilities, self.items))
            p_no = 1 - p_yes

            omega_yes, omega_no, p_yes_given_items, p_no_given_items = self._calculate_posterior_probabilities(item_probabilities, p_yes, p_no)

            information_gain = self._calculate_information_gain(omega_yes, omega_no, p_yes, p_no)

            self.generated_info.append({
                "question": question,
                "p_yes": p_yes,
                "p_no": p_no,
                "p_yes_given_items": p_yes_given_items,
                "p_no_given_items": p_no_given_items,
                "omega_yes": omega_yes,
                "omega_no": omega_no,
                "information_gain": information_gain
            })

            is_terminal_yes = self.depth + 1 >= self.n_extend_layers - 1 or len(omega_yes) <= 2
            is_terminal_no = self.depth + 1 >= self.n_extend_layers - 1 or len(omega_no) <= 2

            yes_node = UoTNode(question, omega_yes, parent=self, reply=True, p_reply=p_yes, 
                            history=self.history + [{"q": question, "a": True, "p": p_yes}], 
                            is_debug=self.is_debug, is_terminal=is_terminal_yes, generated_info=self.generated_info)
            no_node = UoTNode(question, omega_no, parent=self, reply=False, p_reply=p_no, 
                            history=self.history + [{"q": question, "a": False, "p": p_no}], 
                            is_debug=self.is_debug, is_terminal=is_terminal_no, generated_info=self.generated_info)

            yes_node.configure_node(self.n_extend_layers, self.accumulation, self.expected_method, self.n_max_pruning, self.lambda_, self.n_question_candidates)
            no_node.configure_node(self.n_extend_layers, self.accumulation, self.expected_method, self.n_max_pruning, self.lambda_, self.n_question_candidates)

            self.children.extend([yes_node, no_node])

            self.debug_print("generate_children", f"Generated child nodes for question: {question}, p_yes={p_yes}, p_no={p_no}, information_gain={information_gain}")

        self.debug_print("generate_children", f"Total child nodes generated: {len(self.children)}")

    async def custom_bayesian_update(self, p_prime_y: float) -> None:
        print(f"\n[CUSTOM BAYESIAN UPDATE] Starting: p_prime_y={p_prime_y}")
        start_time = time.time()
        try:
            p_s = [item.p_s for item in self.items]
            print(f"[STEP 1] Initial p_s: {p_s}")
            if self.generated_info:
                question_data = next((info for info in self.generated_info if info["question"] == self.question), None)
                if question_data:
                    p_y_given_s = question_data["p_yes_given_items"]
                    p_n_given_s = question_data["p_no_given_items"]
                    print(f"[STEP 2] Using generated info: p_y_given_s={p_y_given_s}")
                else:
                    p_y_given_s = [0.5 for _ in self.items]
                    p_n_given_s = [0.5 for _ in self.items]
                    print("[STEP 2] Using default probabilities: p_y_given_s=[0.5, ...]")
            else:
                p_y_given_s = [0.5 for _ in self.items]
                p_n_given_s = [0.5 for _ in self.items]
                print("[STEP 2] Using default probabilities: p_y_given_s=[0.5, ...]")
            
            p_prime_n = 1 - p_prime_y
            print(f"[STEP 3] p_prime_n: {p_prime_n}")

            p_y = sum(p_y_given_s[i] * p_s[i] for i in range(len(p_s)))
            p_n = 1 - p_y
            print(f"[STEP 4] Calculated p_y: {p_y}, p_n: {p_n}")

            p_s_given_y = [p_y_given_s[i] * p_s[i] / p_y if p_y > 0 else 0 for i in range(len(p_s))]
            p_s_given_n = [p_n_given_s[i] * p_s[i] / p_n if p_n > 0 else 0 for i in range(len(p_s))]
            print(f"[STEP 5] p_s_given_y: {p_s_given_y}")
            print(f"[STEP 5] p_s_given_n: {p_s_given_n}")

            e_p_s = [p_prime_y * p_s_given_y[i] + p_prime_n * p_s_given_n[i] for i in range(len(p_s))]
            print(f"[STEP 6] e_p_s: {e_p_s}")

            r_s = [1 / (1 + self.lambda_ * (p_y_given_s[i] - p_prime_y) ** 2) for i in range(len(p_s))]
            print(f"[STEP 7] r_s: {r_s}")

            e_prime_p_s = [r_s[i] * e_p_s[i] for i in range(len(p_s))]
            print(f"[STEP 8] e_prime_p_s: {e_prime_p_s}")

            sum_e_prime = sum(e_prime_p_s)
            print(f"[STEP 9] sum_e_prime: {sum_e_prime}")

            p_s_new = [e_prime_p_s[i] / sum_e_prime for i in range(len(p_s))]
            print(f"[STEP 10] p_s_new (before update): {p_s_new}")

            print("[STEP 11] Updating individual item probabilities:")
            for item, new_p_s in zip(self.items, p_s_new):
                old_p_s = item.p_s
                item.update_p_s(new_p_s)
                print(f"  - Item {item.get_name()}: old p_s={old_p_s:.6f}, new p_s={item.p_s:.6f}")

            success = True
        except Exception as e:
            print(f"[ERROR] Bayesian update failed: {str(e)}")
            print(f"[ERROR] Error details: {traceback.format_exc()}")
            success = False
        finally:
            end_time = time.time()
            duration = end_time - start_time
            print(f"[TIMING] custom_bayesian_update duration: {duration:.6f} seconds")

        print("[CUSTOM BAYESIAN UPDATE] Finished")


    def get_simplified_history(self) -> List[Dict[str, Union[str, bool, float]]]:
        self.debug_print("get_simplified_history", f"Retrieving simplified history: length={len(self.history)}")
        return self.history

    def get_best_question(self) -> Optional[str]:
        self.debug_print("get_best_question", "Getting best question")
        
        if not self.children or len(self.children) < 2:
            self.debug_print("get_best_question", "Insufficient children, returning None")
            return None
        
        # Yes/Noペアごとに期待報酬を計算
        question_rewards = []
        for i in range(0, len(self.children), 2):
            yes_node = self.children[i]
            no_node = self.children[i+1]
            
            # 期待報酬の計算: E[R(q)] = p_y(q)*r_y(q) + p_n(q)*r_n(q)
            expected_reward = yes_node.p_reply * yes_node.reward + no_node.p_reply * no_node.reward
            
            question_rewards.append((yes_node.question, expected_reward))
        
        # 最大の期待報酬を持つ質問を選択
        best_question, _ = max(question_rewards, key=lambda x: x[1])
        
        self.debug_print("get_best_question", f"Best question: {best_question}")
        return best_question

    def get_yes_no_nodes(self, question: str) -> Tuple['UoTNode', 'UoTNode']:
        yes_node = next((child for child in self.children if child.question == question and child.reply), None)
        no_node = next((child for child in self.children if child.question == question and not child.reply), None)
        return yes_node, no_node

    def create_merged_node(self, yes_node: 'UoTNode', no_node: 'UoTNode') -> 'UoTNode':
        self.debug_print("create_merged_node", "Creating merged node")
        
        merged_items = []
        for yes_item, no_item in zip(yes_node.items, no_node.items):
            merged_p_s = (yes_item.p_s + no_item.p_s) / 2
            merged_item = Item(yes_item.get_name(), yes_item.description, merged_p_s)
            merged_items.append(merged_item)
        
        merged_node = UoTNode(
            question="merged_root",
            items=merged_items,
            parent=None,
            reply=None,
            p_reply=0.5,
            history=self.history,
            is_debug=self.is_debug,
            is_terminal=self.is_terminal,
            max_item_size_at_once=self.max_item_size_at_once,
            generated_info=self.generated_info
        )
        merged_node.configure_node(
            n_extend_layers=self.n_extend_layers,
            accumulation=self.accumulation,
            expected_method=self.expected_method,
            n_max_pruning=self.n_max_pruning,
                lambda_=self.lambda_,
                n_question_candidates=self.n_question_candidates
            )
        self.debug_print("create_merged_node", "Merged node created")
        return merged_node

    async def handle_equal_probability(self, yes_node: 'UoTNode', no_node: 'UoTNode', p_prime_y: float) -> 'UoTNode':
        self.debug_print("handle_equal_probability", "Creating merged node")
        
        merged_items = []
        for yes_item, no_item in zip(yes_node.items, no_node.items):
            merged_p_s = (yes_item.p_s + no_item.p_s) / 2
            merged_item = Item(yes_item.get_name(), yes_item.description, merged_p_s)
            merged_items.append(merged_item)

        merged_node = UoTNode(
            question=f"Merged: {yes_node.question}",
            items=merged_items,
            parent=None,
            reply=None,
            p_reply=p_prime_y,
            history=self.history + [{"q": yes_node.question, "a": None, "p": p_prime_y}],
            is_debug=self.is_debug
        )

        merged_node.configure_node(
            n_extend_layers=self.n_extend_layers,
            accumulation=self.accumulation,
            expected_method=self.expected_method,
            n_max_pruning=self.n_max_pruning,
            lambda_=self.lambda_,
            n_question_candidates=self.n_question_candidates
        )

        # Merge children from yes_node and no_node
        merged_node.children = yes_node.children + no_node.children

        # Recalculate rewards for all children
        await merged_node.calculate_rewards()

        # Prune children to keep only top n_max_pruning pairs
        merged_node.prune_children()

        self.debug_print("handle_equal_probability", "Merged node created")
        return merged_node

    async def update_probabilities(self, p_prime_y: float, is_root: bool = False, parent_items: Optional[List[Item]] = None) -> None:
        self.debug_print("update_probabilities", f"Updating probabilities: p_prime_y={p_prime_y}, is_root={is_root}")
        
        if is_root:
            await self.custom_bayesian_update(p_prime_y)
            self.history.append({"q": self.question, "a": self.reply, "p": p_prime_y})
        else:
            # parent_itemsからitem_probabilitiesを作成
            item_probabilities = [{"p_yes_given_item": self.p_reply} for _ in parent_items]
            omega_yes, omega_no, _, _ = self._calculate_posterior_probabilities(item_probabilities, p_prime_y, 1 - p_prime_y)
            
            # 現在のノードの返答に基づいて適切な確率を選択
            self.items = omega_yes if self.reply else omega_no

        # 正規化を確実に行う
        total_prob = sum(item.p_s for item in self.items)
        for item in self.items:
            item.p_s /= total_prob

        # 子ノードの確率を更新
        if not self.is_terminal and self.children:
            child_updates = [child.update_probabilities(p_prime_y, is_root=False, parent_items=self.items) for child in self.children]
            await asyncio.gather(*child_updates)

        # デバッグ出力
        self.debug_print("update_probabilities", "Updated probabilities:")
        for item in self.items:
            self.debug_print("update_probabilities", f"{item.get_name()}: {item.p_s:.4f}")
        total_after = sum(item.p_s for item in self.items)
        self.debug_print("update_probabilities", f"Total probability after update: {total_after:.4f}")

        
    async def handle_unequal_probability(self, yes_node: 'UoTNode', no_node: 'UoTNode', p_prime_y: float) -> 'UoTNode':
        self.debug_print("handle_unequal_probability", "Handling unequal probability case")
        
        best_child = yes_node if p_prime_y > 0.5 else no_node
        await best_child.update_probabilities(p_prime_y)
        
        self.debug_print("handle_unequal_probability", "Unequal probability case handled")
        return best_child

    def __str__(self) -> str:
        return f"UoTNode(question='{self.question}', depth={self.depth}, items={len(self.items)}, is_terminal={self.is_terminal})"

    def __repr__(self) -> str:
        return f"UoTNode(question='{self.question}', depth={self.depth}, items={len(self.items)}, reward={self.reward:.4f}, information_gain={self.information_gain:.4f}, is_terminal={self.is_terminal})"

    def print_node_info(self, indent: int = 0) -> str:
        indent_str = "  " * indent
        info = f"{indent_str}depth={self.depth},\n"
        info += f"{indent_str}is_terminal={self.is_terminal},\n"
        info += f"{indent_str}question='{self.question}',\n"
        info += f"{indent_str}history={self.history},\n"
        info += f"{indent_str}reply={'Yes' if self.reply else 'No' if self.reply is not None else 'N/A'},\n"
        info += f"{indent_str}p_reply={'N/A' if self.p_reply is None else f'{self.p_reply:.4f}'},\n"
        info += f"{indent_str}reward={self.reward:.4f},\n"
        info += f"{indent_str}information_gain={self.information_gain:.4f},\n"
        info += f"{indent_str}items=[\n"
        for item in self.items:
            info += f"{indent_str}    name={item.get_name()}, p_s={item.p_s:.4f}\n"
        info += f"{indent_str}]"
        return info