import tiktoken

from itertools import zip_longest

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import StrOutputParser, messages_to_dict
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables import RunnableLambda, RunnablePassthrough


from operator import itemgetter

from .prompts import gpt4o_sys_template


class Agent:
    def __init__(self):
        self.chain_dict = self.make_chain()


    def make_chain(self):
        """
        make chain instances

        Returns
        -------
        chain : dict
            chain for the current agent system,
            "chain" : chain instance
            "memory" : memory instance
        """

        # make a chain for fast_and_slow agent system
        gpt4o_model = ChatOpenAI(model="gpt-4o")
        #gpt3_5_turbo_model = ChatOpenAI(model="gpt-3.5-turbo")
        model_paser = gpt4o_model | StrOutputParser()
        #model_paser = gpt3_5_turbo_model | StrOutputParser()

        memory_for_gpt4o = ConversationBufferMemory(return_messages=True)
        memory_for_gpt4o.save_context({"input": ""}, {"output": "こんにちわ。いかがなさいましたか？"})

        gpt4o_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", gpt4o_sys_template),
                MessagesPlaceholder(variable_name="history"),
                ("user", "{input}"),
            ]
        )
        
        gpt4o_chain = (
            RunnablePassthrough.assign(
                history=RunnableLambda(memory_for_gpt4o.load_memory_variables) | itemgetter("history")
            )
            | gpt4o_prompt 
            | model_paser
        )

        return (
            {
                "chain": gpt4o_chain, 
                "memory": memory_for_gpt4o
            }
        )
