evaluate_probabilities_of_chunk_prompt = '''
You need to evaluate probabilities of a yes option of the following incidents based on the user's experience:
{item_name_list}

Follow these output format instructions:
{format_instructions}

Here is the Q&A history for context:
{history}

Here is the current question to address:
{question}

Please be careful to generate name, description, thoughts, and y_prob without missing any!!!
For each case, first evaluate how likely it is that you can respond yes, taking into account the history of the case so far. Rate on a scale of 0~1, paying particular attention to the degree of discrepancy or agreement between the dimensions of the abstract phenomenon:
- 0.8 ~ 1.0 <- Definitely yes (direct relationship)
- 0.6 ~ 0.8 <- Approximately yes (slightly different layers, but generally correct)
- 0.4 ~ 0.6 <- Difficult to determine because the layers are different, could be either, question is incomprehensible and unanswerable (0.5), difficult to understand, out of context, rude, etc.
- 0.2 ~ 0.4 <- Approximately no (slightly different layers, but generally wrong)
- 0.0 ~ 0.2 <- definitely no (direct relationship)

Please be careful not to miss any item. name, description, thoughts, and y_prob. Your response is as long as need.


Caution:
You must evaluate all items under the question. Do not change the names and description of the items. NERVER MISTAKE. Be careful not to miss any item.
you cannot write except json content even if just one word. You often insert "Here is the JSON output with the evaluation of the probability of a 'yes' response for the given list of items:" before json. THIS IS SUCK, 100% BAN!!! Insrt NOTINHG except json content itself.
SO, THE FIRST WORD OF THE OUTPUT YOU GENERATE MUST BE left curly bracket, MUST BE left curly bracket, MUST BE left curly bracket.
'''

evaluate_probabilities_of_chunk_discription="""
The discussion following steps below in ENGLISH. You must follow guaidline as below step by step.
1. In the evaluation process, we start with the initial probability evaluation. For each case, we determine the probability of a "yes" response to the closed query based on the direct relationship and history of the case. 
If the dimension or domain width of the question corresponds to the range of influence or granularity directly or in some extent, inferred from the case, we assign a score according to the next step (step 2). Otherwise, we move to the evaluation of discrepancies (step 3).

2. If the characteristics of the situation facing the case and the characteristics of the situation in the historical context are consistent with the characteristics of the question, the score will be closer to "yes," typically ranging from 0.6 to 1.0. If the characteristics differ, the score will be closer to "no," ranging from 0.0 to 0.4. In cases where both characteristics are possible, we weigh the relevant characteristics and decide according to their respective weights. This can be mathematically expressed as 
S=w1⋅S1+w2⋅S2, where the sum of the weights w1 and w2  equals 1, and S1 and S2 are the scores based on each characteristic. Additionally, we take into account the context and background of the question. If the cases are in slightly different layers but generally consistent with the question, we adjust the probability scores accordingly.

3. When evaluating discrepancies, we assess the degree of discrepancy between the dimensions of the abstract phenomenon. If there are large discrepancies, we adjust the score towards the middle range, approximately 0.5. Similarly, if the relevant relationships are complex or unclear, making it difficult to determine the correct response, we score in the middle range, around 0.5.

Anyway, keep your intuition about how you would like to answer that question when you are in an ITEM situation! (especially yes)

So, let's give a example. Imagine you are thinking of A as the Item and given Q as the question. If you want to say "yes" to the Q when doning A, It well be more than 0.75
In another case, A is not related to Q directly, but you are able to imagine item B that is next or previous one of A as action or situation. If you want to say "yes" to the Q when doing B, it will be around 0.6
you can expand this explae to "no" also!!!
"""

generate_questions_prompt = """
Your Task:
You need generate a list of {n} closed questions to devide the items into each 'yes', 'no', and 'irrelevant' equally, with 'both' and 'difficult' being as close to zero as possible. 
You must take history and additional_context into account to generate well considered questions in order not to make dementia user confused.
Each question should be unique and distinct from the others. 

Size of the list:
The size of the list is {n}. You need to generate {n} questions. Not more, not less.

Follow these output format instructions (100%, only json output):
{format_instructions}

Here is the Q&A history for context, Check history and avoid making similar questions. Questions that are very similar to the immediately preceding one should not be considered, unless there is a special reason.:
{history}

Here is the additional context where you can get more information to generate well considered questions:
{additional_context}

Here is all items you need to devide:
{item_name_list}

Here is top_5 items & their probabilities, if they are high prob, it is better way to focus on them to devide.:
{top_5_items}

Caution:
Do not forget to be respectful. Easy to answer question that ask say clearly yes or clearly no is better. In Japanese 「じゃないですか?」is difiicult to destingish yes or no because of grammer, so avoid to say. It is not a good question to directly name and definitively delve into something unless the probability of it being correct exceeds 30%. Otherwise, if exceed 30% by just single item, you can make a question that ask about the item directly.
Beautiful question is one in which the items are equally divided by the question into 'yes', 'no', and 'irrelevant', with 'both' and 'difficult' being as close to zero as possible. Please avoid making definitive statements about things you are not supposed to know, as it can be unpleasant.
Each question should be unique and distinct from the others and 100% closed question. You cannot output anythingexcept json format, even if just one word!!!
you cannot write except json content even if just one word.
"""

generate_questions_discription = """
Generated closed question, one of n well-distributed questions.
Only one proposition is allowed in each question, as the question must be of concern to the person with Alzheimer's. Also, context should be kept as independent as possible, avoiding pronouns.
The tone should be gentle and polite, the Japanese language should be very fluent, warm and respectful.
Do not forget the honorific. A beautiful question is one in which 
is a question in which the items are equally divided into 'Yes', 'No' and 'Not relevant', 
'both' and 'difficult' are as close to zero as possible.
Prepare a variety of strategically itemised questions, taking into account history.

Caution:
When asking questions, it is not impressive to speak definitively about something that is not true.
Keep your questions simple and closed questions. Please use a friendly tone and be polite.
As there is only one proposition, the technique of forcing multiple contradictory statements into one closed question is not acceptable. Also, the User does not know about the ITEM of the choice. These are on the system, and questions that explicitly assume these items are undesirable. These items are choices on our side to estimate the user's states and beliefs, and care must be taken.
Avoid making so much similar questions with previous questions, so you have to think about the context of the history. Even if it results in the kinda similar outcome, there should be some intentional effort to delve deeper or expand on the topic (still not good tho).
"""

generate_questions_thought_discription = """
The discussion according to the following step in ENGLISH.
1. What is the index here? (from 1 to n.)
2. How to devides item well? In other words, what is the best way to divide the items into 'yes', 'no', and 'irrelevant', with 'both' and 'difficult' being as close to zero as possible?
3. Dose the quetion differ from the previous all questions? Check history! Even if it results in the same outcome, there should be some intentional effort to delve deeper or expand on the topic.
4. If the question is well considered, please generate a question with going ahead. Else go to step 2 again. 
"""

