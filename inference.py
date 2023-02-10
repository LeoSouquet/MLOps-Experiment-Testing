from transformers import pipeline

from clearml import Task

task = Task.init(project_name='HuggingFace Transformers', task_name='Trainer',reuse_last_task_id=False)
logger = task.get_logger()




question = "How many programming languages does BLOOM support?"
context = "BLOOM has 176 billion parameters and can generate text in 46 languages natural languages and 13 programming languages."


question_answerer = pipeline("question-answering", model="my_awesome_qa_model\checkpoint-500")
question_answerer_2 = pipeline("question-answering", model="my_awesome_qa_model\checkpoint-2500")
print(question_answerer(question=question, context=context))
print(question_answerer_2(question=question, context=context))

#logger.report_text(f"Context {context} - Question : {question} - Answer : {question_answerer(question=question, context=context)}")




import pandas as pd
  
# initialize list of lists
data = [[context,question, f"{question_answerer(question=question, context=context)}" ]]
  
# Create the pandas DataFrame
df = pd.DataFrame(data, columns=['context', 'question', 'answer'])
  




print("---")

question = "Which language does BLOOM's mum speak?"
context = "My name is BLOOM and my Mum speaks 2 languages. My mum's name is Josette?"

print(question_answerer(question=question, context=context))

print("---")

df.loc["1"] = [context,question, f"{question_answerer(question=question, context=context)}" ]

# print dataframe.
print(df)

logger.report_table(
    "Test Inference QA", 
    "Context , Question, Answer", 
    iteration=0, 
    table_plot=df
)

"""
question = "Who is Josette's son?"
context = "My name is Leo (I am a Boy) and my Mum speaks 2 languages. My sister is Blandine. My mum's name is Josette?"

print(question_answerer(question=question, context=context))
print(question_answerer_2(question=question, context=context))

print("---")
question = "Who is Josette's daughter?"
context = "My name is Leo (I am a Boy) and my Mum speaks 2 languages. My sister is Blandine. My mum's name is Josette?"

print(question_answerer(question=question, context=context))
print(question_answerer_2(question=question, context=context))
print("---")

"""
"""{'score': 0.16683945059776306, 'start': 10, 'end': 95, 'answer': '176 billion parameters and can generate text in 46 languages natural languages and 13'}
{'score': 0.47666090726852417, 'start': 65, 'end': 72, 'answer': 'Josette'}
{'score': 0.29282262921333313, 'start': 72, 'end': 80, 'answer': 'Blandine'}
{'score': 0.5974862575531006, 'start': 72, 'end': 80, 'answer': 'Blandine'}"
"""
