import os
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
#os.environ['OPENAI_API_KEY'] = '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'
os.environ["HUGGINGFACEHUB_API_TOKEN"] = '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'
from langchain import HuggingFaceHub
#import openai

#question = "Where is the highest mountain in the world? "

#template = """Question: {question}

#Answer: Let's think step by step."""

#prompt = PromptTemplate(template=template, input_variables=["question"])

repo_id = "bigscience/bloom-560m"  # See https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads for some other options


llm = HuggingFaceHub(repo_id=repo_id,client='', model_kwargs={"temperature": 0.5, "max_length": 200})
#llm_chain = LLMChain(prompt=prompt, llm=llm)

#print(llm_chain.run(question))
#name=llm("Suggest anme for  tech firm")
#print(name)

p_template=PromptTemplate(template="Suggest a name for  my {type} firm which is a startup",input_variables=['type'])
c_chain=LLMChain(llm=llm,prompt=p_template)
print(c_chain.run("law"))

