#!/usr/bin/env python3

"""
Data Scientist Jr.: Karina Gonçalves Soares

Link de estudo: 

1. https://medium.com/dev-genius/langchain-in-chains-21-agents-f8616a15cbff - LangChain in Chains #21: Agents!

"""

"""
PRIMEIROS PASSOS COM AGENTES EM LANGCHAIN
=========================================


Os agentes são sistemas construídos em torno dos LLMs (Large Language Models)
e utilizam os mesmos para tomar decisões e realizar suas tarefas.
Os agentes proporcionam versatilidade, podendo ser usados em ambientes complexos e dinâmicos. 
Um exemplo disso é que os LLMs, que não possuem um bom desempenho em operações matemáticas, 
podem demonstrar uma melhoria de desempenho quando integrados aos agentes.


"""
import os
from langchain_openai import ChatOpenAI
from langchain.agents import Tool
from langchain.chains import LLMMathChain

# Verifica se a chave da API está definida
if "OPENAI_API_KEY" not in os.environ:
    raise ValueError("A chave da API OpenAI não foi definida.")

key_openai = os.environ["OPENAI_API_KEY"]

# Verifica se o modelo está disponível
model = "gpt-3.5-turbo"
llm = ChatOpenAI(api_key=key_openai, temperature=0, model=model)

math_chain = LLMMathChain.from_llm(llm=llm)

math_tool = Tool(
    name="Calculadora",
    func=math_chain.run,
    description="Útil para quando você precisa responder perguntas relacionadas a matemática."
)

tools = [math_tool]

# Usa o método invoke do math_tool
response = math_tool.invoke("quantos é 250-123* 2.5")
print(response)


#Comando para rodar o código:
#$python3 nome_do_arquivo

#$ pip install numexpr