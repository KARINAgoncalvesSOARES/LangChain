#!/usr/bin/env python3

"""
Data Scientist Jr.: Karina Gonçalves Soares

Link de estudo: 

1. https://medium.com/dev-genius/langchain-in-chains-21-agents-f8616a15cbff
2. https://python.langchain.com/docs/modules/agents/quick_start

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
from langchain.agents import Tool, initialize_agent
from langchain.chains import LLMMathChain

# Verifica se a chave da API está definida
if "OPENAI_API_KEY" not in os.environ:
    raise ValueError("A chave da API OpenAI não foi definida.")

key_openai = os.environ["OPENAI_API_KEY"]

# Verifica se o modelo está disponível
model = "gpt-3.5-turbo"
llm = ChatOpenAI(api_key=key_openai, temperature=0, model=model)

from langchain.agents import Tool, initialize_agent
from langchain.chains import LLMMathChain

# CREATE TOOLS

math_chain = LLMMathChain.from_llm(llm=llm)

math_tool = Tool(
    name="Calculator",
    func=math_chain.run,
    description="Useful for when you need to answer questions related to Math."
)

tools = [math_tool]

print(tools)

agent = initialize_agent(
    agent="zero-shot-react-description",
    tools=tools,
    llm=llm,
    verbose=True,
    max_iterations=3

)

print(agent)

# Usa o método invoke do math_tool
response = agent.invoke("quantos é 250-123*2.5")
print(response)


#Comando para rodar o código:
#$python3 nome_do_arquivo