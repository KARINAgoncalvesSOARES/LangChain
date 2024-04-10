#!/usr/bin/env python3

"""
Data Scientist Jr.: Karina Gonçalves Soares

Link de estudo: 

1. https://medium.com/dev-genius/langchain-in-chains-21-agents-f8616a15cbff - LangChain in Chains #21: Agents!
2. https://www.hostinger.com.br/tutoriais/o-que-e-react-javascript#:~:text=O%20React%20%C3%A9%20uma%20biblioteca,Model%2DView%2DController). - React

"""

"""
PRIMEIROS PASSOS COM AGENTES EM LANGCHAIN
=========================================


Os agentes são sistemas construídos em torno dos LLMs (Large Language Models)
e utilizam os mesmos para tomar decisões e realizar suas tarefas.
Os agentes proporcionam versatilidade, podendo ser usados em ambientes complexos e dinâmicos. 
Um exemplo disso é que os LLMs, que não possuem um bom desempenho em operações matemáticas, 
podem demonstrar uma melhoria de desempenho quando integrados aos agentes.

Em vez de especificar caminhos ou locais específicos das ferramentas, a função 'load_tools' permite carregar ferramentas
simplesmente fornecendo seus nomes. Isso torna o processo mais cômodo e portátil,
nos dando uma alternativa mais direta para carregar essas ferramentas sem precisar usar a abordagem do LLMMathChain.

"""
import os
from langchain_openai import ChatOpenAI
from langchain.agents import Tool, initialize_agent
from langchain.agents import initialize_agent, load_tools

import warnings
warnings.simplefilter("ignore")

# Verifica se a chave da API está definida
if "OPENAI_API_KEY" not in os.environ:
    raise ValueError("A chave da API OpenAI não foi definida.")

key_openai = os.environ["OPENAI_API_KEY"]

# Verifica se o modelo está disponível
model = "gpt-3.5-turbo"
llm = ChatOpenAI(api_key=key_openai, temperature=0, model=model)


tools = load_tools(
    ['llm-math'],
    llm
)

agent = initialize_agent(
    agent="zero-shot-react-description",
    tools=tools,
    llm=llm,
    verbose=True,
    #max_iterations=3

)

#print(agent)

# Usa o método invoke do math_tool
response = agent.invoke("Kanye West tem 46 anos e Taylor Swift tem 34 anos. Qual será a soma de suas idades daqui a 10 anos?")
print(response)


#Comando para rodar o código:
#$python3 nome_do_arquivo

#pip install numexpr