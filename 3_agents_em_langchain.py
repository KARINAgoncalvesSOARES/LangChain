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

Nesse repositório criamos um agent utilizando "initialize_agent", 
Utilizamos o agente “zero-shot-react-description” (Isso basicamente indica que o agente tem a capacidade do conceito React e não possui memória.)
e o "max_iterations" onde específicamos as interções que o agent realizará durante sua operação, esse parametro ajuda a
controlar o comportamento do agente e como podemos ver ele resolve o problema por etapas usando raciocínio e nos
entrega a resposta correta.

"""
import os
from langchain_openai import ChatOpenAI
from langchain.agents import Tool, initialize_agent
from langchain.chains import LLMMathChain

import warnings
warnings.simplefilter("ignore")

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
    name="Calculadora",
    func=math_chain.run,
    description="Útil para quando você precisa responder perguntas relacionadas a matemática."
)

tools = [math_tool]

#print(tools)

agent = initialize_agent(
    agent="zero-shot-react-description",
    tools=tools,
    llm=llm,
    verbose=True,
    max_iterations=3

)

#print(agent)

# Usa o método invoke do math_tool
response = agent.invoke("quantos é 250-123*2.5")
print(response)


#Comando para rodar o código:
#$python3 nome_do_arquivo