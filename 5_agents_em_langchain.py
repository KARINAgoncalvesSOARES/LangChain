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

Nesse escript estamos criando a nossa própria ferramenta de conhecimentos gerais e estamos adicionando ela ao agent.

"""
import os
from langchain_openai import ChatOpenAI
from langchain.agents import Tool, initialize_agent
from langchain.chains import LLMMathChain
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain


import warnings
warnings.simplefilter("ignore")

# Verifica se a chave da API está definida
if "OPENAI_API_KEY" not in os.environ:
    raise ValueError("A chave da API OpenAI não foi definida.")

key_openai = os.environ["OPENAI_API_KEY"]

# Verifica se o modelo está disponível
model = "gpt-3.5-turbo"
llm = ChatOpenAI(api_key=key_openai, temperature=0, model=model)


# CREATE TOOLS

math_chain = LLMMathChain.from_llm(llm=llm)

math_tool = Tool(
    name="Calculadora",
    func=math_chain.run,
    description="Útil para quando você precisa responder perguntas relacionadas a matemática."
)

tools = [math_tool]

#print(tools)

# Uma cadeia simples para lidar com consultas de conhecimento geral

prompt = PromptTemplate(
    input_variables=["query"],
    template="{query}",
)
chain = LLMChain(llm=llm, prompt=prompt)

tool = Tool(
    name="language model",
    func=chain.run,
    description="use this tool for general purpose",
)

"""
remember:
tools = load_tools(
    ["llm-math"],
    llm=llm
    )
"""

tools.append(tool)


agent = initialize_agent(
    agent="zero-shot-react-description",
    tools=tools,
    llm=llm,
    verbose=True,
    #max_iterations=3

)

#print(agent)

# Usa o método invoke do math_tool
response = agent.invoke("Quem é o fundador da Turquia moderna?")
print(response)


#Comando para rodar o código:
#$python3 nome_do_arquivo

#pip install numexpr