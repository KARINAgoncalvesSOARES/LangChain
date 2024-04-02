#!/usr/bin/env python3

"""
Data Scientist Jr.: Karina Gonçalves Soares
"""

"""
NER
===

O reconhecimento de entidades nomeadas (NER) é uma técnica de processamento de linguagem natural (PNL)
usada para identificar e classificar entidades nomeadas dentro de um texto em categorias predefinidas,
como nomes de pessoas, organizações, locais, datas, quantidades, valores monetários, porcentagens e muito mais.
O objetivo principal do NER é extrair e categorizar entidades específicas mencionadas em dados de texto não-estruturados
para compreender melhor as informações e relacionamentos subjacentes no texto.

Etapas do NER
-------------

Tokenization: dividir o texto em palavras ou tokens individuais.
=============

Part-of-Speech Tagging: Atribuição de classes gramaticais do speech (por exemplo, substantivo, verbo, adjetivo) a cada token.
=======================

Named Entity Classification: identificar tokens que representam entidades nomeadas e atribuí-los a categorias predefinidas, como nomes de pessoas, nomes de organizações, locais, etc.
============================

Entity Extraction: Extrair as entidades nomeadas identificadas junto com suas respectivas categorias do texto.
==================
"""


from typing import List
from pydantic import BaseModel, Field
from langchain.utils.openai_functions import convert_pydantic_to_openai_function
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI # pip install langchain-openai

import warnings
warnings.simplefilter("ignore")



# Carregar a Chave da API da OpenAI:
import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
key_openai  = os.environ['OPENAI_API_KEY']


model = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")


class NER(BaseModel):
    """Queremos extrair as entidades "nome", "idade", "data", "endereço", "telefone" e 'conta bancária'"""
    ner: List[str] = Field(description="A entidade detectada no documento, como nome, idade, data, endereço, telefone, nome da empresa e conta bancária")
    type: List[str] = Field(description="O tipo da entidade detectada com valores possíveis: 'nome', 'idade', 'data', 'endereço', 'telefone', 'nome da empresa' e 'conta bancária'. Para cada entidade detectada no ner este deve ser o tipo correspondente")



extraction_functions = [convert_pydantic_to_openai_function(NER)]
extraction_model = model.bind(functions=extraction_functions, function_call={"name": "NER"})


prompt = ChatPromptTemplate.from_messages([("system", "Extraia as informações relevantes; se não forem fornecidas explicitamente, não adivinhe. Extraia informações parciais"),
                                           ("human", "{input}")
                                          ])


extraction_chain = prompt | extraction_model | JsonOutputFunctionsParser()


# Meu texto, por exemplo:
mytxt = """Me chamo Fulana, eu sou nascida em Perú-Lima e estou querendo visitar a minha cidade,
mas para fazer isso preciso retirar meu passaporte em São Paulo-Brasil atravéz da Polícia Federal. """
  
 

ner_dict = extraction_chain.invoke({"input": mytxt})
print(ner_dict)

print("")

# Seguidamente convertemos para uma Tabela com o Pandas:
print("🤗🤗🤗 Convertendo para um DataFrame 🤗🤗🤗")


import pandas as pd
df = pd.DataFrame(ner_dict)
print(df.head())

# Comando para rodar o código
# $python3 nome_do_arquivo

