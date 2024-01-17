# Databricks notebook source
# MAGIC %md
# MAGIC # Use langchain with llm served from Databricks
# MAGIC
# MAGIC We use a mistral model served from another cluster which has GPU.
# MAGIC
# MAGIC Can be run on a non-gpu cluster.
# MAGIC
# MAGIC ## What is Langchain?
# MAGIC
# MAGIC LangChain is an intuitive open-source Python framework build automation around LLMs), and allows you to build dynamic, data-responsive applications that harness the most recent breakthroughs in natural language processing.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Inference examples

# COMMAND ----------

# MAGIC %pip install -q -U langchain
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md Get llm server constants from constants table

# COMMAND ----------

import os
import requests
import numpy as np
import pandas as pd
import json

def create_tf_serving_json(data):
  return {'inputs': {name: data[name].tolist() for name in data.keys()} if isinstance(data, dict) else data.tolist()}

def score_model(dataset):
  url = 'https://dbc-639f4875-165d.cloud.databricks.com/serving-endpoints/norsk7bqloramistral250/invocations'
  headers = {'Authorization': f'Bearer {os.environ.get("DATABRICKS_TOKEN")}', 'Content-Type': 'application/json'}
  ds_dict = {'dataframe_split': dataset.to_dict(orient='split')} if isinstance(dataset, pd.DataFrame) else create_tf_serving_json(dataset)
  data_json = json.dumps(ds_dict, allow_nan=True)
  response = requests.request(method='POST', headers=headers, url=url, data=data_json)
  if response.status_code != 200:
    raise Exception(f'Request failed with status {response.status_code}, {response.text}')
  return response.json()

score_model({
  "prompt": "Hva er stortinget?",
  "max_tokens": 100,
  "temperature": 0.1
})

# COMMAND ----------



# COMMAND ----------

foo = bardoesntexist

# COMMAND ----------



# COMMAND ----------

server_num = 1
constants_table = f"training.llm_langchain_shared.server{server_num}_constants"
constants_df = spark.read.table(constants_table)
raw_dict = constants_df.toPandas().to_dict()
names = raw_dict['name'].values()
vars = raw_dict['var'].values()
constants = dict(zip(names, vars))
cluster_id = constants['cluster_id']
port = constants['port']
host = constants['host']
api_token = constants['api_token']

# COMMAND ----------

from langchain import PromptTemplate, LLMChain
from langchain.llms import Databricks
llm = Databricks(host=host, cluster_id=cluster_id, cluster_driver_port=port, api_token=api_token,)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prompt parameters

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC You can create a prompt that either has only an instruction or has an instruction with context:

# COMMAND ----------

from langchain import PromptTemplate, LLMChain
from langchain.llms import HuggingFacePipeline
from langchain.llms import Databricks

# template for an instruction with no input
prompt = PromptTemplate(
    input_variables=["instruction"],
    template="{instruction}")

# template for an instruction with input
prompt_with_context = PromptTemplate(
    input_variables=["instruction", "context"],
    template="{instruction}\n\nInput:\n{context}")
    
llm_chain = LLMChain(llm=llm, prompt=prompt)
llm_context_chain = LLMChain(llm=llm, prompt=prompt_with_context)

# COMMAND ----------

# MAGIC %md
# MAGIC Example predicting using a simple instruction:

# COMMAND ----------

print(llm_chain.predict(instruction="Gi meg fem punkter for en god miljøpolitikk, på norsk.").lstrip())

# COMMAND ----------

# MAGIC %md
# MAGIC Example predicting with and without context:

# COMMAND ----------

print(llm_chain.predict(instruction="Hva er stortingets funksjon?").lstrip())

# COMMAND ----------

context = """Stortingets oppgaver som nasjonalforsamling er nedfelt i Grunnloven § 75 og omfatter blant annet lovgivning, bevilgninger (skatter, avgifter, budsjett), kontroll av den utøvende makt (regjeringen) og drøfting av generelle politiske spørsmål som utenrikspolitikk og reformer."""

print(llm_context_chain.predict(instruction="Hva er stortingets funksjon?", context=context).lstrip())

# COMMAND ----------

# MAGIC
# MAGIC %md
# MAGIC
# MAGIC ### Simpler context

# COMMAND ----------

context = """Stortinget er Norges folkevalgte nasjonalforsamling. De viktigste oppgavene til Stortinget er å vedta lover, bestemme statens budsjett og kontrollere regjeringen. Stortinget er også en viktig arena for politisk debatt."""

print(llm_context_chain.predict(instruction="Hva er stortingets funksjon?", context=context).lstrip())

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Task:
# MAGIC
# MAGIC Play around with the context and the instructions.

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Task: Get a good answer in Norwegian about which ideological difference there are between the main parties
# MAGIC in Norway, without context.
# MAGIC

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Task: Get a good answer in Norwegian about which ideological differences there are between the main parties
# MAGIC in Norway, using a context you decide yourself.

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Task: Get a good answer about the crucial conditions for Norwegian fishing exports.
# MAGIC You can use context, but play also around with sentiment.

# COMMAND ----------


