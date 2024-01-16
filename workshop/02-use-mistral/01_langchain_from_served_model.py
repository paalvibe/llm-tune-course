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
# server_num = 1
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

# MAGIC %md
# MAGIC
# MAGIC ### Simpler context

# COMMAND ----------

context = """Stortinget er Norges folkevalgte nasjonalforsamling. De viktigste oppgavene til Stortinget er å vedta lover, bestemme statens budsjett og kontrollere regjeringen. Stortinget er også en viktig arena for politisk debatt."""

print(llm_context_chain.predict(instruction="Hva er stortingets funksjon?", context=context).lstrip())


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



# COMMAND ----------
