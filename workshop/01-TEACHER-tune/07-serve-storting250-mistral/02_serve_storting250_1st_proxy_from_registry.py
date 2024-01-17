# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Serving NorskGPT Mistral-7B-Instruct via vllm with a cluster driver proxy app
# MAGIC
# MAGIC The [Mistral-7B-Instruct-v0.1](https://huggingface.co/bineric/NorskGPT-Mistral-7b) Large Language Model (LLM) is a instruct fine-tuned version of the [Mistral-7B-v0.1](https://huggingface.co/bineric/NorskGPT-Mistral-7b) generative text model using a variety of publicly available conversation datasets.
# MAGIC
# MAGIC [vllm](https://github.com/vllm-project/vllm/tree/main) is an open-source library that makes LLM inference fast with various optimizations.
# MAGIC
# MAGIC Environment for this notebook:
# MAGIC - Runtime: 14.0 GPU ML Runtime
# MAGIC - Instance: `g5.xlarge` on AWS, `Standard_NV36ads_A10_v5` on Azure
# MAGIC - Will not run on g4dn.4xlarge

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install python libs

# COMMAND ----------

# %pip install git+https://github.com/huggingface/peft.git
# %pip install torch==2.1.0 accelerate==0.23.0
# %pip install -U transformers==4.34.0
# %pip install -U transformers==4.34.0
# %pip install bitsandbytes==0.41.1 einops==0.7.0 trl==0.7.1 peft==0.5.0 torch==2.1.0 accelerate==0.23.0 transformers==4.34.0
# %pip install -U vllm==0.2.0
# %pip install -U vllm==0.2.0 transformers==4.34.0 accelerate==0.20.3

%pip install -U transformers==4.34.0
%pip install bitsandbytes==0.41.1 einops==0.7.0 trl==0.7.1 peft==0.5.0

dbutils.library.restartPython()


#18 0.298 dependencies:
#18 0.298 - python=3.10.12
#18 0.298 - pip<=22.3.1
#18 0.298 - pip:
#18 0.298   - mlflow==2.8.1
#18 0.298   - torch
#18 0.298   - transformers
#18 0.298   - accelerate
#18 0.298   - einops
#18 0.298   - loralib
#18 0.298   - bitsandbytes
#18 0.298   - peft

# COMMAND ----------

# Get hugging face token to log into hugging face
# hf_token = dbutils.secrets.get(scope="llmtuning", key="huggingfacekey")

# COMMAND ----------

mlflowmodel_name = "norsk7bqloramistral"
run_id = "4ea2ed325d9644898350feea93d4f5c8"
logged_model = f"runs:/{run_id}/{mlflowmodel_name}"
print(f"logged_model: {logged_model}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## from pretrained example
# MAGIC https://mlflow.org/docs/latest/_modules/mlflow/transformers.html

# COMMAND ----------

import mlflow
mlflowmodel_name = "norsk7bqloramistral250"
models = mlflow.search_registered_models(filter_string=f"name = '{mlflowmodel_name}'")
artifacts_url = models[0].latest_versions[0].source

# COMMAND ----------

loaded_model = mlflow.pyfunc.load_model(model_uri=f"models:/{mlflowmodel_name}/2") # works
# mlflow.set_registry_uri('databricks-uc')
# loaded_model = mlflow.pyfunc.load_model(model_uri=f"models:/{mlflowmodel_name}@2")
# loaded_model = mlflow.pyfunc.load_model("dbfs:/databricks/mlflow-tracking/b011c0bde56242718d3384806e37e7e7/15bd6be40e70461597f6fa785c8a684a/artifacts/norsk7bqloramistral250")

# COMMAND ----------

# Make a prediction using the loaded model
loaded_model.predict(
    {
    "prompt": "Hva er rollen til stortinget?",
    "temperature": 0.5,
    "max_tokens": 150
    }
)

# COMMAND ----------

# Make a prediction using the loaded model
loaded_model.predict(
    {
    "prompt": "Hva er rollen til stortinget?",
    "temperature": 0.9,
    "max_tokens": 150
    }
)

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ## Add instruction prompt with sys section as guard rail

# COMMAND ----------

# Prompt templates as follows could guide the model to follow instructions and respond to the input, and empirically it turns out to make Falcon models produce better responses
DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information. Answer in Norwegian."""

INTRO_BLURB = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
PROMPT_FOR_GENERATION_FORMAT = """
<s>[INST]<<SYS>>
{system_prompt}
<</SYS>>


{instruction}
[/INST]
""".format(
    system_prompt=DEFAULT_SYSTEM_PROMPT,
    instruction="{instruction}"
)

# PROMPT_FOR_GENERATION_FORMAT2 = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

# If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information. Answer in Norwegian.

# ### Instruction:

# """

# COMMAND ----------

import pandas as pd

# Define parameters to generate text
def gen_text_for_serving(prompt, **kwargs):
    orig_prompt = prompt
    # prompt = PROMPT_FOR_GENERATION_FORMAT.format(instruction=prompt)

    # the default max length is pretty small (20), which would cut the generated output in the middle, so it's necessary to increase the threshold to the complete response
    if "context" not in kwargs:
        kwargs["context"] = None

    if "max_tokens" not in kwargs:
        kwargs["max_tokens"] = 512
    if "temperature" not in kwargs:
        kwargs["temperature"] = 0.5
    context = kwargs["context"]
    max_tokens = kwargs["max_tokens"]
    temperature = kwargs["temperature"]

    
    # sampling_params = SamplingParams(**kwargs)

    text_example=pd.DataFrame({
            "prompt":[prompt],
            "temperature": [temperature],
            "max_tokens": [max_tokens],
            "context": context
            })

    # Predict on a Pandas DataFrame.
    ret = loaded_model.predict(text_example)
    ret = ret.replace(orig_prompt, "")
    return ret

output = gen_text_for_serving("Hvis jeg får korona og isolerer meg selv og det ikke er alvorlig, er det noen medisiner jeg kan ta? Svar på norsk.", temperature=0.5, max_tokens=100)
print("output:", output)

# COMMAND ----------

# With context, does not seem to work
output = gen_text_for_serving("Hvis jeg får korona og isolerer meg selv og det ikke er alvorlig, er det noen medisiner jeg kan ta? Svar på norsk.", temperature=0.5, max_tokens=100, context="Hvis jeg får korona kan jeg legge meg på loftet.")
print("output", output)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Serve with Flask

# COMMAND ----------

from flask import Flask, jsonify, request

app = Flask("mistral-7b-instruct")

@app.route('/', methods=['POST'])
def serve_mistral_7b_instruct():
  resp = gen_text_for_serving(**request.json)
  return jsonify(resp)

# COMMAND ----------

from dbruntime.databricks_repl_context import get_context
ctx = get_context()

port = "7271"
driver_proxy_api = f"https://{ctx.browserHostName}/driver-proxy-api/o/0/{ctx.clusterId}/{port}"

print(f"""
driver_proxy_api = '{driver_proxy_api}'
cluster_id = '{ctx.clusterId}'
port = {port}
""")



# COMMAND ----------

# MAGIC %md
# MAGIC Store cluster vars as constants to retrieve in other notebooks

# COMMAND ----------

# Create table in the metastore

server_num = 1
constants_table = f"training.llm_langchain_shared.storting250_server{server_num}_constants"
# DeltaTable.createIfNotExists(spark) \
#   .tableName(constants_table) \
#   .addColumn("key", "STRING") \
#   .addColumn("val", "STRING")\
#   .execute()

catalog = "training"

spark.sql(f"""
CREATE CATALOG IF NOT EXISTS {catalog};
""")


schema = "training.llm_langchain_shared"
# Grant select and modify permissions for the table to all users on the account.
# This also works for other account-level groups and individual users.
spark.sql(f"""
CREATE SCHEMA IF NOT EXISTS {schema};
""")

spark.sql(f"""DROP TABLE IF EXISTS {constants_table}""")
          
spark.sql(f"""
CREATE TABLE IF NOT EXISTS {constants_table}
  (
    name STRING,
    var STRING
  )""")


# Grant select and modify permissions for the table to all users on the account.
# This also works for other account-level groups and individual users.
spark.sql(f"""
  GRANT SELECT
  ON TABLE {constants_table}
  TO `account users`""")

# Set ownership of table to training group so all training users can recreate these credentials
spark.sql(f"""
ALTER TABLE {constants_table} SET OWNER TO `academy-23-24`;""")

# COMMAND ----------

# Parse out host name
from urllib.parse import urlparse
host = urlparse(driver_proxy_api).netloc
print(host) # --> www.example.test

# COMMAND ----------

api_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

# COMMAND ----------

from pyspark.sql import Row
api_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
constants = [
    Row("cluster_id", ctx.clusterId),
    Row("port", port),
    Row("driver_proxy_api", driver_proxy_api),
    Row("host", host),
    Row("api_token", api_token),
]
constants_df = spark.createDataFrame(constants)
constants_df.write.insertInto(constants_table, overwrite=True)

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC Keep `app.run` running, and it could be used with Langchain ([documentation](https://python.langchain.com/docs/modules/model_io/models/llms/integrations/databricks.html#wrapping-a-cluster-driver-proxy-app)), or by call the serving endpoint with:
# MAGIC ```python
# MAGIC import requests
# MAGIC import json
# MAGIC
# MAGIC def request_mistral_7b(prompt, temperature=1.0, max_new_tokens=1024):
# MAGIC   token = ... # TODO: fill in with your Databricks personal access token that can access the cluster that runs this driver proxy notebook
# MAGIC   url = ...   # TODO: fill in with the driver_proxy_api output above
# MAGIC   
# MAGIC   headers = {
# MAGIC       "Content-Type": "application/json",
# MAGIC       "Authentication": f"Bearer {token}"
# MAGIC   }
# MAGIC   data = {
# MAGIC     "prompt": prompt,
# MAGIC     "temperature": temperature,
# MAGIC     "max_new_tokens": max_new_tokens,
# MAGIC   }
# MAGIC
# MAGIC   response = requests.post(url, headers=headers, data=json.dumps(data))
# MAGIC   return response.text
# MAGIC
# MAGIC
# MAGIC request_mistral_7b("What is databricks?")
# MAGIC ```
# MAGIC Or you could try using ai_query([documentation](https://docs.databricks.com/sql/language-manual/functions/ai_query.html)) to call this driver proxy from Databricks SQL with:
# MAGIC ```
# MAGIC SELECT ai_query('cluster_id:port', -- TODO: fill in the cluster_id and port number from output above.
# MAGIC   named_struct('prompt', 'What is databricks?', 'temperature', CAST(0.1 AS Double)),
# MAGIC   'returnType', 'STRING')
# MAGIC ```
# MAGIC Note: The [AI Functions](https://docs.databricks.com/large-language-models/ai-functions.html) is in the public preview, to enable the feature for your workspace, please submit this [form](https://docs.google.com/forms/d/e/1FAIpQLScVyh5eRioqGwuUVxj9JOiKBAo0-FWi7L3f4QWsKeyldqEw8w/viewform).

# COMMAND ----------

app.run(host="0.0.0.0", port=port, debug=True, use_reloader=False)

# COMMAND ----------

# MAGIC %ls -lh /Volumes/training/data/tunedmodels/parliament/

# COMMAND ----------


