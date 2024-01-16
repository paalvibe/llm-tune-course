# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Serving Llama-2-13b-chat-hf via vllm with a cluster driver proxy app
# MAGIC
# MAGIC [Llama 2](https://huggingface.co/meta-llama) is a collection of pretrained and fine-tuned generative text models ranging in scale from 7 billion to 70 billion parameters. It is trained with 2T tokens and supports context length window upto 4K tokens. [Llama-2-13b-chat-hf](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf) is the 13B fine-tuned model, optimized for dialogue use cases and converted for the Hugging Face Transformers format.
# MAGIC
# MAGIC For reference: https://github.com/databricks/databricks-ml-examples/blob/master/llm-models/llamav2/llamav2-13b/01_load_inference.py
# MAGIC
# MAGIC
# MAGIC [vllm](https://github.com/vllm-project/vllm/tree/main) is an open-source library that makes LLM inference fast with various optimizations.
# MAGIC
# MAGIC Environment for this notebook:
# MAGIC - Runtime: 13.2 GPU ML Runtime
# MAGIC - Instance: `g5.12xlarge` on AWS, `Standard_NV72ads_A10_v5` or `Standard_NC24ads_A100_v4` on Azure
# MAGIC GPU instances that have at least 2 A10 GPUs would be enough for inference on single input (batch inference requires slightly more memory).
# MAGIC
# MAGIC Requirements:
# MAGIC   - To get the access of the model on HuggingFace, please visit the [Meta website](https://ai.meta.com/resources/models-and-libraries/llama-downloads) and accept our license terms and acceptable use policy before submitting this form. Requests will be processed in 1-2 days.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install python libs

# COMMAND ----------

# MAGIC %pip install -U vllm==0.2.0 transformers==4.34.0 accelerate==0.20.3
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Inference
# MAGIC The example in the model card should also work on Databricks with the same environment.

# COMMAND ----------

from vllm import LLM

# it is suggested to pin the revision commit hash and not change it for reproducibility because the uploader might change the model afterwards; you can find the commmit history of Mistral-7B-Instruct-v0. in https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1/commits/main
model = "meta-llama/Llama-2-13b-chat-hf"
revision = "c2f3ec81aac798ae26dcc57799a994dfbf521496"

llm = LLM(model=model, revision=revision)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Add instruction prompt with sys section as guard rail

# COMMAND ----------

from vllm import SamplingParams

# Prompt templates as follows could guide the model to follow instructions and respond to the input, and empirically it turns out to make Falcon models produce better responses
DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information. You should answer in Norwegian language."""

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

# Define parameters to generate text
def gen_text_for_serving(prompt, **kwargs):
    prompt = PROMPT_FOR_GENERATION_FORMAT.format(instruction=prompt)

    # the default max length is pretty small (20), which would cut the generated output in the middle, so it's necessary to increase the threshold to the complete response
    if "max_tokens" not in kwargs:
        kwargs["max_tokens"] = 512

    sampling_params = SamplingParams(**kwargs)

    outputs = llm.generate(prompt, sampling_params=sampling_params)
    texts = [out.outputs[0].text for out in outputs]

    return texts[0]

# COMMAND ----------

print(gen_text_for_serving("Hvordan kan jeg lære Python på 3 dager?"))

# COMMAND ----------

# See full list of configurable args: https://github.com/vllm-project/vllm/blob/main/vllm/sampling_params.py
print(gen_text_for_serving("Hvordan kan jeg lære Python på 3 dager?", temperature=0.1, max_tokens=100))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Serve with Flask

# COMMAND ----------

from flask import Flask, jsonify, request

app = Flask("llama-2-13b-chat-hf")

@app.route('/', methods=['POST'])
def serve_llama2():
  resp = gen_text_for_serving(**request.json)
  return jsonify(resp)

# COMMAND ----------

from dbruntime.databricks_repl_context import get_context
ctx = get_context()

port = "7778"
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
constants_table = "training.llm_langchain_shared.serve1_llama_constants"
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


