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

# MAGIC %pip install git+https://github.com/huggingface/peft.git

# COMMAND ----------

dbutils.library.restartPython()

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

import peft

# COMMAND ----------

import torch
# from peft import PeftModel, PeftConfig

class Mistral7BQLORANORSK(mlflow.pyfunc.PythonModel):
  def load_context(self, context):
    self.tokenizer = AutoTokenizer.from_pretrained(context.artifacts['repository'])
    self.tokenizer.pad_token = tokenizer.eos_token
    config = PeftConfig.from_pretrained(context.artifacts['lora'])
    base_model = AutoModelForCausalLM.from_pretrained(
      context.artifacts['repository'],
      return_dict=True,
      load_in_4bit=True,
      device_map={"":0},
      trust_remote_code=True,
    )
    self.model = PeftModel.from_pretrained(base_model, context.artifacts['lora'])

  def predict(self, context, model_input):
    prompt = model_input["prompt"][0]
    temperature = model_input.get("temperature", [1.0])[0]
    max_tokens = model_input.get("max_tokens", [100])[0]
    batch = self.tokenizer(prompt, padding=True, truncation=True,return_tensors='pt').to('cuda')
    with torch.cuda.amp.autocast():
      output_tokens = self.model.generate(
          input_ids = batch.input_ids,
          max_new_tokens=max_tokens,
          temperature=temperature,
          top_p=0.7,
          num_return_sequences=1,
          do_sample=True,
          pad_token_id=tokenizer.eos_token_id,
          eos_token_id=tokenizer.eos_token_id,
      )
    generated_text = self.tokenizer.decode(output_tokens[0], skip_special_tokens=True)

    return generated_text

# COMMAND ----------

# mlflow.set_registry_uri('databricks-uc')
# loaded_model = mlflow.pyfunc.load_model(model_uri=f"models:/{mlflowmodel_name}@2")
# loaded_model = mlflow.pyfunc.load_model("dbfs:/databricks/mlflow-tracking/b011c0bde56242718d3384806e37e7e7/15bd6be40e70461597f6fa785c8a684a/artifacts/norsk7bqloramistral250") # works
loaded_model = mlflow.pyfunc.load_model(artifacts_url)
# Make a prediction using the loaded model

# COMMAND ----------

loaded_model.predict(
    {"prompt": "Hva er rollen til stortinget?"},
    params={
        "temperature": 0.5,
        "max_new_tokens": 150,
    }
)

# COMMAND ----------

foo = bardoesntexist

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ## Load model from mlflow

# COMMAND ----------

# import mlflow
# class Mistral7BQLORANORSK(mlflow.pyfunc.PythonModel):
#   def load_context(self, context):
#     self.tokenizer = AutoTokenizer.from_pretrained(context.artifacts['repository'])
#     self.tokenizer.pad_token = tokenizer.eos_token
#     config = PeftConfig.from_pretrained(context.artifacts['lora'])
#     base_model = AutoModelForCausalLM.from_pretrained(
#       context.artifacts['repository'],
#       return_dict=True,
#       load_in_4bit=True,
#       device_map={"":0},
#       trust_remote_code=True,
#     )
#     self.model = PeftModel.from_pretrained(base_model, context.artifacts['lora'])

#   def predict(self, context, model_input):
#     prompt = model_input["prompt"][0]
#     temperature = model_input.get("temperature", [1.0])[0]
#     max_tokens = model_input.get("max_tokens", [100])[0]
#     batch = self.tokenizer(prompt, padding=True, truncation=True,return_tensors='pt').to('cuda')
#     with torch.cuda.amp.autocast():
#       output_tokens = self.model.generate(
#           input_ids = batch.input_ids,
#           max_new_tokens=max_tokens,
#           temperature=temperature,
#           top_p=0.7,
#           num_return_sequences=1,
#           do_sample=True,
#           pad_token_id=tokenizer.eos_token_id,
#           eos_token_id=tokenizer.eos_token_id,
#       )
#     generated_text = self.tokenizer.decode(output_tokens[0], skip_special_tokens=True)

#     return generated_text

# COMMAND ----------

loaded_model = mlflow.pyfunc.load_model(model_uri=f"models:/{mlflowmodel_name}/production")
# loaded_model = mlflow.pyfunc.load_model(model_uri=logged_model)

# COMMAND ----------

import mlflow
import pandas as pd


# Old Instruction:
# if one get corona and you are self isolating and it is not severe, is there any meds that one can take?

prompt = """Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
Hvis jeg får korona og isolerer meg selv og det ikke er alvorlig, er det noen medisiner jeg kan ta?

### Response: """
# Load model as a PyFuncModel.
# run_id = run.info.run_id
# logged_model = f"runs:/{run_id}/{mlflowmodel_name}"

# experiment_name = "/local_disk0/results-norskgpt-2-serve-experiment"
#experiment_name = "/Volumes/training/data/tunedmodels/serve-experiment"
experiment_name = "/Users/pal.de.vibe@knowit.no/storting250-serve-experiment"
# mlflow.set_experiment(experiment_name) 
with mlflow.start_run(run_name='tnet_e3nn_reg') as run:
# with mlflow.start_run() as run:
    loaded_model = mlflow.pyfunc.load_model(logged_model)
    print(f"logged_model: {logged_model}")

    text_example=pd.DataFrame({
            "prompt":[prompt],
            "temperature": [0.5],
            "max_tokens": [100]})

    # Predict on a Pandas DataFrame.
    loaded_model.predict(text_example)

# COMMAND ----------

loaded_model.predict(text_example)

# COMMAND ----------

# MAGIC %md ## How to serve mlflow model
# MAGIC https://github.com/databricks/databricks-ml-examples/blob/42b2903e36dc849899122c96f2ef3b89c1a14ac4/llm-models/mpt/mpt-7b/02_mlflow_logging_inference.py#L167

# COMMAND ----------

# MAGIC %md
# MAGIC Code showing example model:
# MAGIC ```
# MAGIC
# MAGIC output_schema = Schema([ColSpec(DataType.string)])
# MAGIC signature = ModelSignature(inputs=input_schema, outputs=output_schema)
# MAGIC
# MAGIC
# MAGIC # Define input and output schema
# MAGIC input_schema = Schema([
# MAGIC     ColSpec(DataType.string, "prompt"), 
# MAGIC     ColSpec(DataType.double, "temperature", optional=True), 
# MAGIC     ColSpec(DataType.long, "max_tokens", optional=True)])
# MAGIC output_schema = Schema([ColSpec(DataType.string)])
# MAGIC signature = ModelSignature(inputs=input_schema, outputs=output_schema)
# MAGIC
# MAGIC # Define input example
# MAGIC input_example=pd.DataFrame({
# MAGIC             "prompt":["what is ML?"], 
# MAGIC             "temperature": [0.5],
# MAGIC             "max_tokens": [100]})
# MAGIC
# MAGIC # Log the model with its details such as artifacts, pip requirements and input example
# MAGIC # This may take about 5 minutes to complete
# MAGIC torch_version = torch.__version__.split("+")[0]
# MAGIC with mlflow.start_run() as run:  
# MAGIC     mlflow.pyfunc.log_model(
# MAGIC         "model",
# MAGIC         python_model=MPT(),
# MAGIC         artifacts={'repository' : model_location},
# MAGIC         pip_requirements=[f"torch=={torch_version}", 
# MAGIC                           f"transformers=={transformers.__version__}", 
# MAGIC                           f"accelerate=={accelerate.__version__}", "einops", "sentencepiece"],
# MAGIC         input_example=input_example,
# MAGIC         signature=signature
# MAGIC     )
# MAGIC ```

# COMMAND ----------

# import mlflow
# import pandas as pd
# loaded_model = mlflow.pyfunc.load_model(f"models:/mpt-7b-instruct/latest")

# # Make a prediction using the loaded model
# input_example=pd.DataFrame({"prompt":["what is ML?", "Name 10 colors."], "temperature": [0.5, 0.2],"max_tokens": [100, 200]})
# print(loaded_model.predict(input_example))


# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

foo = nonexist

# COMMAND ----------

# import mlflow
# from langchain import LLMChain, PromptTemplate
# from langchain.llms import MlflowAIGateway

# gateway = MlflowAIGateway(
#     gateway_uri="http://127.0.0.1:5000",
#     route="completions",
#     params={
#         "temperature": 0.0,
#         "top_p": 0.1,
#     },
# )

# llm_chain = LLMChain(
#     llm=gateway,
#     prompt=PromptTemplate(
#         input_variables=["adjective"],
#         template="Tell me a {adjective} joke",
#     ),
# )
# result = llm_chain.run(adjective="funny")
# print(result)

# with mlflow.start_run():
#     model_info = mlflow.langchain.log_model(llm_chain, "model")

# model = mlflow.pyfunc.load_model(model_info.model_uri)
# print(model.predict([{"adjective": "funny"}]))a

# COMMAND ----------

# from huggingface_hub import login
# login(token=hf_token)

# COMMAND ----------

from vllm import LLM

# it is suggested to pin the revision commit hash and not change it for reproducibility because the uploader might change the model afterwards; you can find the commmit history of Mistral-7B-Instruct-v0. in https://huggingface.co/bineric/NorskGPT-Mistral-7b/commits/main
model = "bineric/NorskGPT-Mistral-7b"
revision = "198c803eeec43825fa0f9bb914b2e3d1f798b607"

llm = LLM(model=model, revision=revision, token="hf_token")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Add instruction prompt with sys section as guard rail

# COMMAND ----------

from vllm import SamplingParams

# Prompt templates as follows could guide the model to follow instructions and respond to the input, and empirically it turns out to make Falcon models produce better responses
DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

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

# import mlflow
# import pandas as pd


# # Old Instruction:
# # if one get corona and you are self isolating and it is not severe, is there any meds that one can take?

# prompt = """Below is an instruction that describes a task. Write a response that appropriately completes the request.
# ### Instruction:
# Hvis jeg får korona og isolerer meg selv og det ikke er alvorlig, er det noen medisiner jeg kan ta?

# ### Response: """
# # Load model as a PyFuncModel.
# run_id = run.info.run_id
# # logged_model = f"runs:/{run_id}/{mlflowmodel_name}"

# loaded_model = mlflow.pyfunc.load_model(logged_model)
# print(f"logged_model: {logged_model}")

# text_example=pd.DataFrame({
#             "prompt":[prompt],
#             "temperature": [0.5],
#             "max_tokens": [100]})

# # Predict on a Pandas DataFrame.
# loaded_model.predict(text_example)

# COMMAND ----------

print(gen_text_for_serving("How to master Python in 3 days?"))

# COMMAND ----------

# See full list of configurable args: https://github.com/vllm-project/vllm/blob/main/vllm/sampling_params.py
print(gen_text_for_serving("How to master Python in 3 days?", temperature=0.1, max_tokens=100))

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


