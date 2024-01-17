# Databricks notebook source
# MAGIC %md
# MAGIC # Serve fine tuned Mistral-7B with QLORA
# MAGIC
# MAGIC The [Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1) Large Language Model (LLM) is a pretrained generative text model with 7 billion parameters. Mistral-7B-v0.1 outperforms Llama 2 13B on all benchmarks.
# MAGIC
# MAGIC This notebook is to fine-tune [Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1) models on the [mosaicml/dolly_hhrlhf](https://huggingface.co/datasets/mosaicml/dolly_hhrlhf) dataset.
# MAGIC
# MAGIC Environment for this notebook:
# MAGIC - Runtime: 14.0 GPU ML Runtime
# MAGIC - Instance: `g5.xlarge` on AWS, `Standard_NV36ads_A10_v5` on Azure
# MAGIC
# MAGIC We leverage the PEFT library from Hugging Face, as well as QLoRA for more memory efficient finetuning.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install required packages
# MAGIC
# MAGIC Run the cells below to setup and install the required libraries. For our experiment we will need `accelerate`, `peft`, `transformers`, `datasets` and TRL to leverage the recent [`SFTTrainer`](https://huggingface.co/docs/trl/main/en/sft_trainer). We will use `bitsandbytes` to [quantize the base model into 4bit](https://huggingface.co/blog/4bit-transformers-bitsandbytes). We will also install `einops` as it is a requirement to load Falcon models.

# COMMAND ----------

# %pip install git+https://github.com/huggingface/peft.git
# %pip install torch==2.1.0 accelerate==0.23.0
%pip install -U transformers==4.34.0
%pip install bitsandbytes==0.41.1 einops==0.7.0 trl==0.7.1 peft==0.5.0
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load the fine tuned model from MLFlow

# COMMAND ----------

# MAGIC %md
# MAGIC Run model inference with the model logged in MLFlow.

# COMMAND ----------

mlflowmodel_name = "norsk7bqloramistral"
run_id = "4ea2ed325d9644898350feea93d4f5c8"
logged_model = f"runs:/{run_id}/{mlflowmodel_name}"
print(f"logged_model: {logged_model}")

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

logged_model = "/Volumes/training/data/tunedmodels/parliament/"
loaded_model = mlflow.pyfunc.load_model(logged_model)
print(f"logged_model: {logged_model}")

text_example=pd.DataFrame({
            "prompt":[prompt],
            "temperature": [0.5],
            "max_tokens": [100]})

# Predict on a Pandas DataFrame.
loaded_model.predict(text_example)

# COMMAND ----------

# MAGIC %md
# MAGIC ## from pretrained example
# MAGIC https://mlflow.org/docs/latest/_modules/mlflow/transformers.html

# COMMAND ----------


