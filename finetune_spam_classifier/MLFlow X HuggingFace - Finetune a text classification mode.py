# Databricks notebook source
# MAGIC %md
# MAGIC # Finetune a text classification model
# MAGIC In this notebook, we will show how to finetune a `DistilledBert` model to classify SMS as spam or not. We will also show how you can use MLFlow to track and monitor your finetuning, and register and deploy the finetuned model with MLFlow.
# MAGIC
# MAGIC In this guide we will load the SMS Spam Collection dataset from [DBFS](https://docs.databricks.com/dbfs/index.html) to show a full lifecycle of finetuning with Spark. You can also skip the DBFS part by directly loading SMS Spam Collection dataset from HuggingFace: [link](https://huggingface.co/datasets/sms_spam)
# MAGIC
# MAGIC ## Cluster setup
# MAGIC For this notebook, we recommend a single GPU cluster, such as a `g4dn.xlarge` on AWS or `Standard_NC4as_T4_v3` on Azure. You can [create a single machine cluster](https://docs.databricks.com/clusters/configure.html) using the personal compute policy or by choosing "Single Node" when creating a cluster. This notebook requires Databricks Runtime ML GPU version 11.1 or greater. 

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install dependencies
# MAGIC
# MAGIC We need `datasets` and `evaluate` package by Huggingface. Additionally, TF 2.13 has a conflict with transformers <= 4.28, so w need to downgrade to TF 2.12. 

# COMMAND ----------

!pip install -q datasets evaluate tensorflow==2.12.0

# COMMAND ----------

# MAGIC %md
# MAGIC Restart the python runtime to use the updated dependencies.

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## [Optional] Download data and copy to Databricks file system
# MAGIC Let's download and extract the dataset, we will use
# MAGIC [SMS Spam Collection Dataset](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection) from the UCI Machine Learning Repository.

# COMMAND ----------

# MAGIC %sh wget https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip

# COMMAND ----------

# MAGIC %sh unzip -o smsspamcollection.zip

# COMMAND ----------

# MAGIC %md
# MAGIC Get the working directory, we will need it in our next cell.

# COMMAND ----------

# MAGIC %md
# MAGIC Copy the dataset to Databricks file system (DBFS). The `tutorial_path` sets the path in DBFS that the notebook uses to write the sample dataset. It is deleted by the last command in this notebook.
# MAGIC
# MAGIC You can find the path to dataset by clicking on the triple dot next to `SMSSpamCollection` on the left sidebar, then Copy => Path.

# COMMAND ----------

tutorial_path = "/FileStore/sms_tutorial" 
dbutils.fs.mkdirs(f"dbfs:{tutorial_path}")
dbutils.fs.cp(
    "file:/Workspace/Repos/chen.qian@databricks.com/mlflow-guide/finetune_spam_classifier/SMSSpamCollection", 
    f"dbfs:{tutorial_path}/SMSSpamCollection.tsv",
)

# COMMAND ----------

# MAGIC %md
# MAGIC Now your data lives in DBFS, we can load the dataset into a DataFrame. The file is tab separated and does not contain a header, so we specify the separator using `sep` and specify the column names explicitly.

# COMMAND ----------

sms = spark.read.csv(
    f"{tutorial_path}/SMSSpamCollection.tsv", 
    header=False, 
    inferSchema=True, 
    sep="\t"
).toDF("label", "text")
print(f"Total number of data records: {sms.count()}")

# Print out some sample data.
display(sms.take(10))

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Convert string labels to integers, since finetuning requires an integer label.
# MAGIC
# MAGIC In this exact dataset, we have the following mapping:
# MAGIC ```
# MAGIC {
# MAGIC   "ham": 0,
# MAGIC   "spam": 1,
# MAGIC }
# MAGIC ```

# COMMAND ----------

id2label = {0: "ham", 1: "spam"}
label2id = {'ham': 0, 'spam': 1}

# COMMAND ----------

# MAGIC %md
# MAGIC Replace the string labels with the IDs in the DataFrame.

# COMMAND ----------

from pyspark.sql.functions import pandas_udf
import pandas as pd

# `pandas_udf` is the annotator that transforms a custom function
# into a udf, so we can call this function inside `select`.
@pandas_udf('integer')
def replace_labels_with_ids(labels: pd.Series) -> pd.Series:
  return labels.apply(lambda x: label2id[x])

sms_id_labels = sms.select(replace_labels_with_ids(sms.label).alias('label'), sms.text)
display(sms_id_labels.take(10))

# COMMAND ----------

# MAGIC %md
# MAGIC Now let's convert the dataframe into a HuggingFace dataset. HuggingFace supports loading from Spark DataFrames using `datasets.Dataset.from_spark`. See the Hugging Face documentation to learn more about the [from_spark()](https://huggingface.co/docs/datasets/use_with_spark) method. 
# MAGIC
# MAGIC Dataset.from_spark caches the dataset. In this example, the model is trained on the driver, and the cached data is parallelized using Spark, so `cache_dir` must be accessible to the driver and to all the workers. You can use the Databricks File System (DBFS) root([AWS](https://docs.databricks.com/dbfs/index.html#what-is-the-dbfs-root)| [Azure](https://learn.microsoft.com/azure/databricks/dbfs/#what-is-the-dbfs-root) |[GCP](https://docs.gcp.databricks.com/dbfs/index.html#what-is-the-dbfs-root)) or mount point ([AWS](https://docs.databricks.com/dbfs/mounts.html) | [Azure](https://learn.microsoft.com/azure/databricks/dbfs/mounts) | [GCP](https://docs.gcp.databricks.com/dbfs/mounts.html)). 
# MAGIC
# MAGIC By using DBFS, you can reference "local" paths when creating the `transformers` compatible datasets used for model training.

# COMMAND ----------

(train, test) = sms_id_labels.persist().randomSplit([0.8, 0.2])

import datasets
train_dataset = datasets.Dataset.from_spark(train, cache_dir="/dbfs/cache/train")
test_dataset = datasets.Dataset.from_spark(test, cache_dir="/dbfs/cache/test")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Alternative way to load dataset
# MAGIC
# MAGIC If you skip the previous step to load dataset from Spark, uncomment and run the command below to load dataset directly from HuggingFace.

# COMMAND ----------

from datasets import load_dataset

sms_dataset = load_dataset("sms_spam")
sms_train_test = sms_dataset["train"].train_test_split(test_size=0.2)
# For consistency, we rename "sms" => "text".
sms_train_test = sms_train_test.rename_column("sms", "text")
train_dataset = sms_train_test["train"]
test_dataset = sms_train_test["test"]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data preprocessing
# MAGIC
# MAGIC Before finetuning, let's tokenize and shuffle the datasets. Since the [Trainer](https://huggingface.co/docs/transformers/main/en/main_classes/trainer) does not need the untokenized `text` columns for training,
# MAGIC the notebook removes them from the dataset.
# MAGIC In this step, `datasets` also caches the transformed datasets on local disk for fast subsequent loading during model training.

# COMMAND ----------

from transformers import AutoTokenizer

# Load the tokenizer for "distilbert-base-uncased" model.
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
def tokenize_function(examples):
    # Pad/truncate each text to 512 tokens. Enforcing the same shape
    # could make the training faster.
    return tokenizer(
        examples["text"], 
        padding="max_length",
        truncation=True,
        max_length=128,
    )

train_tokenized = train_dataset.map(tokenize_function).remove_columns(["text"]).shuffle(seed=42)
test_tokenized = test_dataset.map(tokenize_function).remove_columns(["text"]).shuffle(seed=42)

# COMMAND ----------

# MAGIC %md
# MAGIC # Model finetuning
# MAGIC
# MAGIC We have prepared the data, let's kick off the finetuning!
# MAGIC
# MAGIC For finetuning we will rely on HuggingFace `Trainer` API.

# COMMAND ----------

# MAGIC %md
# MAGIC Create the evaluation metric to log. Loss is also logged, but adding other metrics such as accuracy can make modeling performance easier to understand. For classification task, we use `accuracy` as the tracking metric.

# COMMAND ----------

import numpy as np
import evaluate
metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# COMMAND ----------

# MAGIC %md
# MAGIC Set training arguments.
# MAGIC Please refer to [transformers documentation](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments) 
# MAGIC for the full arg list. Don't panick on the long list of args, usually we just need a few out of that.
# MAGIC
# MAGIC **Important: Note that you cannot set `training_output_dir` in the working directory due to the writing restriction, we recommend using some directory under `/tmp`.**

# COMMAND ----------

from transformers import TrainingArguments, Trainer

# Set the output directory to somewhere inside /tmp.
training_output_dir = "/tmp/weird_mouse/sms_trainer"
training_args = TrainingArguments(
    output_dir=training_output_dir, 
    evaluation_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
)

# COMMAND ----------

# MAGIC %md
# MAGIC Let's load the pretrained Distilled Bert model, and specify the label mappings and the number of classes.

# COMMAND ----------

from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", 
    num_labels=2, 
    label2id=label2id, 
    id2label=id2label,
)

# COMMAND ----------

# MAGIC %md
# MAGIC Construct the trainer object with the model, arguments, datasets, collator, and metrics created above.

# COMMAND ----------

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=test_tokenized,
    compute_metrics=compute_metrics,
)

# COMMAND ----------

# MAGIC %md
# MAGIC Train the model, meanwhile we log metrics and results to MLflow.

# COMMAND ----------

import mlflow

with mlflow.start_run() as run:
    trainer.train()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Let's wrap the model into a HuggingFace `text-classification` pipeline so that we can directly feed text data for spam classification.

# COMMAND ----------

from transformers import pipeline

pipe = pipeline(
    "text-classification", 
    model=trainer.model, 
    batch_size=8, 
    tokenizer=tokenizer,
    device=0,
)
sample_text = ("WINNER!! As a valued network customer you have been selected "
               "to receivea £900 prize reward! To claim call 09061701461. Claim "
               "code KL341. Valid 12 hours only.")
print("Model prediction: ", pipe(sample_text))

pipeline_output_dir = "/tmp/weird_mouse/sms_pipeline"
pipe.save_pretrained(pipeline_output_dir)

# COMMAND ----------

# MAGIC %md
# MAGIC You are almost there! Now let's convert the HuggingFace `text-classication` pipeline into a MLFlow model so that we can register and serve the model, and share it.

# COMMAND ----------

model_artifact_path = "sms_spam_model"

mlflow.transformers.log_model(
    transformers_model=pipe, 
    artifact_path=model_artifact_path, 
    input_example="Hi there!",
)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Click on the `experiment` link above, or click the "MLFlow Experiment" button on the sidebar to the right to open the MLFlow experiment. You will find the run id there, like the image below.
# MAGIC
# MAGIC ![MLflow experiment](/Workspace/Repos/chen.qian@databricks.com/mlflow-guide/mlflow_experiment.png)
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## That's almost it! 
# MAGIC
# MAGIC You have made it! You have learned how to finetune a spam classifier with HuggingFace and MLflow. You can continue reading if you use DBFS, otherwise that's all about the guide. Thanks for reading!

# COMMAND ----------

# MAGIC %md
# MAGIC ## Batch inference
# MAGIC
# MAGIC Load the MLflow model as a UDF and use it for batch scoring.

# COMMAND ----------

# You can find the identifier in the MLflow experiment page.
logged_model = 'runs:/1cffd782552d4baa9bb63558ce7dc450/sms_spam_model'

# Load model as a Spark UDF. Override result_type if the model does not return double values.
sms_spam_model_udf = mlflow.pyfunc.spark_udf(
    spark, 
    model_uri=logged_model, 
    result_type='string',
)

test = test.select(test.text, test.label, sms_spam_model_udf(test.text).alias("prediction"))
display(test)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cleanup
# MAGIC Remove the files placed in DBFS.

# COMMAND ----------

dbutils.fs.rm(f"dbfs:{tutorial_path}", recurse=True)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC That's it, thanks for reading!
