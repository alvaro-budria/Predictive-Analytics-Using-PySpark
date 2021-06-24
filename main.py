"""

This script controls the execution of the three pipelines (Data_Management, Data_Analysis and Run_Time_Classifier).

The expected input is an aircaft registration code (XX-XXX) and a valid date (ddmmyy).

The output is, for the given aircraft and date, if this aicraft will undergo unsheduled maintenance
in the next seven days.

"""

import os
import sys
import pyspark
from pyspark import SparkConf
from pyspark.sql import SparkSession

import Data_Management
import Data_Analysis
import Run_Time_Classifier

HADOOP_HOME = "./resources/hadoop_home"
JDBC_JAR = "./resources/postgresql-42.2.8.jar"
PYSPARK_PYTHON = "python" # here use your own environment variable
PYSPARK_DRIVER_PYTHON = "python"

if(__name__== "__main__"):
    os.environ["HADOOP_HOME"] = HADOOP_HOME
    sys.path.append(HADOOP_HOME + "\\bin")
    os.environ["PYSPARK_PYTHON"] = PYSPARK_PYTHON
    os.environ["PYSPARK_DRIVER_PYTHON"] = PYSPARK_DRIVER_PYTHON

    conf = SparkConf()  # create the configuration
    conf.set("spark.jars", JDBC_JAR)

    spark = SparkSession.builder \
        .config(conf=conf) \
        .master("local") \
        .appName("Training") \
        .getOrCreate()

    sc = pyspark.SparkContext.getOrCreate()

    if len(sys.argv) < 3:
        print("Error: not enough data provided. Please specify an aircraft (XX-XXX) and a date (ddmmyy)")

    else:
        exec_time = Data_Management.run(spark)
        [model, accuracy, precision, recall] = Data_Analysis.run(spark, exec_time)
        A = sys.argv[1]
        date = sys.argv[2]
        pred = Run_Time_Classifier.run(spark, model, sys.argv[1], sys.argv[2])
        if pred == 1:
            print("The aircraft is predicted to have at least one unscheduled maintenance in the next 7 days.")
        else: # pred == 0
            print("The aircraft is predicted to have no unscheduled maintenance in the next 7 days.")
