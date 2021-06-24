"""
This script loads the necessary data from the different data sources and formats data
so that it can be fed to the ML algorith. More precisely, this script performs the following tasks:

1.  In function "prepare_data", the KPIs from the DataWarehouse (DW) and the sensor data
    from the .csv files is loaded, with functions "read_DW" and "read_sensor" respectively.
2.  This data is labeled, with the help of "label_data" as either having
    an unscheduled maintenance (1) or not having any maintenance (0) in the next seven days.
    This step involves joining data from the AMOS database.
3.  Finally, each row in the DataFram containing the labaled data is converted to
    LabeledPoint format (or LibSVM format) in order to feed it to the ML algorithm.
"""

import pyspark.sql.functions as f  # split, substring
from pyspark.mllib.util import MLUtils # saveAsLibSVMFile
from pyspark.mllib.regression import LabeledPoint
from pyspark.sql.functions import lit, datediff, to_date, col
import time


# Reads the sensor measurements from the .csv files in dir "filename"
# and returns their daily averages
def read_sensor(spark):
    filename = "./resources/trainingData/*.csv" #path to data directory
    # select files of aircraft A
    sensor = ( spark.read
    .option("header", "True")
    .option("delimiter",";")
    .csv(filename)
    .drop("series") )

    sensor = ( sensor
               .withColumn("filename", f.input_file_name()) # retrieve aircraft id
               .withColumn('aircraft', f.substring('filename', -10, 6))
               .drop("filename")
               .withColumn("valueNum", sensor["value"].cast("float")) # Change type of column "value" to float
               .drop("value") )

    # keep only YYYY-MM-DD from "date" columne
    split_date = f.split(sensor.date, " ")
    sensor = ( sensor
    .withColumn("dateid", split_date.getItem(0))
    .drop("date") )

    # calculate average measurment for aircraft and day
    avg = ( sensor
    .groupBy(["dateid", "aircraft"]).avg("valueNum")
    .withColumnRenamed("avg(valueNum)", "value") )
    return avg


# Retrieves the KPIs for each aircraft
def read_DW(spark):
    username = "MyUsername"
    password = "MyPassword"

    DW = ( spark.read
    .format("jdbc")
    .option("driver","org.postgresql.Driver")
    .option("url", "jdbc:postgresql://postgresfib.fib.upc.edu:6433/DW?sslmode=require")
    .option("dbtable", "public.aircraftutilization")
    .option("user", username)
    .option("password", password)
    .load()
    .select( "aircraftid", "timeid", "flighthours",  # keep useful columns
             "flightcycles", "delayedminutes", "scheduledoutofservice") )

    return DW


# Retrieves the necessary columns from AMOS database
def read_AMOS(spark):
    username = "MyUsername"
    password = "MyPassword"

    AMOS = ( spark.read
    .format("jdbc")
    .option("driver","org.postgresql.Driver")
    .option("url", "jdbc:postgresql://postgresfib.fib.upc.edu:6433/AMOS?sslmode=require")
    .option("dbtable", "oldinstance.maintenanceevents")
    .option("user", username)
    .option("password", password)
    .load()
    .select( "aircraftregistration", "starttime", "kind") )  # keep useful columns

    # keep rows for maintenance
    AMOS = ( AMOS
             .filter(AMOS.kind == "Maintenance")
             .drop("kind") )

    # keep only YYYY-MM-DD from "starttime" columne
    split_date = f.split(AMOS.starttime, " ")
    AMOS = ( AMOS
    .withColumn("date", split_date.getItem(0))
    .drop("starttime") )

    return AMOS


# If a row in data has a corresponding row in maintenanceevents (starttime within 7 days of dateid)
# then tag it with a 1 in data, otherwise with a 0
def label_data(spark, data):
    maint_ev = read_AMOS(spark)

    label1 = (data
    .join(maint_ev, (datediff( to_date(col("date")), to_date(col("dateid")) ) >= 0)
                  & (datediff( to_date(col("date")), to_date(col("dateid")) ) <= 6)
                  & (data.aircraft == maint_ev.aircraftregistration), 'left_semi')
    .drop("date").drop("dateid")
    .drop("aircraft").drop("aircraftregistration")
    .withColumn("label", lit(1)) )

    label0 = (data
    .join(maint_ev, (datediff( to_date(col("date")), to_date(col("dateid")) ) >= 0)
                  & (datediff( to_date(col("date")), to_date(col("dateid")) ) <= 6)
                  & (data.aircraft == maint_ev.aircraftregistration), 'left_anti')
    .drop("date").drop("dateid")
    .drop("aircraft").drop("aircraftregistration")
    .withColumn("label", lit(0)) )

    data = ( label1.union(label0)
            .select("label", "flighthours", "flightcycles", "delayedminutes", "value") )
    return data


# loads data from the datawarehouse and AMOS database, join the data, selects and typecasts columns
def prepare_data(spark):
    # load from the databases
    sensor = read_sensor(spark)
    KPIs = read_DW(spark)

    # join by date and aircraft
    # inner join so that if a date has no row in either "sensor" or "KPIs"
    # then this date is eliminated
    data = ( sensor
    .join(KPIs, [sensor.dateid ==  KPIs.timeid, sensor.aircraft == KPIs.aircraftid], "inner")
    .select("dateid", "aircraft", "scheduledoutofservice", "flighthours",
            "flightcycles", "delayedminutes", "value") )

    # typecast columns
    data = ( data.withColumn("scheduledoutofservice",data["scheduledoutofservice"].cast('float'))
                 .withColumn("flighthours",data["flighthours"].cast('float'))
                 .withColumn("flightcycles",data["flightcycles"].cast('float'))
                 .withColumn("delayedminutes",data["delayedminutes"].cast('float')) )

    # remove rows that are scheduled to be out of service in the next seven days
    # remove scheduledoutofservice column
    data = ( data
    .filter(data.scheduledoutofservice == 0)
    .drop("scheduledoutofservice") )

    # put a label on each row (0: no unscheduled maintenance, 1: scheduled maintenance)
    data = label_data(spark, data)
    return data


# arranges the contents of "data into LabeledPoint format and loads them onto a file in directory "data"
# returns the time of execution (exec_time) so that later the file can be reached
def to_libSVM(data):
    # rows to LabeledPoint
    data = data.rdd.map(lambda line: LabeledPoint(line[0], line[1:]))

    # load onto file
    exec_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
    path = "./data/{}".format(exec_time)
    MLUtils.saveAsLibSVMFile(data, path)

    return(exec_time)


def run(spark):
    data = prepare_data(spark)
    return to_libSVM(data)
