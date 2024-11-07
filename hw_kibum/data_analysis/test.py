from pyspark import SparkContext
sc = SparkContext.getOrCreate()
print(sc.version)