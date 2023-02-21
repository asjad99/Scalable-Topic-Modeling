from __future__ import print_function

# $example on$
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
# $example off$
from pyspark.sql import SparkSession
from pyspark.ml.feature import StopWordsRemover
from pyspark.ml.clustering import LDA


if __name__ == "__main__":
    
    spark = SparkSession\
        .builder\
        .appName("TfIdfExample")\
        .getOrCreate()

    # $example on$
    sentenceData = spark.createDataFrame([
        (0.0, "Hi I heard about Spark"),
        (1.0, "I wish Java could use case classes"),
        (2.0, "Logistic regression models are neat")
    ], ["id", "raw_sentence"])

    #---------------------------------------------------------------------
    #tokenize 
    #---------------------------------------------------------------------

    tokenizer = Tokenizer(inputCol="raw_sentence", outputCol="words")
    wordsData = tokenizer.transform(sentenceData)

    #---------------------------------------------------------------------
    #remove stop words 
    #---------------------------------------------------------------------

    remover = StopWordsRemover(inputCol="words", outputCol="filtered")
    wordsData_updated = remover.transform(wordsData)

    #---------------------------------------------------------------------
    #TF-IDF conversion 
    #---------------------------------------------------------------------

    hashingTF = HashingTF(inputCol="filtered", outputCol="rawFeatures", numFeatures=20)
    featurizedData = hashingTF.transform(wordsData_updated)
    # alternatively, CountVectorizer can also be used to get term frequency vectors

    idf = IDF(inputCol="rawFeatures", outputCol="features")
    idfModel = idf.fit(featurizedData)
    rescaledData = idfModel.transform(featurizedData)

    rescaledData.select("id", "features").show(truncate=False)
    # $example off$

    #---------------------------------------------------------------------
    #create a LDA Model
    #---------------------------------------------------------------------
    lda = LDA(k=10, maxIter=10)
    model = lda.fit(rescaledData)
    ll = model.logLikelihood(rescaledData)
    lp = model.logPerplexity(rescaledData)
    print("The lower bound on the log likelihood of the entire corpus: " + str(ll))
    print("The upper bound bound on perplexity: " + str(lp))
    topics = model.describeTopics(3)
    print("The topics described by their top-weighted terms:")
    topics.show(truncate=False)

    #show the results 

    transformed = model.transform(dataset)
    transformed.show(truncate=False)
    
    spark.stop()