# imports and general configs
from pyspark.sql import SparkSession
from pyspark.sql.functions import isnan, when, count, col
from pyspark.ml.feature import VectorAssembler, OneHotEncoder
from pyspark.ml import Pipeline
from pyspark.ml.feature import StandardScaler
from pyspark.ml.feature import BucketedRandomProjectionLSH
import time
import random

if __name__ == '__main__':
    DATALAKE_PATH = 's3://datalake-sandbox'

    spark = SparkSession.builder.appName('tcc').getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    

    # load the data
    ## load des y3 gold data
    data_path = f'{DATALAKE_PATH}/tidbalma_tcc/des_data/'
    raw_df = spark.read.csv(data_path, header=True, inferSchema=True)

    print('Shape of DES data:', raw_df.count(), len(raw_df.columns)) # (11670190, 370)


    ## load tanoglidis' lsbg data
    lsbgs_path = f'{DATALAKE_PATH}/tidbalma_tcc/LSBG_catalog_v2.csv'
    tanoglidis_lsbgs = spark.read.csv(lsbgs_path, header=True, inferSchema=True)

    print('Shape of Tanoglidis LSBG data:', tanoglidis_lsbgs.count(), len(tanoglidis_lsbgs.columns)) # (23790, 45)


    ## check how many of tanoglidis' lsbgs we have in our data
    lsbgs = raw_df.join(tanoglidis_lsbgs,'coadd_object_id', 'leftsemi')

    print('Matched LSBGS:',  lsbgs.count(), len(lsbgs.columns)) # (18685, 370)


    ## load artefact data (non-lsbgs)
    artifacts1_path = f'{DATALAKE_PATH}/tidbalma_tcc/random_negative_all_1.txt'
    tanoglidis_artifacts1 = spark.read.csv(artifacts1_path, header=True, inferSchema=True)

    artifacts2_path = f'{DATALAKE_PATH}/tidbalma_tcc/random_negative_all_2.txt'
    tanoglidis_artifacts2 = spark.read.csv(artifacts2_path, header=True, inferSchema=True)

    print('Shape of artifacts 1 data:', tanoglidis_artifacts1.count(), len(tanoglidis_artifacts1.columns)) # (20000, 24)
    print('Shape of artifacts 2 data:', tanoglidis_artifacts2.count(), len(tanoglidis_artifacts2.columns)) # (20000, 24)


    ## check how many of these artifacts we have in our data
    artifacts1 = raw_df.join(tanoglidis_artifacts1,'coadd_object_id', 'leftsemi')
    print('Matched artifacts 1:',  artifacts1.count(), len(artifacts1.columns)) # (4790, 370)

    artifacts2 = raw_df.join(tanoglidis_artifacts2,'coadd_object_id', 'leftsemi')
    print('Matched artifacts 2:',  artifacts2.count(), len(artifacts2.columns)) # (7241, 370)


    # preprocess the data
    ## check for null values
    nulls = raw_df.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in raw_df.columns])
    print('Are there NULLs?', nulls.toPandas().isna().any().any()) #  False


    ## drop flags columns
    cols_to_drop = tuple([i for i in raw_df.columns if i.split('_')[0] == 'FLAGS' or
                                                    i.split('_')[0] == 'SEXTRACTOR' or
                                                    i.split('_')[0] == 'IMAFLAGS' or 
                                                    i == 'MOF_FLAGS' or
                                                    i == 'SOF_FLAGS'])

    # cols_to_drop = tuple([i for i in raw_df.columns if i.split('_')[0] == 'FLAGS' or i.split('_')[0] == 'SEXTRACTOR' or i.split('_')[0] == 'IMAFLAGS' or i == 'MOF_FLAGS' or i == 'SOF_FLAGS'])

    df = raw_df.drop(*cols_to_drop)
    lsbgs = lsbgs.drop(*cols_to_drop)
    artifacts1 = artifacts1.drop(*cols_to_drop)
    artifacts2 = artifacts2.drop(*cols_to_drop)

    print('Shape of DES data after dropping columns:', df.count(), len(df.columns)) # (11670190, 354)


    ## feature engineering: get feature columns and process them
    coadd_id = df.columns[0]
    ra_dec = df.columns[1:3]
    features = df.columns[3:]

    print('Number of features for training/inference:', len(features)) # 351


    ## create one-hot encoder model for categorical columns
    categorical_columns = ['EXTENDED_CLASS_COADD', # 0 = hi-con stars; 1 = candidate stars; 2 = candidate galaxies; 3  = hi-con galaxies; -9 = NO DATA
                        'EXTENDED_CLASS_MASH_SOF', # same as above
                        'EXTENDED_CLASS_MASH_MOF', # same as above
                        'EXTENDED_CLASS_MOF', # same as above
                        'EXTENDED_CLASS_SOF', # same as above
                        'EXTENDED_CLASS_WAVG' # same as above
                        ]

    #categorical_columns = ['EXTENDED_CLASS_COADD', 'EXTENDED_CLASS_MASH_SOF', 'EXTENDED_CLASS_MASH_MOF', 'EXTENDED_CLASS_MOF', 'EXTENDED_CLASS_SOF', 'EXTENDED_CLASS_WAVG']

    categorical_columns_dummy = []
    for c in categorical_columns:
        categorical_columns_dummy.append(c + "_ohe")

    encoders = [OneHotEncoder(inputCol=x, outputCol=y) for x,y in zip(categorical_columns, categorical_columns_dummy)]

    print('Number of categorical columns:', len(categorical_columns)) # 6

    numerical_columns = [x for x in features if x not in categorical_columns]

    print('Number of numerical columns:', len(numerical_columns)) # 345


    ## create vector assembler model to transform columns to spark's format
    columns_assembler = numerical_columns + categorical_columns_dummy
    assembler = VectorAssembler(inputCols=columns_assembler, outputCol='features')


    ## create rescaler model
    scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", withStd=True, withMean=True)


    ## create pipeline that applies all pre-processing steps to our data, making the `data` variable to be used by our model
    preprocessing = []
    preprocessing.extend(encoders)
    preprocessing.append(assembler)
    preprocessing.append(scaler)

    pipeline = Pipeline(stages=preprocessing)
    
    start = time.time()
    pipeline_model = pipeline.fit(df)
    end = time.time()
    print('Time to preprocess dataset:', end - start) # 577

    pipeline_model.save(f'{DATALAKE_PATH}/tidbalma_tcc/models/pipeline')

    data = pipeline_model.transform(df).select('COADD_OBJECT_ID', 'RA', 'DEC', 'scaledFeatures')
    print('Data:')
    print(data.show(3))


    # train the model
    brp = BucketedRandomProjectionLSH(inputCol="scaledFeatures", outputCol="hashes", bucketLength=2.0, numHashTables=3)
    
    start = time.time()
    model = brp.fit(data)
    end = time.time()
    print('Time to fit the model:', end - start) # 0.3
    model.save(f'{DATALAKE_PATH}/tidbalma_tcc/models/brp_model')

    # evaluate the model
    ## apply pre-processing pipeline to tanoglidis' lsbgs
    data_lsbgs = pipeline_model.transform(lsbgs).select('COADD_OBJECT_ID', 'RA', 'DEC', 'scaledFeatures')


    ## get random lsbg keys from tanoglidis' catalog
    amount_data_lsbgs = data_lsbgs.count()
    keys = random.sample(range(0, amount_data_lsbgs), 10)

    for k in keys:
        key = data_lsbgs.collect()[k]['scaledFeatures']

        start = time.time()
        closest_lsbgs_to_key = model.approxNearestNeighbors(data, key, 25001).select('COADD_OBJECT_ID', 'RA', 'DEC', 'distCol')
        end = time.time()
        print('Time to find 25000 nearest neighbors:', end - start)

        closest_lsbgs_to_key.write.csv(f'{DATALAKE_PATH}/tidbalma_tcc/results/closest_lsbgs_to_key-{k}.csv')

        del closest_lsbgs_to_key


    ## apply pre-processing pipeline to tanoglidis' artifacts and find nearest objects to random artefact keys
    data_artifacts1 = pipeline_model.transform(artifacts1).select('COADD_OBJECT_ID', 'RA', 'DEC', 'scaledFeatures')
    amount_data_artifacts1 = data_artifacts1.count()
    keys = random.sample(range(0, amount_data_artifacts1), 5)

    for k in keys:
        key = data_artifacts1.collect()[k]['scaledFeatures']

        closest_artifacts1_to_key = model.approxNearestNeighbors(data, key, 25001).select('COADD_OBJECT_ID', 'RA', 'DEC', 'distCol')
        closest_artifacts1_to_key.write.csv(f'{DATALAKE_PATH}/tidbalma_tcc/results/closest_artifacts1_to_key-{k}.csv')

        del closest_artifacts1_to_key

    data_artifacts2 = pipeline_model.transform(artifacts2).select('COADD_OBJECT_ID', 'RA', 'DEC', 'scaledFeatures')
    amount_data_artifacts2 = data_artifacts2.count()
    keys = random.sample(range(0, amount_data_artifacts2), 5)

    for k in keys:
        key = data_artifacts2.collect()[k]['scaledFeatures']

        closest_artifacts2_to_key = model.approxNearestNeighbors(data, key, 25001).select('COADD_OBJECT_ID', 'RA', 'DEC', 'distCol')
        closest_artifacts2_to_key.write.csv(f'{DATALAKE_PATH}/tidbalma_tcc/results/closest_artifacts2_to_key-{k}.csv')

        del closest_artifacts2_to_key

    spark.stop()
