from google.cloud import bigquery
from google.cloud import bigquery_storage_v1beta1

bq_client = bigquery.Client(
    project="ai-roadmap-new",
)

bq_storage_client = bigquery_storage_v1beta1.BigQueryStorageClient(
)

query_string = """
    SELECT
      is_male,
      gestation_weeks,
      mother_age,
      mother_race,
      weight_pounds,
      CASE WHEN weight_pounds > 5.5 THEN "NORMAL" ELSE "LOW" END AS weight_cat,
    FROM
      `bigquery-public-data.samples.natality`
    WHERE
      weight_pounds IS NOT NULL
      and year >= 2000
      and mother_race is not null
    LIMIT
        100
"""

data_frame = (
    bq_client.query(query_string)
    .result()
    .to_dataframe(bqstorage_client=bq_storage_client)
)
print(data_frame.head())
