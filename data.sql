SELECT
  is_male,
  gestation_weeks,
  mother_age,
  mother_race,
  plurality,
  CAST (cigarette_use as int64) as cigarette_use,
  weight_pounds,
  CASE
    WHEN weight_pounds > 5.5 and weight_pounds < 9 THEN "NORMAL"
    WHEN weight_pounds >= 9 THEN "HIGH"
  ELSE
    "LOW"
  END
  AS weight_cat
FROM
  `bigquery-public-data.samples.natality`
WHERE
  weight_pounds < 9
  AND year >= 2003
  AND mother_race IS NOT NULL
  AND cigarette_use IS NOT NULL
LIMIT
  500000