---
pretty_name: Evaluation run of meta-llama/Llama-2-7b-hf
dataset_summary: "Dataset automatically created during the evaluation run of model\
  \ [meta-llama/Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf) on\
  \ the [Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard).\n\
  \nThe dataset is composed of 127 configuration, each one coresponding to one of\
  \ the evaluated task.\n\nThe dataset has been created from 16 run(s). Each run can\
  \ be found as a specific split in each configuration, the split being named using\
  \ the timestamp of the run.The \"train\" split is always pointing to the latest\
  \ results.\n\nAn additional configuration \"results\" store all the aggregated results\
  \ of the run (and is used to compute and display the aggregated metrics on the [Open\
  \ LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)).\n\
  \nTo load the details from a run, you can for instance do the following:\n```python\n\
  from datasets import load_dataset\ndata = load_dataset(\"open-llm-leaderboard/details_meta-llama__Llama-2-7b-hf\"\
  ,\n\t\"harness_gsm8k_5\",\n\tsplit=\"train\")\n```\n\n## Latest results\n\nThese\
  \ are the [latest results from run 2023-12-02T13:00:54.924067](https://huggingface.co/datasets/open-llm-leaderboard/details_meta-llama__Llama-2-7b-hf/blob/main/results_2023-12-02T13-00-54.924067.json)(note\
  \ that their might be results for other tasks in the repos if successive evals didn't\
  \ cover the same tasks. You find each in the results and the \"latest\" split for\
  \ each eval):\n\n```python\n{\n    \"all\": {\n        \"acc\": 0.14480667172100076,\n\
  \        \"acc_stderr\": 0.009693234799052708\n    },\n    \"harness|gsm8k|5\":\
  \ {\n        \"acc\": 0.14480667172100076,\n        \"acc_stderr\": 0.009693234799052708\n\
  \    }\n}\n```"
repo_url: https://huggingface.co/meta-llama/Llama-2-7b-hf
leaderboard_url: https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard
point_of_contact: clementine@hf.co
configs:
- config_name: harness_arc_challenge_25
  data_files:
  - split: 2023_08_19T16_35_46.942696
    path:
    - '**/details_harness|arc:challenge|25_2023-08-19T16:35:46.942696.parquet'
  - split: 2023_08_24T09_19_51.585793
    path:
    - '**/details_harness|arc:challenge|25_2023-08-24T09:19:51.585793.parquet'
  - split: 2023_08_29T17_54_59.197645
    path:
    - '**/details_harness|arc:challenge|25_2023-08-29T17:54:59.197645.parquet'
  - split: 2023_09_15T09_53_02.418861
    path:
    - '**/details_harness|arc:challenge|25_2023-09-15T09-53-02.418861.parquet'
  - split: latest
    path:
    - '**/details_harness|arc:challenge|25_2023-09-15T09-53-02.418861.parquet'
- config_name: harness_drop_0
  data_files:
  - split: 2023_09_14T20_50_38.766533
    path:
    - '**/details_harness|drop|0_2023-09-14T20-50-38.766533.parquet'
  - split: 2023_09_15T08_35_01.075146
    path:
    - '**/details_harness|drop|0_2023-09-15T08-35-01.075146.parquet'
  - split: latest
    path:
    - '**/details_harness|drop|0_2023-09-15T08-35-01.075146.parquet'
- config_name: harness_drop_3
  data_files:
  - split: 2023_09_08T17_00_44.389859
    path:
    - '**/details_harness|drop|3_2023-09-08T17-00-44.389859.parquet'
  - split: 2023_09_09T12_32_30.613622
    path:
    - '**/details_harness|drop|3_2023-09-09T12-32-30.613622.parquet'
  - split: 2023_09_20T14_39_46.791628
    path:
    - '**/details_harness|drop|3_2023-09-20T14-39-46.791628.parquet'
  - split: latest
    path:
    - '**/details_harness|drop|3_2023-09-20T14-39-46.791628.parquet'
- config_name: harness_gsm8k_0
  data_files:
  - split: 2023_09_15T08_35_01.075146
    path:
    - '**/details_harness|gsm8k|0_2023-09-15T08-35-01.075146.parquet'
  - split: latest
    path:
    - '**/details_harness|gsm8k|0_2023-09-15T08-35-01.075146.parquet'
- config_name: harness_gsm8k_5
  data_files:
  - split: 2023_09_08T17_00_44.389859
    path:
    - '**/details_harness|gsm8k|5_2023-09-08T17-00-44.389859.parquet'
  - split: 2023_09_09T12_32_30.613622
    path:
    - '**/details_harness|gsm8k|5_2023-09-09T12-32-30.613622.parquet'
  - split: 2023_09_20T14_39_46.791628
    path:
    - '**/details_harness|gsm8k|5_2023-09-20T14-39-46.791628.parquet'
  - split: 2023_12_02T13_00_06.695936
    path:
    - '**/details_harness|gsm8k|5_2023-12-02T13-00-06.695936.parquet'
  - split: 2023_12_02T13_00_54.924067
    path:
    - '**/details_harness|gsm8k|5_2023-12-02T13-00-54.924067.parquet'
  - split: latest
    path:
    - '**/details_harness|gsm8k|5_2023-12-02T13-00-54.924067.parquet'
- config_name: harness_hellaswag_10
  data_files:
  - split: 2023_08_19T16_35_46.942696
    path:
    - '**/details_harness|hellaswag|10_2023-08-19T16:35:46.942696.parquet'
  - split: 2023_08_24T09_19_51.585793
    path:
    - '**/details_harness|hellaswag|10_2023-08-24T09:19:51.585793.parquet'
  - split: 2023_08_29T17_54_59.197645
    path:
    - '**/details_harness|hellaswag|10_2023-08-29T17:54:59.197645.parquet'
  - split: 2023_09_15T09_53_02.418861
    path:
    - '**/details_harness|hellaswag|10_2023-09-15T09-53-02.418861.parquet'
  - split: latest
    path:
    - '**/details_harness|hellaswag|10_2023-09-15T09-53-02.418861.parquet'
- config_name: harness_hendrycksTest_0
  data_files:
  - split: 2023_08_21T20_09_03.352670
    path:
    - '**/details_harness|hendrycksTest-abstract_algebra|0_2023-08-21T20:09:03.352670.parquet'
  - split: 2023_08_21T20_15_29.093529
    path:
    - '**/details_harness|hendrycksTest-abstract_algebra|0_2023-08-21T20:15:29.093529.parquet'
  - split: 2023_08_21T20_20_08.261679
    path:
    - '**/details_harness|hendrycksTest-abstract_algebra|0_2023-08-21T20:20:08.261679.parquet'
  - split: latest
    path:
    - '**/details_harness|hendrycksTest-abstract_algebra|0_2023-08-21T20:20:08.261679.parquet'
- config_name: harness_hendrycksTest_5
  data_files:
  - split: 2023_08_19T16_35_46.942696
    path:
    - '**/details_harness|hendrycksTest-abstract_algebra|5_2023-08-19T16:35:46.942696.parquet'
    - '**/details_harness|hendrycksTest-anatomy|5_2023-08-19T16:35:46.942696.parquet'
    - '**/details_harness|hendrycksTest-astronomy|5_2023-08-19T16:35:46.942696.parquet'
    - '**/details_harness|hendrycksTest-business_ethics|5_2023-08-19T16:35:46.942696.parquet'
    - '**/details_harness|hendrycksTest-clinical_knowledge|5_2023-08-19T16:35:46.942696.parquet'
    - '**/details_harness|hendrycksTest-college_biology|5_2023-08-19T16:35:46.942696.parquet'
    - '**/details_harness|hendrycksTest-college_chemistry|5_2023-08-19T16:35:46.942696.parquet'
    - '**/details_harness|hendrycksTest-college_computer_science|5_2023-08-19T16:35:46.942696.parquet'
    - '**/details_harness|hendrycksTest-college_mathematics|5_2023-08-19T16:35:46.942696.parquet'
    - '**/details_harness|hendrycksTest-college_medicine|5_2023-08-19T16:35:46.942696.parquet'
    - '**/details_harness|hendrycksTest-college_physics|5_2023-08-19T16:35:46.942696.parquet'
    - '**/details_harness|hendrycksTest-computer_security|5_2023-08-19T16:35:46.942696.parquet'
    - '**/details_harness|hendrycksTest-conceptual_physics|5_2023-08-19T16:35:46.942696.parquet'
    - '**/details_harness|hendrycksTest-econometrics|5_2023-08-19T16:35:46.942696.parquet'
    - '**/details_harness|hendrycksTest-electrical_engineering|5_2023-08-19T16:35:46.942696.parquet'
    - '**/details_harness|hendrycksTest-elementary_mathematics|5_2023-08-19T16:35:46.942696.parquet'
    - '**/details_harness|hendrycksTest-formal_logic|5_2023-08-19T16:35:46.942696.parquet'
    - '**/details_harness|hendrycksTest-global_facts|5_2023-08-19T16:35:46.942696.parquet'
    - '**/details_harness|hendrycksTest-high_school_biology|5_2023-08-19T16:35:46.942696.parquet'
    - '**/details_harness|hendrycksTest-high_school_chemistry|5_2023-08-19T16:35:46.942696.parquet'
    - '**/details_harness|hendrycksTest-high_school_computer_science|5_2023-08-19T16:35:46.942696.parquet'
    - '**/details_harness|hendrycksTest-high_school_european_history|5_2023-08-19T16:35:46.942696.parquet'
    - '**/details_harness|hendrycksTest-high_school_geography|5_2023-08-19T16:35:46.942696.parquet'
    - '**/details_harness|hendrycksTest-high_school_government_and_politics|5_2023-08-19T16:35:46.942696.parquet'
    - '**/details_harness|hendrycksTest-high_school_macroeconomics|5_2023-08-19T16:35:46.942696.parquet'
    - '**/details_harness|hendrycksTest-high_school_mathematics|5_2023-08-19T16:35:46.942696.parquet'
    - '**/details_harness|hendrycksTest-high_school_microeconomics|5_2023-08-19T16:35:46.942696.parquet'
    - '**/details_harness|hendrycksTest-high_school_physics|5_2023-08-19T16:35:46.942696.parquet'
    - '**/details_harness|hendrycksTest-high_school_psychology|5_2023-08-19T16:35:46.942696.parquet'
    - '**/details_harness|hendrycksTest-high_school_statistics|5_2023-08-19T16:35:46.942696.parquet'
    - '**/details_harness|hendrycksTest-high_school_us_history|5_2023-08-19T16:35:46.942696.parquet'
    - '**/details_harness|hendrycksTest-high_school_world_history|5_2023-08-19T16:35:46.942696.parquet'
    - '**/details_harness|hendrycksTest-human_aging|5_2023-08-19T16:35:46.942696.parquet'
    - '**/details_harness|hendrycksTest-human_sexuality|5_2023-08-19T16:35:46.942696.parquet'
    - '**/details_harness|hendrycksTest-international_law|5_2023-08-19T16:35:46.942696.parquet'
    - '**/details_harness|hendrycksTest-jurisprudence|5_2023-08-19T16:35:46.942696.parquet'
    - '**/details_harness|hendrycksTest-logical_fallacies|5_2023-08-19T16:35:46.942696.parquet'
    - '**/details_harness|hendrycksTest-machine_learning|5_2023-08-19T16:35:46.942696.parquet'
    - '**/details_harness|hendrycksTest-management|5_2023-08-19T16:35:46.942696.parquet'
    - '**/details_harness|hendrycksTest-marketing|5_2023-08-19T16:35:46.942696.parquet'
    - '**/details_harness|hendrycksTest-medical_genetics|5_2023-08-19T16:35:46.942696.parquet'
    - '**/details_harness|hendrycksTest-miscellaneous|5_2023-08-19T16:35:46.942696.parquet'
    - '**/details_harness|hendrycksTest-moral_disputes|5_2023-08-19T16:35:46.942696.parquet'
    - '**/details_harness|hendrycksTest-moral_scenarios|5_2023-08-19T16:35:46.942696.parquet'
    - '**/details_harness|hendrycksTest-nutrition|5_2023-08-19T16:35:46.942696.parquet'
    - '**/details_harness|hendrycksTest-philosophy|5_2023-08-19T16:35:46.942696.parquet'
    - '**/details_harness|hendrycksTest-prehistory|5_2023-08-19T16:35:46.942696.parquet'
    - '**/details_harness|hendrycksTest-professional_accounting|5_2023-08-19T16:35:46.942696.parquet'
    - '**/details_harness|hendrycksTest-professional_law|5_2023-08-19T16:35:46.942696.parquet'
    - '**/details_harness|hendrycksTest-professional_medicine|5_2023-08-19T16:35:46.942696.parquet'
    - '**/details_harness|hendrycksTest-professional_psychology|5_2023-08-19T16:35:46.942696.parquet'
    - '**/details_harness|hendrycksTest-public_relations|5_2023-08-19T16:35:46.942696.parquet'
    - '**/details_harness|hendrycksTest-security_studies|5_2023-08-19T16:35:46.942696.parquet'
    - '**/details_harness|hendrycksTest-sociology|5_2023-08-19T16:35:46.942696.parquet'
    - '**/details_harness|hendrycksTest-us_foreign_policy|5_2023-08-19T16:35:46.942696.parquet'
    - '**/details_harness|hendrycksTest-virology|5_2023-08-19T16:35:46.942696.parquet'
    - '**/details_harness|hendrycksTest-world_religions|5_2023-08-19T16:35:46.942696.parquet'
  - split: 2023_08_21T17_55_50.567332
    path:
    - '**/details_harness|hendrycksTest-abstract_algebra|5_2023-08-21T17:55:50.567332.parquet'
  - split: 2023_08_24T09_19_51.585793
    path:
    - '**/details_harness|hendrycksTest-abstract_algebra|5_2023-08-24T09:19:51.585793.parquet'
    - '**/details_harness|hendrycksTest-anatomy|5_2023-08-24T09:19:51.585793.parquet'
    - '**/details_harness|hendrycksTest-astronomy|5_2023-08-24T09:19:51.585793.parquet'
    - '**/details_harness|hendrycksTest-business_ethics|5_2023-08-24T09:19:51.585793.parquet'
    - '**/details_harness|hendrycksTest-clinical_knowledge|5_2023-08-24T09:19:51.585793.parquet'
    - '**/details_harness|hendrycksTest-college_biology|5_2023-08-24T09:19:51.585793.parquet'
    - '**/details_harness|hendrycksTest-college_chemistry|5_2023-08-24T09:19:51.585793.parquet'
    - '**/details_harness|hendrycksTest-college_computer_science|5_2023-08-24T09:19:51.585793.parquet'
    - '**/details_harness|hendrycksTest-college_mathematics|5_2023-08-24T09:19:51.585793.parquet'
    - '**/details_harness|hendrycksTest-college_medicine|5_2023-08-24T09:19:51.585793.parquet'
    - '**/details_harness|hendrycksTest-college_physics|5_2023-08-24T09:19:51.585793.parquet'
    - '**/details_harness|hendrycksTest-computer_security|5_2023-08-24T09:19:51.585793.parquet'
    - '**/details_harness|hendrycksTest-conceptual_physics|5_2023-08-24T09:19:51.585793.parquet'
    - '**/details_harness|hendrycksTest-econometrics|5_2023-08-24T09:19:51.585793.parquet'
    - '**/details_harness|hendrycksTest-electrical_engineering|5_2023-08-24T09:19:51.585793.parquet'
    - '**/details_harness|hendrycksTest-elementary_mathematics|5_2023-08-24T09:19:51.585793.parquet'
    - '**/details_harness|hendrycksTest-formal_logic|5_2023-08-24T09:19:51.585793.parquet'
    - '**/details_harness|hendrycksTest-global_facts|5_2023-08-24T09:19:51.585793.parquet'
    - '**/details_harness|hendrycksTest-high_school_biology|5_2023-08-24T09:19:51.585793.parquet'
    - '**/details_harness|hendrycksTest-high_school_chemistry|5_2023-08-24T09:19:51.585793.parquet'
    - '**/details_harness|hendrycksTest-high_school_computer_science|5_2023-08-24T09:19:51.585793.parquet'
    - '**/details_harness|hendrycksTest-high_school_european_history|5_2023-08-24T09:19:51.585793.parquet'
    - '**/details_harness|hendrycksTest-high_school_geography|5_2023-08-24T09:19:51.585793.parquet'
    - '**/details_harness|hendrycksTest-high_school_government_and_politics|5_2023-08-24T09:19:51.585793.parquet'
    - '**/details_harness|hendrycksTest-high_school_macroeconomics|5_2023-08-24T09:19:51.585793.parquet'
    - '**/details_harness|hendrycksTest-high_school_mathematics|5_2023-08-24T09:19:51.585793.parquet'
    - '**/details_harness|hendrycksTest-high_school_microeconomics|5_2023-08-24T09:19:51.585793.parquet'
    - '**/details_harness|hendrycksTest-high_school_physics|5_2023-08-24T09:19:51.585793.parquet'
    - '**/details_harness|hendrycksTest-high_school_psychology|5_2023-08-24T09:19:51.585793.parquet'
    - '**/details_harness|hendrycksTest-high_school_statistics|5_2023-08-24T09:19:51.585793.parquet'
    - '**/details_harness|hendrycksTest-high_school_us_history|5_2023-08-24T09:19:51.585793.parquet'
    - '**/details_harness|hendrycksTest-high_school_world_history|5_2023-08-24T09:19:51.585793.parquet'
    - '**/details_harness|hendrycksTest-human_aging|5_2023-08-24T09:19:51.585793.parquet'
    - '**/details_harness|hendrycksTest-human_sexuality|5_2023-08-24T09:19:51.585793.parquet'
    - '**/details_harness|hendrycksTest-international_law|5_2023-08-24T09:19:51.585793.parquet'
    - '**/details_harness|hendrycksTest-jurisprudence|5_2023-08-24T09:19:51.585793.parquet'
    - '**/details_harness|hendrycksTest-logical_fallacies|5_2023-08-24T09:19:51.585793.parquet'
    - '**/details_harness|hendrycksTest-machine_learning|5_2023-08-24T09:19:51.585793.parquet'
    - '**/details_harness|hendrycksTest-management|5_2023-08-24T09:19:51.585793.parquet'
    - '**/details_harness|hendrycksTest-marketing|5_2023-08-24T09:19:51.585793.parquet'
    - '**/details_harness|hendrycksTest-medical_genetics|5_2023-08-24T09:19:51.585793.parquet'
    - '**/details_harness|hendrycksTest-miscellaneous|5_2023-08-24T09:19:51.585793.parquet'
    - '**/details_harness|hendrycksTest-moral_disputes|5_2023-08-24T09:19:51.585793.parquet'
    - '**/details_harness|hendrycksTest-moral_scenarios|5_2023-08-24T09:19:51.585793.parquet'
    - '**/details_harness|hendrycksTest-nutrition|5_2023-08-24T09:19:51.585793.parquet'
    - '**/details_harness|hendrycksTest-philosophy|5_2023-08-24T09:19:51.585793.parquet'
    - '**/details_harness|hendrycksTest-prehistory|5_2023-08-24T09:19:51.585793.parquet'
    - '**/details_harness|hendrycksTest-professional_accounting|5_2023-08-24T09:19:51.585793.parquet'
    - '**/details_harness|hendrycksTest-professional_law|5_2023-08-24T09:19:51.585793.parquet'
    - '**/details_harness|hendrycksTest-professional_medicine|5_2023-08-24T09:19:51.585793.parquet'
    - '**/details_harness|hendrycksTest-professional_psychology|5_2023-08-24T09:19:51.585793.parquet'
    - '**/details_harness|hendrycksTest-public_relations|5_2023-08-24T09:19:51.585793.parquet'
    - '**/details_harness|hendrycksTest-security_studies|5_2023-08-24T09:19:51.585793.parquet'
    - '**/details_harness|hendrycksTest-sociology|5_2023-08-24T09:19:51.585793.parquet'
    - '**/details_harness|hendrycksTest-us_foreign_policy|5_2023-08-24T09:19:51.585793.parquet'
    - '**/details_harness|hendrycksTest-virology|5_2023-08-24T09:19:51.585793.parquet'
    - '**/details_harness|hendrycksTest-world_religions|5_2023-08-24T09:19:51.585793.parquet'
  - split: 2023_08_29T17_54_59.197645
    path:
    - '**/details_harness|hendrycksTest-abstract_algebra|5_2023-08-29T17:54:59.197645.parquet'
    - '**/details_harness|hendrycksTest-anatomy|5_2023-08-29T17:54:59.197645.parquet'
    - '**/details_harness|hendrycksTest-astronomy|5_2023-08-29T17:54:59.197645.parquet'
    - '**/details_harness|hendrycksTest-business_ethics|5_2023-08-29T17:54:59.197645.parquet'
    - '**/details_harness|hendrycksTest-clinical_knowledge|5_2023-08-29T17:54:59.197645.parquet'
    - '**/details_harness|hendrycksTest-college_biology|5_2023-08-29T17:54:59.197645.parquet'
    - '**/details_harness|hendrycksTest-college_chemistry|5_2023-08-29T17:54:59.197645.parquet'
    - '**/details_harness|hendrycksTest-college_computer_science|5_2023-08-29T17:54:59.197645.parquet'
    - '**/details_harness|hendrycksTest-college_mathematics|5_2023-08-29T17:54:59.197645.parquet'
    - '**/details_harness|hendrycksTest-college_medicine|5_2023-08-29T17:54:59.197645.parquet'
    - '**/details_harness|hendrycksTest-college_physics|5_2023-08-29T17:54:59.197645.parquet'
    - '**/details_harness|hendrycksTest-computer_security|5_2023-08-29T17:54:59.197645.parquet'
    - '**/details_harness|hendrycksTest-conceptual_physics|5_2023-08-29T17:54:59.197645.parquet'
    - '**/details_harness|hendrycksTest-econometrics|5_2023-08-29T17:54:59.197645.parquet'
    - '**/details_harness|hendrycksTest-electrical_engineering|5_2023-08-29T17:54:59.197645.parquet'
    - '**/details_harness|hendrycksTest-elementary_mathematics|5_2023-08-29T17:54:59.197645.parquet'
    - '**/details_harness|hendrycksTest-formal_logic|5_2023-08-29T17:54:59.197645.parquet'
    - '**/details_harness|hendrycksTest-global_facts|5_2023-08-29T17:54:59.197645.parquet'
    - '**/details_harness|hendrycksTest-high_school_biology|5_2023-08-29T17:54:59.197645.parquet'
    - '**/details_harness|hendrycksTest-high_school_chemistry|5_2023-08-29T17:54:59.197645.parquet'
    - '**/details_harness|hendrycksTest-high_school_computer_science|5_2023-08-29T17:54:59.197645.parquet'
    - '**/details_harness|hendrycksTest-high_school_european_history|5_2023-08-29T17:54:59.197645.parquet'
    - '**/details_harness|hendrycksTest-high_school_geography|5_2023-08-29T17:54:59.197645.parquet'
    - '**/details_harness|hendrycksTest-high_school_government_and_politics|5_2023-08-29T17:54:59.197645.parquet'
    - '**/details_harness|hendrycksTest-high_school_macroeconomics|5_2023-08-29T17:54:59.197645.parquet'
    - '**/details_harness|hendrycksTest-high_school_mathematics|5_2023-08-29T17:54:59.197645.parquet'
    - '**/details_harness|hendrycksTest-high_school_microeconomics|5_2023-08-29T17:54:59.197645.parquet'
    - '**/details_harness|hendrycksTest-high_school_physics|5_2023-08-29T17:54:59.197645.parquet'
    - '**/details_harness|hendrycksTest-high_school_psychology|5_2023-08-29T17:54:59.197645.parquet'
    - '**/details_harness|hendrycksTest-high_school_statistics|5_2023-08-29T17:54:59.197645.parquet'
    - '**/details_harness|hendrycksTest-high_school_us_history|5_2023-08-29T17:54:59.197645.parquet'
    - '**/details_harness|hendrycksTest-high_school_world_history|5_2023-08-29T17:54:59.197645.parquet'
    - '**/details_harness|hendrycksTest-human_aging|5_2023-08-29T17:54:59.197645.parquet'
    - '**/details_harness|hendrycksTest-human_sexuality|5_2023-08-29T17:54:59.197645.parquet'
    - '**/details_harness|hendrycksTest-international_law|5_2023-08-29T17:54:59.197645.parquet'
    - '**/details_harness|hendrycksTest-jurisprudence|5_2023-08-29T17:54:59.197645.parquet'
    - '**/details_harness|hendrycksTest-logical_fallacies|5_2023-08-29T17:54:59.197645.parquet'
    - '**/details_harness|hendrycksTest-machine_learning|5_2023-08-29T17:54:59.197645.parquet'
    - '**/details_harness|hendrycksTest-management|5_2023-08-29T17:54:59.197645.parquet'
    - '**/details_harness|hendrycksTest-marketing|5_2023-08-29T17:54:59.197645.parquet'
    - '**/details_harness|hendrycksTest-medical_genetics|5_2023-08-29T17:54:59.197645.parquet'
    - '**/details_harness|hendrycksTest-miscellaneous|5_2023-08-29T17:54:59.197645.parquet'
    - '**/details_harness|hendrycksTest-moral_disputes|5_2023-08-29T17:54:59.197645.parquet'
    - '**/details_harness|hendrycksTest-moral_scenarios|5_2023-08-29T17:54:59.197645.parquet'
    - '**/details_harness|hendrycksTest-nutrition|5_2023-08-29T17:54:59.197645.parquet'
    - '**/details_harness|hendrycksTest-philosophy|5_2023-08-29T17:54:59.197645.parquet'
    - '**/details_harness|hendrycksTest-prehistory|5_2023-08-29T17:54:59.197645.parquet'
    - '**/details_harness|hendrycksTest-professional_accounting|5_2023-08-29T17:54:59.197645.parquet'
    - '**/details_harness|hendrycksTest-professional_law|5_2023-08-29T17:54:59.197645.parquet'
    - '**/details_harness|hendrycksTest-professional_medicine|5_2023-08-29T17:54:59.197645.parquet'
    - '**/details_harness|hendrycksTest-professional_psychology|5_2023-08-29T17:54:59.197645.parquet'
    - '**/details_harness|hendrycksTest-public_relations|5_2023-08-29T17:54:59.197645.parquet'
    - '**/details_harness|hendrycksTest-security_studies|5_2023-08-29T17:54:59.197645.parquet'
    - '**/details_harness|hendrycksTest-sociology|5_2023-08-29T17:54:59.197645.parquet'
    - '**/details_harness|hendrycksTest-us_foreign_policy|5_2023-08-29T17:54:59.197645.parquet'
    - '**/details_harness|hendrycksTest-virology|5_2023-08-29T17:54:59.197645.parquet'
    - '**/details_harness|hendrycksTest-world_religions|5_2023-08-29T17:54:59.197645.parquet'
  - split: 2023_09_15T09_53_02.418861
    path:
    - '**/details_harness|hendrycksTest-abstract_algebra|5_2023-09-15T09-53-02.418861.parquet'
    - '**/details_harness|hendrycksTest-anatomy|5_2023-09-15T09-53-02.418861.parquet'
    - '**/details_harness|hendrycksTest-astronomy|5_2023-09-15T09-53-02.418861.parquet'
    - '**/details_harness|hendrycksTest-business_ethics|5_2023-09-15T09-53-02.418861.parquet'
    - '**/details_harness|hendrycksTest-clinical_knowledge|5_2023-09-15T09-53-02.418861.parquet'
    - '**/details_harness|hendrycksTest-college_biology|5_2023-09-15T09-53-02.418861.parquet'
    - '**/details_harness|hendrycksTest-college_chemistry|5_2023-09-15T09-53-02.418861.parquet'
    - '**/details_harness|hendrycksTest-college_computer_science|5_2023-09-15T09-53-02.418861.parquet'
    - '**/details_harness|hendrycksTest-college_mathematics|5_2023-09-15T09-53-02.418861.parquet'
    - '**/details_harness|hendrycksTest-college_medicine|5_2023-09-15T09-53-02.418861.parquet'
    - '**/details_harness|hendrycksTest-college_physics|5_2023-09-15T09-53-02.418861.parquet'
    - '**/details_harness|hendrycksTest-computer_security|5_2023-09-15T09-53-02.418861.parquet'
    - '**/details_harness|hendrycksTest-conceptual_physics|5_2023-09-15T09-53-02.418861.parquet'
    - '**/details_harness|hendrycksTest-econometrics|5_2023-09-15T09-53-02.418861.parquet'
    - '**/details_harness|hendrycksTest-electrical_engineering|5_2023-09-15T09-53-02.418861.parquet'
    - '**/details_harness|hendrycksTest-elementary_mathematics|5_2023-09-15T09-53-02.418861.parquet'
    - '**/details_harness|hendrycksTest-formal_logic|5_2023-09-15T09-53-02.418861.parquet'
    - '**/details_harness|hendrycksTest-global_facts|5_2023-09-15T09-53-02.418861.parquet'
    - '**/details_harness|hendrycksTest-high_school_biology|5_2023-09-15T09-53-02.418861.parquet'
    - '**/details_harness|hendrycksTest-high_school_chemistry|5_2023-09-15T09-53-02.418861.parquet'
    - '**/details_harness|hendrycksTest-high_school_computer_science|5_2023-09-15T09-53-02.418861.parquet'
    - '**/details_harness|hendrycksTest-high_school_european_history|5_2023-09-15T09-53-02.418861.parquet'
    - '**/details_harness|hendrycksTest-high_school_geography|5_2023-09-15T09-53-02.418861.parquet'
    - '**/details_harness|hendrycksTest-high_school_government_and_politics|5_2023-09-15T09-53-02.418861.parquet'
    - '**/details_harness|hendrycksTest-high_school_macroeconomics|5_2023-09-15T09-53-02.418861.parquet'
    - '**/details_harness|hendrycksTest-high_school_mathematics|5_2023-09-15T09-53-02.418861.parquet'
    - '**/details_harness|hendrycksTest-high_school_microeconomics|5_2023-09-15T09-53-02.418861.parquet'
    - '**/details_harness|hendrycksTest-high_school_physics|5_2023-09-15T09-53-02.418861.parquet'
    - '**/details_harness|hendrycksTest-high_school_psychology|5_2023-09-15T09-53-02.418861.parquet'
    - '**/details_harness|hendrycksTest-high_school_statistics|5_2023-09-15T09-53-02.418861.parquet'
    - '**/details_harness|hendrycksTest-high_school_us_history|5_2023-09-15T09-53-02.418861.parquet'
    - '**/details_harness|hendrycksTest-high_school_world_history|5_2023-09-15T09-53-02.418861.parquet'
    - '**/details_harness|hendrycksTest-human_aging|5_2023-09-15T09-53-02.418861.parquet'
    - '**/details_harness|hendrycksTest-human_sexuality|5_2023-09-15T09-53-02.418861.parquet'
    - '**/details_harness|hendrycksTest-international_law|5_2023-09-15T09-53-02.418861.parquet'
    - '**/details_harness|hendrycksTest-jurisprudence|5_2023-09-15T09-53-02.418861.parquet'
    - '**/details_harness|hendrycksTest-logical_fallacies|5_2023-09-15T09-53-02.418861.parquet'
    - '**/details_harness|hendrycksTest-machine_learning|5_2023-09-15T09-53-02.418861.parquet'
    - '**/details_harness|hendrycksTest-management|5_2023-09-15T09-53-02.418861.parquet'
    - '**/details_harness|hendrycksTest-marketing|5_2023-09-15T09-53-02.418861.parquet'
    - '**/details_harness|hendrycksTest-medical_genetics|5_2023-09-15T09-53-02.418861.parquet'
    - '**/details_harness|hendrycksTest-miscellaneous|5_2023-09-15T09-53-02.418861.parquet'
    - '**/details_harness|hendrycksTest-moral_disputes|5_2023-09-15T09-53-02.418861.parquet'
    - '**/details_harness|hendrycksTest-moral_scenarios|5_2023-09-15T09-53-02.418861.parquet'
    - '**/details_harness|hendrycksTest-nutrition|5_2023-09-15T09-53-02.418861.parquet'
    - '**/details_harness|hendrycksTest-philosophy|5_2023-09-15T09-53-02.418861.parquet'
    - '**/details_harness|hendrycksTest-prehistory|5_2023-09-15T09-53-02.418861.parquet'
    - '**/details_harness|hendrycksTest-professional_accounting|5_2023-09-15T09-53-02.418861.parquet'
    - '**/details_harness|hendrycksTest-professional_law|5_2023-09-15T09-53-02.418861.parquet'
    - '**/details_harness|hendrycksTest-professional_medicine|5_2023-09-15T09-53-02.418861.parquet'
    - '**/details_harness|hendrycksTest-professional_psychology|5_2023-09-15T09-53-02.418861.parquet'
    - '**/details_harness|hendrycksTest-public_relations|5_2023-09-15T09-53-02.418861.parquet'
    - '**/details_harness|hendrycksTest-security_studies|5_2023-09-15T09-53-02.418861.parquet'
    - '**/details_harness|hendrycksTest-sociology|5_2023-09-15T09-53-02.418861.parquet'
    - '**/details_harness|hendrycksTest-us_foreign_policy|5_2023-09-15T09-53-02.418861.parquet'
    - '**/details_harness|hendrycksTest-virology|5_2023-09-15T09-53-02.418861.parquet'
    - '**/details_harness|hendrycksTest-world_religions|5_2023-09-15T09-53-02.418861.parquet'
  - split: latest
    path:
    - '**/details_harness|hendrycksTest-abstract_algebra|5_2023-09-15T09-53-02.418861.parquet'
    - '**/details_harness|hendrycksTest-anatomy|5_2023-09-15T09-53-02.418861.parquet'
    - '**/details_harness|hendrycksTest-astronomy|5_2023-09-15T09-53-02.418861.parquet'
    - '**/details_harness|hendrycksTest-business_ethics|5_2023-09-15T09-53-02.418861.parquet'
    - '**/details_harness|hendrycksTest-clinical_knowledge|5_2023-09-15T09-53-02.418861.parquet'
    - '**/details_harness|hendrycksTest-college_biology|5_2023-09-15T09-53-02.418861.parquet'
    - '**/details_harness|hendrycksTest-college_chemistry|5_2023-09-15T09-53-02.418861.parquet'
    - '**/details_harness|hendrycksTest-college_computer_science|5_2023-09-15T09-53-02.418861.parquet'
    - '**/details_harness|hendrycksTest-college_mathematics|5_2023-09-15T09-53-02.418861.parquet'
    - '**/details_harness|hendrycksTest-college_medicine|5_2023-09-15T09-53-02.418861.parquet'
    - '**/details_harness|hendrycksTest-college_physics|5_2023-09-15T09-53-02.418861.parquet'
    - '**/details_harness|hendrycksTest-computer_security|5_2023-09-15T09-53-02.418861.parquet'
    - '**/details_harness|hendrycksTest-conceptual_physics|5_2023-09-15T09-53-02.418861.parquet'
    - '**/details_harness|hendrycksTest-econometrics|5_2023-09-15T09-53-02.418861.parquet'
    - '**/details_harness|hendrycksTest-electrical_engineering|5_2023-09-15T09-53-02.418861.parquet'
    - '**/details_harness|hendrycksTest-elementary_mathematics|5_2023-09-15T09-53-02.418861.parquet'
    - '**/details_harness|hendrycksTest-formal_logic|5_2023-09-15T09-53-02.418861.parquet'
    - '**/details_harness|hendrycksTest-global_facts|5_2023-09-15T09-53-02.418861.parquet'
    - '**/details_harness|hendrycksTest-high_school_biology|5_2023-09-15T09-53-02.418861.parquet'
    - '**/details_harness|hendrycksTest-high_school_chemistry|5_2023-09-15T09-53-02.418861.parquet'
    - '**/details_harness|hendrycksTest-high_school_computer_science|5_2023-09-15T09-53-02.418861.parquet'
    - '**/details_harness|hendrycksTest-high_school_european_history|5_2023-09-15T09-53-02.418861.parquet'
    - '**/details_harness|hendrycksTest-high_school_geography|5_2023-09-15T09-53-02.418861.parquet'
    - '**/details_harness|hendrycksTest-high_school_government_and_politics|5_2023-09-15T09-53-02.418861.parquet'
    - '**/details_harness|hendrycksTest-high_school_macroeconomics|5_2023-09-15T09-53-02.418861.parquet'
    - '**/details_harness|hendrycksTest-high_school_mathematics|5_2023-09-15T09-53-02.418861.parquet'
    - '**/details_harness|hendrycksTest-high_school_microeconomics|5_2023-09-15T09-53-02.418861.parquet'
    - '**/details_harness|hendrycksTest-high_school_physics|5_2023-09-15T09-53-02.418861.parquet'
    - '**/details_harness|hendrycksTest-high_school_psychology|5_2023-09-15T09-53-02.418861.parquet'
    - '**/details_harness|hendrycksTest-high_school_statistics|5_2023-09-15T09-53-02.418861.parquet'
    - '**/details_harness|hendrycksTest-high_school_us_history|5_2023-09-15T09-53-02.418861.parquet'
    - '**/details_harness|hendrycksTest-high_school_world_history|5_2023-09-15T09-53-02.418861.parquet'
    - '**/details_harness|hendrycksTest-human_aging|5_2023-09-15T09-53-02.418861.parquet'
    - '**/details_harness|hendrycksTest-human_sexuality|5_2023-09-15T09-53-02.418861.parquet'
    - '**/details_harness|hendrycksTest-international_law|5_2023-09-15T09-53-02.418861.parquet'
    - '**/details_harness|hendrycksTest-jurisprudence|5_2023-09-15T09-53-02.418861.parquet'
    - '**/details_harness|hendrycksTest-logical_fallacies|5_2023-09-15T09-53-02.418861.parquet'
    - '**/details_harness|hendrycksTest-machine_learning|5_2023-09-15T09-53-02.418861.parquet'
    - '**/details_harness|hendrycksTest-management|5_2023-09-15T09-53-02.418861.parquet'
    - '**/details_harness|hendrycksTest-marketing|5_2023-09-15T09-53-02.418861.parquet'
    - '**/details_harness|hendrycksTest-medical_genetics|5_2023-09-15T09-53-02.418861.parquet'
    - '**/details_harness|hendrycksTest-miscellaneous|5_2023-09-15T09-53-02.418861.parquet'
    - '**/details_harness|hendrycksTest-moral_disputes|5_2023-09-15T09-53-02.418861.parquet'
    - '**/details_harness|hendrycksTest-moral_scenarios|5_2023-09-15T09-53-02.418861.parquet'
    - '**/details_harness|hendrycksTest-nutrition|5_2023-09-15T09-53-02.418861.parquet'
    - '**/details_harness|hendrycksTest-philosophy|5_2023-09-15T09-53-02.418861.parquet'
    - '**/details_harness|hendrycksTest-prehistory|5_2023-09-15T09-53-02.418861.parquet'
    - '**/details_harness|hendrycksTest-professional_accounting|5_2023-09-15T09-53-02.418861.parquet'
    - '**/details_harness|hendrycksTest-professional_law|5_2023-09-15T09-53-02.418861.parquet'
    - '**/details_harness|hendrycksTest-professional_medicine|5_2023-09-15T09-53-02.418861.parquet'
    - '**/details_harness|hendrycksTest-professional_psychology|5_2023-09-15T09-53-02.418861.parquet'
    - '**/details_harness|hendrycksTest-public_relations|5_2023-09-15T09-53-02.418861.parquet'
    - '**/details_harness|hendrycksTest-security_studies|5_2023-09-15T09-53-02.418861.parquet'
    - '**/details_harness|hendrycksTest-sociology|5_2023-09-15T09-53-02.418861.parquet'
    - '**/details_harness|hendrycksTest-us_foreign_policy|5_2023-09-15T09-53-02.418861.parquet'
    - '**/details_harness|hendrycksTest-virology|5_2023-09-15T09-53-02.418861.parquet'
    - '**/details_harness|hendrycksTest-world_religions|5_2023-09-15T09-53-02.418861.parquet'
- config_name: harness_hendrycksTest_abstract_algebra_0
  data_files:
  - split: 2023_08_21T20_09_03.352670
    path:
    - '**/details_harness|hendrycksTest-abstract_algebra|0_2023-08-21T20:09:03.352670.parquet'
  - split: 2023_08_21T20_15_29.093529
    path:
    - '**/details_harness|hendrycksTest-abstract_algebra|0_2023-08-21T20:15:29.093529.parquet'
  - split: 2023_08_21T20_20_08.261679
    path:
    - '**/details_harness|hendrycksTest-abstract_algebra|0_2023-08-21T20:20:08.261679.parquet'
  - split: latest
    path:
    - '**/details_harness|hendrycksTest-abstract_algebra|0_2023-08-21T20:20:08.261679.parquet'
- config_name: harness_hendrycksTest_abstract_algebra_5
  data_files:
  - split: 2023_08_19T16_35_46.942696
    path:
    - '**/details_harness|hendrycksTest-abstract_algebra|5_2023-08-19T16:35:46.942696.parquet'
  - split: 2023_08_21T17_55_50.567332
    path:
    - '**/details_harness|hendrycksTest-abstract_algebra|5_2023-08-21T17:55:50.567332.parquet'
  - split: 2023_08_24T09_19_51.585793
    path:
    - '**/details_harness|hendrycksTest-abstract_algebra|5_2023-08-24T09:19:51.585793.parquet'
  - split: 2023_08_29T17_54_59.197645
    path:
    - '**/details_harness|hendrycksTest-abstract_algebra|5_2023-08-29T17:54:59.197645.parquet'
  - split: 2023_09_15T09_53_02.418861
    path:
    - '**/details_harness|hendrycksTest-abstract_algebra|5_2023-09-15T09-53-02.418861.parquet'
  - split: latest
    path:
    - '**/details_harness|hendrycksTest-abstract_algebra|5_2023-09-15T09-53-02.418861.parquet'
- config_name: harness_hendrycksTest_anatomy_5
  data_files:
  - split: 2023_08_19T16_35_46.942696
    path:
    - '**/details_harness|hendrycksTest-anatomy|5_2023-08-19T16:35:46.942696.parquet'
  - split: 2023_08_24T09_19_51.585793
    path:
    - '**/details_harness|hendrycksTest-anatomy|5_2023-08-24T09:19:51.585793.parquet'
  - split: 2023_08_29T17_54_59.197645
    path:
    - '**/details_harness|hendrycksTest-anatomy|5_2023-08-29T17:54:59.197645.parquet'
  - split: 2023_09_15T09_53_02.418861
    path:
    - '**/details_harness|hendrycksTest-anatomy|5_2023-09-15T09-53-02.418861.parquet'
  - split: latest
    path:
    - '**/details_harness|hendrycksTest-anatomy|5_2023-09-15T09-53-02.418861.parquet'
- config_name: harness_hendrycksTest_astronomy_5
  data_files:
  - split: 2023_08_19T16_35_46.942696
    path:
    - '**/details_harness|hendrycksTest-astronomy|5_2023-08-19T16:35:46.942696.parquet'
  - split: 2023_08_24T09_19_51.585793
    path:
    - '**/details_harness|hendrycksTest-astronomy|5_2023-08-24T09:19:51.585793.parquet'
  - split: 2023_08_29T17_54_59.197645
    path:
    - '**/details_harness|hendrycksTest-astronomy|5_2023-08-29T17:54:59.197645.parquet'
  - split: 2023_09_15T09_53_02.418861
    path:
    - '**/details_harness|hendrycksTest-astronomy|5_2023-09-15T09-53-02.418861.parquet'
  - split: latest
    path:
    - '**/details_harness|hendrycksTest-astronomy|5_2023-09-15T09-53-02.418861.parquet'
- config_name: harness_hendrycksTest_business_ethics_5
  data_files:
  - split: 2023_08_19T16_35_46.942696
    path:
    - '**/details_harness|hendrycksTest-business_ethics|5_2023-08-19T16:35:46.942696.parquet'
  - split: 2023_08_24T09_19_51.585793
    path:
    - '**/details_harness|hendrycksTest-business_ethics|5_2023-08-24T09:19:51.585793.parquet'
  - split: 2023_08_29T17_54_59.197645
    path:
    - '**/details_harness|hendrycksTest-business_ethics|5_2023-08-29T17:54:59.197645.parquet'
  - split: 2023_09_15T09_53_02.418861
    path:
    - '**/details_harness|hendrycksTest-business_ethics|5_2023-09-15T09-53-02.418861.parquet'
  - split: latest
    path:
    - '**/details_harness|hendrycksTest-business_ethics|5_2023-09-15T09-53-02.418861.parquet'
- config_name: harness_hendrycksTest_clinical_knowledge_5
  data_files:
  - split: 2023_08_19T16_35_46.942696
    path:
    - '**/details_harness|hendrycksTest-clinical_knowledge|5_2023-08-19T16:35:46.942696.parquet'
  - split: 2023_08_24T09_19_51.585793
    path:
    - '**/details_harness|hendrycksTest-clinical_knowledge|5_2023-08-24T09:19:51.585793.parquet'
  - split: 2023_08_29T17_54_59.197645
    path:
    - '**/details_harness|hendrycksTest-clinical_knowledge|5_2023-08-29T17:54:59.197645.parquet'
  - split: 2023_09_15T09_53_02.418861
    path:
    - '**/details_harness|hendrycksTest-clinical_knowledge|5_2023-09-15T09-53-02.418861.parquet'
  - split: latest
    path:
    - '**/details_harness|hendrycksTest-clinical_knowledge|5_2023-09-15T09-53-02.418861.parquet'
- config_name: harness_hendrycksTest_college_biology_5
  data_files:
  - split: 2023_08_19T16_35_46.942696
    path:
    - '**/details_harness|hendrycksTest-college_biology|5_2023-08-19T16:35:46.942696.parquet'
  - split: 2023_08_24T09_19_51.585793
    path:
    - '**/details_harness|hendrycksTest-college_biology|5_2023-08-24T09:19:51.585793.parquet'
  - split: 2023_08_29T17_54_59.197645
    path:
    - '**/details_harness|hendrycksTest-college_biology|5_2023-08-29T17:54:59.197645.parquet'
  - split: 2023_09_15T09_53_02.418861
    path:
    - '**/details_harness|hendrycksTest-college_biology|5_2023-09-15T09-53-02.418861.parquet'
  - split: latest
    path:
    - '**/details_harness|hendrycksTest-college_biology|5_2023-09-15T09-53-02.418861.parquet'
- config_name: harness_hendrycksTest_college_chemistry_5
  data_files:
  - split: 2023_08_19T16_35_46.942696
    path:
    - '**/details_harness|hendrycksTest-college_chemistry|5_2023-08-19T16:35:46.942696.parquet'
  - split: 2023_08_24T09_19_51.585793
    path:
    - '**/details_harness|hendrycksTest-college_chemistry|5_2023-08-24T09:19:51.585793.parquet'
  - split: 2023_08_29T17_54_59.197645
    path:
    - '**/details_harness|hendrycksTest-college_chemistry|5_2023-08-29T17:54:59.197645.parquet'
  - split: 2023_09_15T09_53_02.418861
    path:
    - '**/details_harness|hendrycksTest-college_chemistry|5_2023-09-15T09-53-02.418861.parquet'
  - split: latest
    path:
    - '**/details_harness|hendrycksTest-college_chemistry|5_2023-09-15T09-53-02.418861.parquet'
- config_name: harness_hendrycksTest_college_computer_science_5
  data_files:
  - split: 2023_08_19T16_35_46.942696
    path:
    - '**/details_harness|hendrycksTest-college_computer_science|5_2023-08-19T16:35:46.942696.parquet'
  - split: 2023_08_24T09_19_51.585793
    path:
    - '**/details_harness|hendrycksTest-college_computer_science|5_2023-08-24T09:19:51.585793.parquet'
  - split: 2023_08_29T17_54_59.197645
    path:
    - '**/details_harness|hendrycksTest-college_computer_science|5_2023-08-29T17:54:59.197645.parquet'
  - split: 2023_09_15T09_53_02.418861
    path:
    - '**/details_harness|hendrycksTest-college_computer_science|5_2023-09-15T09-53-02.418861.parquet'
  - split: latest
    path:
    - '**/details_harness|hendrycksTest-college_computer_science|5_2023-09-15T09-53-02.418861.parquet'
- config_name: harness_hendrycksTest_college_mathematics_5
  data_files:
  - split: 2023_08_19T16_35_46.942696
    path:
    - '**/details_harness|hendrycksTest-college_mathematics|5_2023-08-19T16:35:46.942696.parquet'
  - split: 2023_08_24T09_19_51.585793
    path:
    - '**/details_harness|hendrycksTest-college_mathematics|5_2023-08-24T09:19:51.585793.parquet'
  - split: 2023_08_29T17_54_59.197645
    path:
    - '**/details_harness|hendrycksTest-college_mathematics|5_2023-08-29T17:54:59.197645.parquet'
  - split: 2023_09_15T09_53_02.418861
    path:
    - '**/details_harness|hendrycksTest-college_mathematics|5_2023-09-15T09-53-02.418861.parquet'
  - split: latest
    path:
    - '**/details_harness|hendrycksTest-college_mathematics|5_2023-09-15T09-53-02.418861.parquet'
- config_name: harness_hendrycksTest_college_medicine_5
  data_files:
  - split: 2023_08_19T16_35_46.942696
    path:
    - '**/details_harness|hendrycksTest-college_medicine|5_2023-08-19T16:35:46.942696.parquet'
  - split: 2023_08_24T09_19_51.585793
    path:
    - '**/details_harness|hendrycksTest-college_medicine|5_2023-08-24T09:19:51.585793.parquet'
  - split: 2023_08_29T17_54_59.197645
    path:
    - '**/details_harness|hendrycksTest-college_medicine|5_2023-08-29T17:54:59.197645.parquet'
  - split: 2023_09_15T09_53_02.418861
    path:
    - '**/details_harness|hendrycksTest-college_medicine|5_2023-09-15T09-53-02.418861.parquet'
  - split: latest
    path:
    - '**/details_harness|hendrycksTest-college_medicine|5_2023-09-15T09-53-02.418861.parquet'
- config_name: harness_hendrycksTest_college_physics_5
  data_files:
  - split: 2023_08_19T16_35_46.942696
    path:
    - '**/details_harness|hendrycksTest-college_physics|5_2023-08-19T16:35:46.942696.parquet'
  - split: 2023_08_24T09_19_51.585793
    path:
    - '**/details_harness|hendrycksTest-college_physics|5_2023-08-24T09:19:51.585793.parquet'
  - split: 2023_08_29T17_54_59.197645
    path:
    - '**/details_harness|hendrycksTest-college_physics|5_2023-08-29T17:54:59.197645.parquet'
  - split: 2023_09_15T09_53_02.418861
    path:
    - '**/details_harness|hendrycksTest-college_physics|5_2023-09-15T09-53-02.418861.parquet'
  - split: latest
    path:
    - '**/details_harness|hendrycksTest-college_physics|5_2023-09-15T09-53-02.418861.parquet'
- config_name: harness_hendrycksTest_computer_security_5
  data_files:
  - split: 2023_08_19T16_35_46.942696
    path:
    - '**/details_harness|hendrycksTest-computer_security|5_2023-08-19T16:35:46.942696.parquet'
  - split: 2023_08_24T09_19_51.585793
    path:
    - '**/details_harness|hendrycksTest-computer_security|5_2023-08-24T09:19:51.585793.parquet'
  - split: 2023_08_29T17_54_59.197645
    path:
    - '**/details_harness|hendrycksTest-computer_security|5_2023-08-29T17:54:59.197645.parquet'
  - split: 2023_09_15T09_53_02.418861
    path:
    - '**/details_harness|hendrycksTest-computer_security|5_2023-09-15T09-53-02.418861.parquet'
  - split: latest
    path:
    - '**/details_harness|hendrycksTest-computer_security|5_2023-09-15T09-53-02.418861.parquet'
- config_name: harness_hendrycksTest_conceptual_physics_5
  data_files:
  - split: 2023_08_19T16_35_46.942696
    path:
    - '**/details_harness|hendrycksTest-conceptual_physics|5_2023-08-19T16:35:46.942696.parquet'
  - split: 2023_08_24T09_19_51.585793
    path:
    - '**/details_harness|hendrycksTest-conceptual_physics|5_2023-08-24T09:19:51.585793.parquet'
  - split: 2023_08_29T17_54_59.197645
    path:
    - '**/details_harness|hendrycksTest-conceptual_physics|5_2023-08-29T17:54:59.197645.parquet'
  - split: 2023_09_15T09_53_02.418861
    path:
    - '**/details_harness|hendrycksTest-conceptual_physics|5_2023-09-15T09-53-02.418861.parquet'
  - split: latest
    path:
    - '**/details_harness|hendrycksTest-conceptual_physics|5_2023-09-15T09-53-02.418861.parquet'
- config_name: harness_hendrycksTest_econometrics_5
  data_files:
  - split: 2023_08_19T16_35_46.942696
    path:
    - '**/details_harness|hendrycksTest-econometrics|5_2023-08-19T16:35:46.942696.parquet'
  - split: 2023_08_24T09_19_51.585793
    path:
    - '**/details_harness|hendrycksTest-econometrics|5_2023-08-24T09:19:51.585793.parquet'
  - split: 2023_08_29T17_54_59.197645
    path:
    - '**/details_harness|hendrycksTest-econometrics|5_2023-08-29T17:54:59.197645.parquet'
  - split: 2023_09_15T09_53_02.418861
    path:
    - '**/details_harness|hendrycksTest-econometrics|5_2023-09-15T09-53-02.418861.parquet'
  - split: latest
    path:
    - '**/details_harness|hendrycksTest-econometrics|5_2023-09-15T09-53-02.418861.parquet'
- config_name: harness_hendrycksTest_electrical_engineering_5
  data_files:
  - split: 2023_08_19T16_35_46.942696
    path:
    - '**/details_harness|hendrycksTest-electrical_engineering|5_2023-08-19T16:35:46.942696.parquet'
  - split: 2023_08_24T09_19_51.585793
    path:
    - '**/details_harness|hendrycksTest-electrical_engineering|5_2023-08-24T09:19:51.585793.parquet'
  - split: 2023_08_29T17_54_59.197645
    path:
    - '**/details_harness|hendrycksTest-electrical_engineering|5_2023-08-29T17:54:59.197645.parquet'
  - split: 2023_09_15T09_53_02.418861
    path:
    - '**/details_harness|hendrycksTest-electrical_engineering|5_2023-09-15T09-53-02.418861.parquet'
  - split: latest
    path:
    - '**/details_harness|hendrycksTest-electrical_engineering|5_2023-09-15T09-53-02.418861.parquet'
- config_name: harness_hendrycksTest_elementary_mathematics_5
  data_files:
  - split: 2023_08_19T16_35_46.942696
    path:
    - '**/details_harness|hendrycksTest-elementary_mathematics|5_2023-08-19T16:35:46.942696.parquet'
  - split: 2023_08_24T09_19_51.585793
    path:
    - '**/details_harness|hendrycksTest-elementary_mathematics|5_2023-08-24T09:19:51.585793.parquet'
  - split: 2023_08_29T17_54_59.197645
    path:
    - '**/details_harness|hendrycksTest-elementary_mathematics|5_2023-08-29T17:54:59.197645.parquet'
  - split: 2023_09_15T09_53_02.418861
    path:
    - '**/details_harness|hendrycksTest-elementary_mathematics|5_2023-09-15T09-53-02.418861.parquet'
  - split: latest
    path:
    - '**/details_harness|hendrycksTest-elementary_mathematics|5_2023-09-15T09-53-02.418861.parquet'
- config_name: harness_hendrycksTest_formal_logic_5
  data_files:
  - split: 2023_08_19T16_35_46.942696
    path:
    - '**/details_harness|hendrycksTest-formal_logic|5_2023-08-19T16:35:46.942696.parquet'
  - split: 2023_08_24T09_19_51.585793
    path:
    - '**/details_harness|hendrycksTest-formal_logic|5_2023-08-24T09:19:51.585793.parquet'
  - split: 2023_08_29T17_54_59.197645
    path:
    - '**/details_harness|hendrycksTest-formal_logic|5_2023-08-29T17:54:59.197645.parquet'
  - split: 2023_09_15T09_53_02.418861
    path:
    - '**/details_harness|hendrycksTest-formal_logic|5_2023-09-15T09-53-02.418861.parquet'
  - split: latest
    path:
    - '**/details_harness|hendrycksTest-formal_logic|5_2023-09-15T09-53-02.418861.parquet'
- config_name: harness_hendrycksTest_global_facts_5
  data_files:
  - split: 2023_08_19T16_35_46.942696
    path:
    - '**/details_harness|hendrycksTest-global_facts|5_2023-08-19T16:35:46.942696.parquet'
  - split: 2023_08_24T09_19_51.585793
    path:
    - '**/details_harness|hendrycksTest-global_facts|5_2023-08-24T09:19:51.585793.parquet'
  - split: 2023_08_29T17_54_59.197645
    path:
    - '**/details_harness|hendrycksTest-global_facts|5_2023-08-29T17:54:59.197645.parquet'
  - split: 2023_09_15T09_53_02.418861
    path:
    - '**/details_harness|hendrycksTest-global_facts|5_2023-09-15T09-53-02.418861.parquet'
  - split: latest
    path:
    - '**/details_harness|hendrycksTest-global_facts|5_2023-09-15T09-53-02.418861.parquet'
- config_name: harness_hendrycksTest_high_school_biology_5
  data_files:
  - split: 2023_08_19T16_35_46.942696
    path:
    - '**/details_harness|hendrycksTest-high_school_biology|5_2023-08-19T16:35:46.942696.parquet'
  - split: 2023_08_24T09_19_51.585793
    path:
    - '**/details_harness|hendrycksTest-high_school_biology|5_2023-08-24T09:19:51.585793.parquet'
  - split: 2023_08_29T17_54_59.197645
    path:
    - '**/details_harness|hendrycksTest-high_school_biology|5_2023-08-29T17:54:59.197645.parquet'
  - split: 2023_09_15T09_53_02.418861
    path:
    - '**/details_harness|hendrycksTest-high_school_biology|5_2023-09-15T09-53-02.418861.parquet'
  - split: latest
    path:
    - '**/details_harness|hendrycksTest-high_school_biology|5_2023-09-15T09-53-02.418861.parquet'
- config_name: harness_hendrycksTest_high_school_chemistry_5
  data_files:
  - split: 2023_08_19T16_35_46.942696
    path:
    - '**/details_harness|hendrycksTest-high_school_chemistry|5_2023-08-19T16:35:46.942696.parquet'
  - split: 2023_08_24T09_19_51.585793
    path:
    - '**/details_harness|hendrycksTest-high_school_chemistry|5_2023-08-24T09:19:51.585793.parquet'
  - split: 2023_08_29T17_54_59.197645
    path:
    - '**/details_harness|hendrycksTest-high_school_chemistry|5_2023-08-29T17:54:59.197645.parquet'
  - split: 2023_09_15T09_53_02.418861
    path:
    - '**/details_harness|hendrycksTest-high_school_chemistry|5_2023-09-15T09-53-02.418861.parquet'
  - split: latest
    path:
    - '**/details_harness|hendrycksTest-high_school_chemistry|5_2023-09-15T09-53-02.418861.parquet'
- config_name: harness_hendrycksTest_high_school_computer_science_5
  data_files:
  - split: 2023_08_19T16_35_46.942696
    path:
    - '**/details_harness|hendrycksTest-high_school_computer_science|5_2023-08-19T16:35:46.942696.parquet'
  - split: 2023_08_24T09_19_51.585793
    path:
    - '**/details_harness|hendrycksTest-high_school_computer_science|5_2023-08-24T09:19:51.585793.parquet'
  - split: 2023_08_29T17_54_59.197645
    path:
    - '**/details_harness|hendrycksTest-high_school_computer_science|5_2023-08-29T17:54:59.197645.parquet'
  - split: 2023_09_15T09_53_02.418861
    path:
    - '**/details_harness|hendrycksTest-high_school_computer_science|5_2023-09-15T09-53-02.418861.parquet'
  - split: latest
    path:
    - '**/details_harness|hendrycksTest-high_school_computer_science|5_2023-09-15T09-53-02.418861.parquet'
- config_name: harness_hendrycksTest_high_school_european_history_5
  data_files:
  - split: 2023_08_19T16_35_46.942696
    path:
    - '**/details_harness|hendrycksTest-high_school_european_history|5_2023-08-19T16:35:46.942696.parquet'
  - split: 2023_08_24T09_19_51.585793
    path:
    - '**/details_harness|hendrycksTest-high_school_european_history|5_2023-08-24T09:19:51.585793.parquet'
  - split: 2023_08_29T17_54_59.197645
    path:
    - '**/details_harness|hendrycksTest-high_school_european_history|5_2023-08-29T17:54:59.197645.parquet'
  - split: 2023_09_15T09_53_02.418861
    path:
    - '**/details_harness|hendrycksTest-high_school_european_history|5_2023-09-15T09-53-02.418861.parquet'
  - split: latest
    path:
    - '**/details_harness|hendrycksTest-high_school_european_history|5_2023-09-15T09-53-02.418861.parquet'
- config_name: harness_hendrycksTest_high_school_geography_5
  data_files:
  - split: 2023_08_19T16_35_46.942696
    path:
    - '**/details_harness|hendrycksTest-high_school_geography|5_2023-08-19T16:35:46.942696.parquet'
  - split: 2023_08_24T09_19_51.585793
    path:
    - '**/details_harness|hendrycksTest-high_school_geography|5_2023-08-24T09:19:51.585793.parquet'
  - split: 2023_08_29T17_54_59.197645
    path:
    - '**/details_harness|hendrycksTest-high_school_geography|5_2023-08-29T17:54:59.197645.parquet'
  - split: 2023_09_15T09_53_02.418861
    path:
    - '**/details_harness|hendrycksTest-high_school_geography|5_2023-09-15T09-53-02.418861.parquet'
  - split: latest
    path:
    - '**/details_harness|hendrycksTest-high_school_geography|5_2023-09-15T09-53-02.418861.parquet'
- config_name: harness_hendrycksTest_high_school_government_and_politics_5
  data_files:
  - split: 2023_08_19T16_35_46.942696
    path:
    - '**/details_harness|hendrycksTest-high_school_government_and_politics|5_2023-08-19T16:35:46.942696.parquet'
  - split: 2023_08_24T09_19_51.585793
    path:
    - '**/details_harness|hendrycksTest-high_school_government_and_politics|5_2023-08-24T09:19:51.585793.parquet'
  - split: 2023_08_29T17_54_59.197645
    path:
    - '**/details_harness|hendrycksTest-high_school_government_and_politics|5_2023-08-29T17:54:59.197645.parquet'
  - split: 2023_09_15T09_53_02.418861
    path:
    - '**/details_harness|hendrycksTest-high_school_government_and_politics|5_2023-09-15T09-53-02.418861.parquet'
  - split: latest
    path:
    - '**/details_harness|hendrycksTest-high_school_government_and_politics|5_2023-09-15T09-53-02.418861.parquet'
- config_name: harness_hendrycksTest_high_school_macroeconomics_5
  data_files:
  - split: 2023_08_19T16_35_46.942696
    path:
    - '**/details_harness|hendrycksTest-high_school_macroeconomics|5_2023-08-19T16:35:46.942696.parquet'
  - split: 2023_08_24T09_19_51.585793
    path:
    - '**/details_harness|hendrycksTest-high_school_macroeconomics|5_2023-08-24T09:19:51.585793.parquet'
  - split: 2023_08_29T17_54_59.197645
    path:
    - '**/details_harness|hendrycksTest-high_school_macroeconomics|5_2023-08-29T17:54:59.197645.parquet'
  - split: 2023_09_15T09_53_02.418861
    path:
    - '**/details_harness|hendrycksTest-high_school_macroeconomics|5_2023-09-15T09-53-02.418861.parquet'
  - split: latest
    path:
    - '**/details_harness|hendrycksTest-high_school_macroeconomics|5_2023-09-15T09-53-02.418861.parquet'
- config_name: harness_hendrycksTest_high_school_mathematics_5
  data_files:
  - split: 2023_08_19T16_35_46.942696
    path:
    - '**/details_harness|hendrycksTest-high_school_mathematics|5_2023-08-19T16:35:46.942696.parquet'
  - split: 2023_08_24T09_19_51.585793
    path:
    - '**/details_harness|hendrycksTest-high_school_mathematics|5_2023-08-24T09:19:51.585793.parquet'
  - split: 2023_08_29T17_54_59.197645
    path:
    - '**/details_harness|hendrycksTest-high_school_mathematics|5_2023-08-29T17:54:59.197645.parquet'
  - split: 2023_09_15T09_53_02.418861
    path:
    - '**/details_harness|hendrycksTest-high_school_mathematics|5_2023-09-15T09-53-02.418861.parquet'
  - split: latest
    path:
    - '**/details_harness|hendrycksTest-high_school_mathematics|5_2023-09-15T09-53-02.418861.parquet'
- config_name: harness_hendrycksTest_high_school_microeconomics_5
  data_files:
  - split: 2023_08_19T16_35_46.942696
    path:
    - '**/details_harness|hendrycksTest-high_school_microeconomics|5_2023-08-19T16:35:46.942696.parquet'
  - split: 2023_08_24T09_19_51.585793
    path:
    - '**/details_harness|hendrycksTest-high_school_microeconomics|5_2023-08-24T09:19:51.585793.parquet'
  - split: 2023_08_29T17_54_59.197645
    path:
    - '**/details_harness|hendrycksTest-high_school_microeconomics|5_2023-08-29T17:54:59.197645.parquet'
  - split: 2023_09_15T09_53_02.418861
    path:
    - '**/details_harness|hendrycksTest-high_school_microeconomics|5_2023-09-15T09-53-02.418861.parquet'
  - split: latest
    path:
    - '**/details_harness|hendrycksTest-high_school_microeconomics|5_2023-09-15T09-53-02.418861.parquet'
- config_name: harness_hendrycksTest_high_school_physics_5
  data_files:
  - split: 2023_08_19T16_35_46.942696
    path:
    - '**/details_harness|hendrycksTest-high_school_physics|5_2023-08-19T16:35:46.942696.parquet'
  - split: 2023_08_24T09_19_51.585793
    path:
    - '**/details_harness|hendrycksTest-high_school_physics|5_2023-08-24T09:19:51.585793.parquet'
  - split: 2023_08_29T17_54_59.197645
    path:
    - '**/details_harness|hendrycksTest-high_school_physics|5_2023-08-29T17:54:59.197645.parquet'
  - split: 2023_09_15T09_53_02.418861
    path:
    - '**/details_harness|hendrycksTest-high_school_physics|5_2023-09-15T09-53-02.418861.parquet'
  - split: latest
    path:
    - '**/details_harness|hendrycksTest-high_school_physics|5_2023-09-15T09-53-02.418861.parquet'
- config_name: harness_hendrycksTest_high_school_psychology_5
  data_files:
  - split: 2023_08_19T16_35_46.942696
    path:
    - '**/details_harness|hendrycksTest-high_school_psychology|5_2023-08-19T16:35:46.942696.parquet'
  - split: 2023_08_24T09_19_51.585793
    path:
    - '**/details_harness|hendrycksTest-high_school_psychology|5_2023-08-24T09:19:51.585793.parquet'
  - split: 2023_08_29T17_54_59.197645
    path:
    - '**/details_harness|hendrycksTest-high_school_psychology|5_2023-08-29T17:54:59.197645.parquet'
  - split: 2023_09_15T09_53_02.418861
    path:
    - '**/details_harness|hendrycksTest-high_school_psychology|5_2023-09-15T09-53-02.418861.parquet'
  - split: latest
    path:
    - '**/details_harness|hendrycksTest-high_school_psychology|5_2023-09-15T09-53-02.418861.parquet'
- config_name: harness_hendrycksTest_high_school_statistics_5
  data_files:
  - split: 2023_08_19T16_35_46.942696
    path:
    - '**/details_harness|hendrycksTest-high_school_statistics|5_2023-08-19T16:35:46.942696.parquet'
  - split: 2023_08_24T09_19_51.585793
    path:
    - '**/details_harness|hendrycksTest-high_school_statistics|5_2023-08-24T09:19:51.585793.parquet'
  - split: 2023_08_29T17_54_59.197645
    path:
    - '**/details_harness|hendrycksTest-high_school_statistics|5_2023-08-29T17:54:59.197645.parquet'
  - split: 2023_09_15T09_53_02.418861
    path:
    - '**/details_harness|hendrycksTest-high_school_statistics|5_2023-09-15T09-53-02.418861.parquet'
  - split: latest
    path:
    - '**/details_harness|hendrycksTest-high_school_statistics|5_2023-09-15T09-53-02.418861.parquet'
- config_name: harness_hendrycksTest_high_school_us_history_5
  data_files:
  - split: 2023_08_19T16_35_46.942696
    path:
    - '**/details_harness|hendrycksTest-high_school_us_history|5_2023-08-19T16:35:46.942696.parquet'
  - split: 2023_08_24T09_19_51.585793
    path:
    - '**/details_harness|hendrycksTest-high_school_us_history|5_2023-08-24T09:19:51.585793.parquet'
  - split: 2023_08_29T17_54_59.197645
    path:
    - '**/details_harness|hendrycksTest-high_school_us_history|5_2023-08-29T17:54:59.197645.parquet'
  - split: 2023_09_15T09_53_02.418861
    path:
    - '**/details_harness|hendrycksTest-high_school_us_history|5_2023-09-15T09-53-02.418861.parquet'
  - split: latest
    path:
    - '**/details_harness|hendrycksTest-high_school_us_history|5_2023-09-15T09-53-02.418861.parquet'
- config_name: harness_hendrycksTest_high_school_world_history_5
  data_files:
  - split: 2023_08_19T16_35_46.942696
    path:
    - '**/details_harness|hendrycksTest-high_school_world_history|5_2023-08-19T16:35:46.942696.parquet'
  - split: 2023_08_24T09_19_51.585793
    path:
    - '**/details_harness|hendrycksTest-high_school_world_history|5_2023-08-24T09:19:51.585793.parquet'
  - split: 2023_08_29T17_54_59.197645
    path:
    - '**/details_harness|hendrycksTest-high_school_world_history|5_2023-08-29T17:54:59.197645.parquet'
  - split: 2023_09_15T09_53_02.418861
    path:
    - '**/details_harness|hendrycksTest-high_school_world_history|5_2023-09-15T09-53-02.418861.parquet'
  - split: latest
    path:
    - '**/details_harness|hendrycksTest-high_school_world_history|5_2023-09-15T09-53-02.418861.parquet'
- config_name: harness_hendrycksTest_human_aging_5
  data_files:
  - split: 2023_08_19T16_35_46.942696
    path:
    - '**/details_harness|hendrycksTest-human_aging|5_2023-08-19T16:35:46.942696.parquet'
  - split: 2023_08_24T09_19_51.585793
    path:
    - '**/details_harness|hendrycksTest-human_aging|5_2023-08-24T09:19:51.585793.parquet'
  - split: 2023_08_29T17_54_59.197645
    path:
    - '**/details_harness|hendrycksTest-human_aging|5_2023-08-29T17:54:59.197645.parquet'
  - split: 2023_09_15T09_53_02.418861
    path:
    - '**/details_harness|hendrycksTest-human_aging|5_2023-09-15T09-53-02.418861.parquet'
  - split: latest
    path:
    - '**/details_harness|hendrycksTest-human_aging|5_2023-09-15T09-53-02.418861.parquet'
- config_name: harness_hendrycksTest_human_sexuality_5
  data_files:
  - split: 2023_08_19T16_35_46.942696
    path:
    - '**/details_harness|hendrycksTest-human_sexuality|5_2023-08-19T16:35:46.942696.parquet'
  - split: 2023_08_24T09_19_51.585793
    path:
    - '**/details_harness|hendrycksTest-human_sexuality|5_2023-08-24T09:19:51.585793.parquet'
  - split: 2023_08_29T17_54_59.197645
    path:
    - '**/details_harness|hendrycksTest-human_sexuality|5_2023-08-29T17:54:59.197645.parquet'
  - split: 2023_09_15T09_53_02.418861
    path:
    - '**/details_harness|hendrycksTest-human_sexuality|5_2023-09-15T09-53-02.418861.parquet'
  - split: latest
    path:
    - '**/details_harness|hendrycksTest-human_sexuality|5_2023-09-15T09-53-02.418861.parquet'
- config_name: harness_hendrycksTest_international_law_5
  data_files:
  - split: 2023_08_19T16_35_46.942696
    path:
    - '**/details_harness|hendrycksTest-international_law|5_2023-08-19T16:35:46.942696.parquet'
  - split: 2023_08_24T09_19_51.585793
    path:
    - '**/details_harness|hendrycksTest-international_law|5_2023-08-24T09:19:51.585793.parquet'
  - split: 2023_08_29T17_54_59.197645
    path:
    - '**/details_harness|hendrycksTest-international_law|5_2023-08-29T17:54:59.197645.parquet'
  - split: 2023_09_15T09_53_02.418861
    path:
    - '**/details_harness|hendrycksTest-international_law|5_2023-09-15T09-53-02.418861.parquet'
  - split: latest
    path:
    - '**/details_harness|hendrycksTest-international_law|5_2023-09-15T09-53-02.418861.parquet'
- config_name: harness_hendrycksTest_jurisprudence_5
  data_files:
  - split: 2023_08_19T16_35_46.942696
    path:
    - '**/details_harness|hendrycksTest-jurisprudence|5_2023-08-19T16:35:46.942696.parquet'
  - split: 2023_08_24T09_19_51.585793
    path:
    - '**/details_harness|hendrycksTest-jurisprudence|5_2023-08-24T09:19:51.585793.parquet'
  - split: 2023_08_29T17_54_59.197645
    path:
    - '**/details_harness|hendrycksTest-jurisprudence|5_2023-08-29T17:54:59.197645.parquet'
  - split: 2023_09_15T09_53_02.418861
    path:
    - '**/details_harness|hendrycksTest-jurisprudence|5_2023-09-15T09-53-02.418861.parquet'
  - split: latest
    path:
    - '**/details_harness|hendrycksTest-jurisprudence|5_2023-09-15T09-53-02.418861.parquet'
- config_name: harness_hendrycksTest_logical_fallacies_5
  data_files:
  - split: 2023_08_19T16_35_46.942696
    path:
    - '**/details_harness|hendrycksTest-logical_fallacies|5_2023-08-19T16:35:46.942696.parquet'
  - split: 2023_08_24T09_19_51.585793
    path:
    - '**/details_harness|hendrycksTest-logical_fallacies|5_2023-08-24T09:19:51.585793.parquet'
  - split: 2023_08_29T17_54_59.197645
    path:
    - '**/details_harness|hendrycksTest-logical_fallacies|5_2023-08-29T17:54:59.197645.parquet'
  - split: 2023_09_15T09_53_02.418861
    path:
    - '**/details_harness|hendrycksTest-logical_fallacies|5_2023-09-15T09-53-02.418861.parquet'
  - split: latest
    path:
    - '**/details_harness|hendrycksTest-logical_fallacies|5_2023-09-15T09-53-02.418861.parquet'
- config_name: harness_hendrycksTest_machine_learning_5
  data_files:
  - split: 2023_08_19T16_35_46.942696
    path:
    - '**/details_harness|hendrycksTest-machine_learning|5_2023-08-19T16:35:46.942696.parquet'
  - split: 2023_08_24T09_19_51.585793
    path:
    - '**/details_harness|hendrycksTest-machine_learning|5_2023-08-24T09:19:51.585793.parquet'
  - split: 2023_08_29T17_54_59.197645
    path:
    - '**/details_harness|hendrycksTest-machine_learning|5_2023-08-29T17:54:59.197645.parquet'
  - split: 2023_09_15T09_53_02.418861
    path:
    - '**/details_harness|hendrycksTest-machine_learning|5_2023-09-15T09-53-02.418861.parquet'
  - split: latest
    path:
    - '**/details_harness|hendrycksTest-machine_learning|5_2023-09-15T09-53-02.418861.parquet'
- config_name: harness_hendrycksTest_management_5
  data_files:
  - split: 2023_08_19T16_35_46.942696
    path:
    - '**/details_harness|hendrycksTest-management|5_2023-08-19T16:35:46.942696.parquet'
  - split: 2023_08_24T09_19_51.585793
    path:
    - '**/details_harness|hendrycksTest-management|5_2023-08-24T09:19:51.585793.parquet'
  - split: 2023_08_29T17_54_59.197645
    path:
    - '**/details_harness|hendrycksTest-management|5_2023-08-29T17:54:59.197645.parquet'
  - split: 2023_09_15T09_53_02.418861
    path:
    - '**/details_harness|hendrycksTest-management|5_2023-09-15T09-53-02.418861.parquet'
  - split: latest
    path:
    - '**/details_harness|hendrycksTest-management|5_2023-09-15T09-53-02.418861.parquet'
- config_name: harness_hendrycksTest_marketing_5
  data_files:
  - split: 2023_08_19T16_35_46.942696
    path:
    - '**/details_harness|hendrycksTest-marketing|5_2023-08-19T16:35:46.942696.parquet'
  - split: 2023_08_24T09_19_51.585793
    path:
    - '**/details_harness|hendrycksTest-marketing|5_2023-08-24T09:19:51.585793.parquet'
  - split: 2023_08_29T17_54_59.197645
    path:
    - '**/details_harness|hendrycksTest-marketing|5_2023-08-29T17:54:59.197645.parquet'
  - split: 2023_09_15T09_53_02.418861
    path:
    - '**/details_harness|hendrycksTest-marketing|5_2023-09-15T09-53-02.418861.parquet'
  - split: latest
    path:
    - '**/details_harness|hendrycksTest-marketing|5_2023-09-15T09-53-02.418861.parquet'
- config_name: harness_hendrycksTest_medical_genetics_5
  data_files:
  - split: 2023_08_19T16_35_46.942696
    path:
    - '**/details_harness|hendrycksTest-medical_genetics|5_2023-08-19T16:35:46.942696.parquet'
  - split: 2023_08_24T09_19_51.585793
    path:
    - '**/details_harness|hendrycksTest-medical_genetics|5_2023-08-24T09:19:51.585793.parquet'
  - split: 2023_08_29T17_54_59.197645
    path:
    - '**/details_harness|hendrycksTest-medical_genetics|5_2023-08-29T17:54:59.197645.parquet'
  - split: 2023_09_15T09_53_02.418861
    path:
    - '**/details_harness|hendrycksTest-medical_genetics|5_2023-09-15T09-53-02.418861.parquet'
  - split: latest
    path:
    - '**/details_harness|hendrycksTest-medical_genetics|5_2023-09-15T09-53-02.418861.parquet'
- config_name: harness_hendrycksTest_miscellaneous_5
  data_files:
  - split: 2023_08_19T16_35_46.942696
    path:
    - '**/details_harness|hendrycksTest-miscellaneous|5_2023-08-19T16:35:46.942696.parquet'
  - split: 2023_08_24T09_19_51.585793
    path:
    - '**/details_harness|hendrycksTest-miscellaneous|5_2023-08-24T09:19:51.585793.parquet'
  - split: 2023_08_29T17_54_59.197645
    path:
    - '**/details_harness|hendrycksTest-miscellaneous|5_2023-08-29T17:54:59.197645.parquet'
  - split: 2023_09_15T09_53_02.418861
    path:
    - '**/details_harness|hendrycksTest-miscellaneous|5_2023-09-15T09-53-02.418861.parquet'
  - split: latest
    path:
    - '**/details_harness|hendrycksTest-miscellaneous|5_2023-09-15T09-53-02.418861.parquet'
- config_name: harness_hendrycksTest_moral_disputes_5
  data_files:
  - split: 2023_08_19T16_35_46.942696
    path:
    - '**/details_harness|hendrycksTest-moral_disputes|5_2023-08-19T16:35:46.942696.parquet'
  - split: 2023_08_24T09_19_51.585793
    path:
    - '**/details_harness|hendrycksTest-moral_disputes|5_2023-08-24T09:19:51.585793.parquet'
  - split: 2023_08_29T17_54_59.197645
    path:
    - '**/details_harness|hendrycksTest-moral_disputes|5_2023-08-29T17:54:59.197645.parquet'
  - split: 2023_09_15T09_53_02.418861
    path:
    - '**/details_harness|hendrycksTest-moral_disputes|5_2023-09-15T09-53-02.418861.parquet'
  - split: latest
    path:
    - '**/details_harness|hendrycksTest-moral_disputes|5_2023-09-15T09-53-02.418861.parquet'
- config_name: harness_hendrycksTest_moral_scenarios_5
  data_files:
  - split: 2023_08_19T16_35_46.942696
    path:
    - '**/details_harness|hendrycksTest-moral_scenarios|5_2023-08-19T16:35:46.942696.parquet'
  - split: 2023_08_24T09_19_51.585793
    path:
    - '**/details_harness|hendrycksTest-moral_scenarios|5_2023-08-24T09:19:51.585793.parquet'
  - split: 2023_08_29T17_54_59.197645
    path:
    - '**/details_harness|hendrycksTest-moral_scenarios|5_2023-08-29T17:54:59.197645.parquet'
  - split: 2023_09_15T09_53_02.418861
    path:
    - '**/details_harness|hendrycksTest-moral_scenarios|5_2023-09-15T09-53-02.418861.parquet'
  - split: latest
    path:
    - '**/details_harness|hendrycksTest-moral_scenarios|5_2023-09-15T09-53-02.418861.parquet'
- config_name: harness_hendrycksTest_nutrition_5
  data_files:
  - split: 2023_08_19T16_35_46.942696
    path:
    - '**/details_harness|hendrycksTest-nutrition|5_2023-08-19T16:35:46.942696.parquet'
  - split: 2023_08_24T09_19_51.585793
    path:
    - '**/details_harness|hendrycksTest-nutrition|5_2023-08-24T09:19:51.585793.parquet'
  - split: 2023_08_29T17_54_59.197645
    path:
    - '**/details_harness|hendrycksTest-nutrition|5_2023-08-29T17:54:59.197645.parquet'
  - split: 2023_09_15T09_53_02.418861
    path:
    - '**/details_harness|hendrycksTest-nutrition|5_2023-09-15T09-53-02.418861.parquet'
  - split: latest
    path:
    - '**/details_harness|hendrycksTest-nutrition|5_2023-09-15T09-53-02.418861.parquet'
- config_name: harness_hendrycksTest_philosophy_5
  data_files:
  - split: 2023_08_19T16_35_46.942696
    path:
    - '**/details_harness|hendrycksTest-philosophy|5_2023-08-19T16:35:46.942696.parquet'
  - split: 2023_08_24T09_19_51.585793
    path:
    - '**/details_harness|hendrycksTest-philosophy|5_2023-08-24T09:19:51.585793.parquet'
  - split: 2023_08_29T17_54_59.197645
    path:
    - '**/details_harness|hendrycksTest-philosophy|5_2023-08-29T17:54:59.197645.parquet'
  - split: 2023_09_15T09_53_02.418861
    path:
    - '**/details_harness|hendrycksTest-philosophy|5_2023-09-15T09-53-02.418861.parquet'
  - split: latest
    path:
    - '**/details_harness|hendrycksTest-philosophy|5_2023-09-15T09-53-02.418861.parquet'
- config_name: harness_hendrycksTest_prehistory_5
  data_files:
  - split: 2023_08_19T16_35_46.942696
    path:
    - '**/details_harness|hendrycksTest-prehistory|5_2023-08-19T16:35:46.942696.parquet'
  - split: 2023_08_24T09_19_51.585793
    path:
    - '**/details_harness|hendrycksTest-prehistory|5_2023-08-24T09:19:51.585793.parquet'
  - split: 2023_08_29T17_54_59.197645
    path:
    - '**/details_harness|hendrycksTest-prehistory|5_2023-08-29T17:54:59.197645.parquet'
  - split: 2023_09_15T09_53_02.418861
    path:
    - '**/details_harness|hendrycksTest-prehistory|5_2023-09-15T09-53-02.418861.parquet'
  - split: latest
    path:
    - '**/details_harness|hendrycksTest-prehistory|5_2023-09-15T09-53-02.418861.parquet'
- config_name: harness_hendrycksTest_professional_accounting_5
  data_files:
  - split: 2023_08_19T16_35_46.942696
    path:
    - '**/details_harness|hendrycksTest-professional_accounting|5_2023-08-19T16:35:46.942696.parquet'
  - split: 2023_08_24T09_19_51.585793
    path:
    - '**/details_harness|hendrycksTest-professional_accounting|5_2023-08-24T09:19:51.585793.parquet'
  - split: 2023_08_29T17_54_59.197645
    path:
    - '**/details_harness|hendrycksTest-professional_accounting|5_2023-08-29T17:54:59.197645.parquet'
  - split: 2023_09_15T09_53_02.418861
    path:
    - '**/details_harness|hendrycksTest-professional_accounting|5_2023-09-15T09-53-02.418861.parquet'
  - split: latest
    path:
    - '**/details_harness|hendrycksTest-professional_accounting|5_2023-09-15T09-53-02.418861.parquet'
- config_name: harness_hendrycksTest_professional_law_5
  data_files:
  - split: 2023_08_19T16_35_46.942696
    path:
    - '**/details_harness|hendrycksTest-professional_law|5_2023-08-19T16:35:46.942696.parquet'
  - split: 2023_08_24T09_19_51.585793
    path:
    - '**/details_harness|hendrycksTest-professional_law|5_2023-08-24T09:19:51.585793.parquet'
  - split: 2023_08_29T17_54_59.197645
    path:
    - '**/details_harness|hendrycksTest-professional_law|5_2023-08-29T17:54:59.197645.parquet'
  - split: 2023_09_15T09_53_02.418861
    path:
    - '**/details_harness|hendrycksTest-professional_law|5_2023-09-15T09-53-02.418861.parquet'
  - split: latest
    path:
    - '**/details_harness|hendrycksTest-professional_law|5_2023-09-15T09-53-02.418861.parquet'
- config_name: harness_hendrycksTest_professional_medicine_5
  data_files:
  - split: 2023_08_19T16_35_46.942696
    path:
    - '**/details_harness|hendrycksTest-professional_medicine|5_2023-08-19T16:35:46.942696.parquet'
  - split: 2023_08_24T09_19_51.585793
    path:
    - '**/details_harness|hendrycksTest-professional_medicine|5_2023-08-24T09:19:51.585793.parquet'
  - split: 2023_08_29T17_54_59.197645
    path:
    - '**/details_harness|hendrycksTest-professional_medicine|5_2023-08-29T17:54:59.197645.parquet'
  - split: 2023_09_15T09_53_02.418861
    path:
    - '**/details_harness|hendrycksTest-professional_medicine|5_2023-09-15T09-53-02.418861.parquet'
  - split: latest
    path:
    - '**/details_harness|hendrycksTest-professional_medicine|5_2023-09-15T09-53-02.418861.parquet'
- config_name: harness_hendrycksTest_professional_psychology_5
  data_files:
  - split: 2023_08_19T16_35_46.942696
    path:
    - '**/details_harness|hendrycksTest-professional_psychology|5_2023-08-19T16:35:46.942696.parquet'
  - split: 2023_08_24T09_19_51.585793
    path:
    - '**/details_harness|hendrycksTest-professional_psychology|5_2023-08-24T09:19:51.585793.parquet'
  - split: 2023_08_29T17_54_59.197645
    path:
    - '**/details_harness|hendrycksTest-professional_psychology|5_2023-08-29T17:54:59.197645.parquet'
  - split: 2023_09_15T09_53_02.418861
    path:
    - '**/details_harness|hendrycksTest-professional_psychology|5_2023-09-15T09-53-02.418861.parquet'
  - split: latest
    path:
    - '**/details_harness|hendrycksTest-professional_psychology|5_2023-09-15T09-53-02.418861.parquet'
- config_name: harness_hendrycksTest_public_relations_5
  data_files:
  - split: 2023_08_19T16_35_46.942696
    path:
    - '**/details_harness|hendrycksTest-public_relations|5_2023-08-19T16:35:46.942696.parquet'
  - split: 2023_08_24T09_19_51.585793
    path:
    - '**/details_harness|hendrycksTest-public_relations|5_2023-08-24T09:19:51.585793.parquet'
  - split: 2023_08_29T17_54_59.197645
    path:
    - '**/details_harness|hendrycksTest-public_relations|5_2023-08-29T17:54:59.197645.parquet'
  - split: 2023_09_15T09_53_02.418861
    path:
    - '**/details_harness|hendrycksTest-public_relations|5_2023-09-15T09-53-02.418861.parquet'
  - split: latest
    path:
    - '**/details_harness|hendrycksTest-public_relations|5_2023-09-15T09-53-02.418861.parquet'
- config_name: harness_hendrycksTest_security_studies_5
  data_files:
  - split: 2023_08_19T16_35_46.942696
    path:
    - '**/details_harness|hendrycksTest-security_studies|5_2023-08-19T16:35:46.942696.parquet'
  - split: 2023_08_24T09_19_51.585793
    path:
    - '**/details_harness|hendrycksTest-security_studies|5_2023-08-24T09:19:51.585793.parquet'
  - split: 2023_08_29T17_54_59.197645
    path:
    - '**/details_harness|hendrycksTest-security_studies|5_2023-08-29T17:54:59.197645.parquet'
  - split: 2023_09_15T09_53_02.418861
    path:
    - '**/details_harness|hendrycksTest-security_studies|5_2023-09-15T09-53-02.418861.parquet'
  - split: latest
    path:
    - '**/details_harness|hendrycksTest-security_studies|5_2023-09-15T09-53-02.418861.parquet'
- config_name: harness_hendrycksTest_sociology_5
  data_files:
  - split: 2023_08_19T16_35_46.942696
    path:
    - '**/details_harness|hendrycksTest-sociology|5_2023-08-19T16:35:46.942696.parquet'
  - split: 2023_08_24T09_19_51.585793
    path:
    - '**/details_harness|hendrycksTest-sociology|5_2023-08-24T09:19:51.585793.parquet'
  - split: 2023_08_29T17_54_59.197645
    path:
    - '**/details_harness|hendrycksTest-sociology|5_2023-08-29T17:54:59.197645.parquet'
  - split: 2023_09_15T09_53_02.418861
    path:
    - '**/details_harness|hendrycksTest-sociology|5_2023-09-15T09-53-02.418861.parquet'
  - split: latest
    path:
    - '**/details_harness|hendrycksTest-sociology|5_2023-09-15T09-53-02.418861.parquet'
- config_name: harness_hendrycksTest_us_foreign_policy_5
  data_files:
  - split: 2023_08_19T16_35_46.942696
    path:
    - '**/details_harness|hendrycksTest-us_foreign_policy|5_2023-08-19T16:35:46.942696.parquet'
  - split: 2023_08_24T09_19_51.585793
    path:
    - '**/details_harness|hendrycksTest-us_foreign_policy|5_2023-08-24T09:19:51.585793.parquet'
  - split: 2023_08_29T17_54_59.197645
    path:
    - '**/details_harness|hendrycksTest-us_foreign_policy|5_2023-08-29T17:54:59.197645.parquet'
  - split: 2023_09_15T09_53_02.418861
    path:
    - '**/details_harness|hendrycksTest-us_foreign_policy|5_2023-09-15T09-53-02.418861.parquet'
  - split: latest
    path:
    - '**/details_harness|hendrycksTest-us_foreign_policy|5_2023-09-15T09-53-02.418861.parquet'
- config_name: harness_hendrycksTest_virology_5
  data_files:
  - split: 2023_08_19T16_35_46.942696
    path:
    - '**/details_harness|hendrycksTest-virology|5_2023-08-19T16:35:46.942696.parquet'
  - split: 2023_08_24T09_19_51.585793
    path:
    - '**/details_harness|hendrycksTest-virology|5_2023-08-24T09:19:51.585793.parquet'
  - split: 2023_08_29T17_54_59.197645
    path:
    - '**/details_harness|hendrycksTest-virology|5_2023-08-29T17:54:59.197645.parquet'
  - split: 2023_09_15T09_53_02.418861
    path:
    - '**/details_harness|hendrycksTest-virology|5_2023-09-15T09-53-02.418861.parquet'
  - split: latest
    path:
    - '**/details_harness|hendrycksTest-virology|5_2023-09-15T09-53-02.418861.parquet'
- config_name: harness_hendrycksTest_world_religions_5
  data_files:
  - split: 2023_08_19T16_35_46.942696
    path:
    - '**/details_harness|hendrycksTest-world_religions|5_2023-08-19T16:35:46.942696.parquet'
  - split: 2023_08_24T09_19_51.585793
    path:
    - '**/details_harness|hendrycksTest-world_religions|5_2023-08-24T09:19:51.585793.parquet'
  - split: 2023_08_29T17_54_59.197645
    path:
    - '**/details_harness|hendrycksTest-world_religions|5_2023-08-29T17:54:59.197645.parquet'
  - split: 2023_09_15T09_53_02.418861
    path:
    - '**/details_harness|hendrycksTest-world_religions|5_2023-09-15T09-53-02.418861.parquet'
  - split: latest
    path:
    - '**/details_harness|hendrycksTest-world_religions|5_2023-09-15T09-53-02.418861.parquet'
- config_name: harness_truthfulqa_mc_0
  data_files:
  - split: 2023_08_19T16_35_46.942696
    path:
    - '**/details_harness|truthfulqa:mc|0_2023-08-19T16:35:46.942696.parquet'
  - split: 2023_08_24T09_19_51.585793
    path:
    - '**/details_harness|truthfulqa:mc|0_2023-08-24T09:19:51.585793.parquet'
  - split: 2023_08_29T17_54_59.197645
    path:
    - '**/details_harness|truthfulqa:mc|0_2023-08-29T17:54:59.197645.parquet'
  - split: 2023_09_15T09_53_02.418861
    path:
    - '**/details_harness|truthfulqa:mc|0_2023-09-15T09-53-02.418861.parquet'
  - split: latest
    path:
    - '**/details_harness|truthfulqa:mc|0_2023-09-15T09-53-02.418861.parquet'
- config_name: harness_winogrande_0
  data_files:
  - split: 2023_09_15T08_35_01.075146
    path:
    - '**/details_harness|winogrande|0_2023-09-15T08-35-01.075146.parquet'
  - split: latest
    path:
    - '**/details_harness|winogrande|0_2023-09-15T08-35-01.075146.parquet'
- config_name: harness_winogrande_5
  data_files:
  - split: 2023_09_08T17_00_44.389859
    path:
    - '**/details_harness|winogrande|5_2023-09-08T17-00-44.389859.parquet'
  - split: 2023_09_09T12_32_30.613622
    path:
    - '**/details_harness|winogrande|5_2023-09-09T12-32-30.613622.parquet'
  - split: 2023_09_20T14_39_46.791628
    path:
    - '**/details_harness|winogrande|5_2023-09-20T14-39-46.791628.parquet'
  - split: latest
    path:
    - '**/details_harness|winogrande|5_2023-09-20T14-39-46.791628.parquet'
- config_name: original_mmlu_5
  data_files:
  - split: 2023_08_28T19_52_01.926454
    path:
    - '**/details_original|mmlu:abstract_algebra|5_2023-08-28T19:52:01.926454.parquet'
    - '**/details_original|mmlu:anatomy|5_2023-08-28T19:52:01.926454.parquet'
    - '**/details_original|mmlu:astronomy|5_2023-08-28T19:52:01.926454.parquet'
    - '**/details_original|mmlu:business_ethics|5_2023-08-28T19:52:01.926454.parquet'
    - '**/details_original|mmlu:clinical_knowledge|5_2023-08-28T19:52:01.926454.parquet'
    - '**/details_original|mmlu:college_biology|5_2023-08-28T19:52:01.926454.parquet'
    - '**/details_original|mmlu:college_chemistry|5_2023-08-28T19:52:01.926454.parquet'
    - '**/details_original|mmlu:college_computer_science|5_2023-08-28T19:52:01.926454.parquet'
    - '**/details_original|mmlu:college_mathematics|5_2023-08-28T19:52:01.926454.parquet'
    - '**/details_original|mmlu:college_medicine|5_2023-08-28T19:52:01.926454.parquet'
    - '**/details_original|mmlu:college_physics|5_2023-08-28T19:52:01.926454.parquet'
    - '**/details_original|mmlu:computer_security|5_2023-08-28T19:52:01.926454.parquet'
    - '**/details_original|mmlu:conceptual_physics|5_2023-08-28T19:52:01.926454.parquet'
    - '**/details_original|mmlu:econometrics|5_2023-08-28T19:52:01.926454.parquet'
    - '**/details_original|mmlu:electrical_engineering|5_2023-08-28T19:52:01.926454.parquet'
    - '**/details_original|mmlu:elementary_mathematics|5_2023-08-28T19:52:01.926454.parquet'
    - '**/details_original|mmlu:formal_logic|5_2023-08-28T19:52:01.926454.parquet'
    - '**/details_original|mmlu:global_facts|5_2023-08-28T19:52:01.926454.parquet'
    - '**/details_original|mmlu:high_school_biology|5_2023-08-28T19:52:01.926454.parquet'
    - '**/details_original|mmlu:high_school_chemistry|5_2023-08-28T19:52:01.926454.parquet'
    - '**/details_original|mmlu:high_school_computer_science|5_2023-08-28T19:52:01.926454.parquet'
    - '**/details_original|mmlu:high_school_european_history|5_2023-08-28T19:52:01.926454.parquet'
    - '**/details_original|mmlu:high_school_geography|5_2023-08-28T19:52:01.926454.parquet'
    - '**/details_original|mmlu:high_school_government_and_politics|5_2023-08-28T19:52:01.926454.parquet'
    - '**/details_original|mmlu:high_school_macroeconomics|5_2023-08-28T19:52:01.926454.parquet'
    - '**/details_original|mmlu:high_school_mathematics|5_2023-08-28T19:52:01.926454.parquet'
    - '**/details_original|mmlu:high_school_microeconomics|5_2023-08-28T19:52:01.926454.parquet'
    - '**/details_original|mmlu:high_school_physics|5_2023-08-28T19:52:01.926454.parquet'
    - '**/details_original|mmlu:high_school_psychology|5_2023-08-28T19:52:01.926454.parquet'
    - '**/details_original|mmlu:high_school_statistics|5_2023-08-28T19:52:01.926454.parquet'
    - '**/details_original|mmlu:high_school_us_history|5_2023-08-28T19:52:01.926454.parquet'
    - '**/details_original|mmlu:high_school_world_history|5_2023-08-28T19:52:01.926454.parquet'
    - '**/details_original|mmlu:human_aging|5_2023-08-28T19:52:01.926454.parquet'
    - '**/details_original|mmlu:human_sexuality|5_2023-08-28T19:52:01.926454.parquet'
    - '**/details_original|mmlu:international_law|5_2023-08-28T19:52:01.926454.parquet'
    - '**/details_original|mmlu:jurisprudence|5_2023-08-28T19:52:01.926454.parquet'
    - '**/details_original|mmlu:logical_fallacies|5_2023-08-28T19:52:01.926454.parquet'
    - '**/details_original|mmlu:machine_learning|5_2023-08-28T19:52:01.926454.parquet'
    - '**/details_original|mmlu:management|5_2023-08-28T19:52:01.926454.parquet'
    - '**/details_original|mmlu:marketing|5_2023-08-28T19:52:01.926454.parquet'
    - '**/details_original|mmlu:medical_genetics|5_2023-08-28T19:52:01.926454.parquet'
    - '**/details_original|mmlu:miscellaneous|5_2023-08-28T19:52:01.926454.parquet'
    - '**/details_original|mmlu:moral_disputes|5_2023-08-28T19:52:01.926454.parquet'
    - '**/details_original|mmlu:moral_scenarios|5_2023-08-28T19:52:01.926454.parquet'
    - '**/details_original|mmlu:nutrition|5_2023-08-28T19:52:01.926454.parquet'
    - '**/details_original|mmlu:philosophy|5_2023-08-28T19:52:01.926454.parquet'
    - '**/details_original|mmlu:prehistory|5_2023-08-28T19:52:01.926454.parquet'
    - '**/details_original|mmlu:professional_accounting|5_2023-08-28T19:52:01.926454.parquet'
    - '**/details_original|mmlu:professional_law|5_2023-08-28T19:52:01.926454.parquet'
    - '**/details_original|mmlu:professional_medicine|5_2023-08-28T19:52:01.926454.parquet'
    - '**/details_original|mmlu:professional_psychology|5_2023-08-28T19:52:01.926454.parquet'
    - '**/details_original|mmlu:public_relations|5_2023-08-28T19:52:01.926454.parquet'
    - '**/details_original|mmlu:security_studies|5_2023-08-28T19:52:01.926454.parquet'
    - '**/details_original|mmlu:sociology|5_2023-08-28T19:52:01.926454.parquet'
    - '**/details_original|mmlu:us_foreign_policy|5_2023-08-28T19:52:01.926454.parquet'
    - '**/details_original|mmlu:virology|5_2023-08-28T19:52:01.926454.parquet'
    - '**/details_original|mmlu:world_religions|5_2023-08-28T19:52:01.926454.parquet'
  - split: latest
    path:
    - '**/details_original|mmlu:abstract_algebra|5_2023-08-28T19:52:01.926454.parquet'
    - '**/details_original|mmlu:anatomy|5_2023-08-28T19:52:01.926454.parquet'
    - '**/details_original|mmlu:astronomy|5_2023-08-28T19:52:01.926454.parquet'
    - '**/details_original|mmlu:business_ethics|5_2023-08-28T19:52:01.926454.parquet'
    - '**/details_original|mmlu:clinical_knowledge|5_2023-08-28T19:52:01.926454.parquet'
    - '**/details_original|mmlu:college_biology|5_2023-08-28T19:52:01.926454.parquet'
    - '**/details_original|mmlu:college_chemistry|5_2023-08-28T19:52:01.926454.parquet'
    - '**/details_original|mmlu:college_computer_science|5_2023-08-28T19:52:01.926454.parquet'
    - '**/details_original|mmlu:college_mathematics|5_2023-08-28T19:52:01.926454.parquet'
    - '**/details_original|mmlu:college_medicine|5_2023-08-28T19:52:01.926454.parquet'
    - '**/details_original|mmlu:college_physics|5_2023-08-28T19:52:01.926454.parquet'
    - '**/details_original|mmlu:computer_security|5_2023-08-28T19:52:01.926454.parquet'
    - '**/details_original|mmlu:conceptual_physics|5_2023-08-28T19:52:01.926454.parquet'
    - '**/details_original|mmlu:econometrics|5_2023-08-28T19:52:01.926454.parquet'
    - '**/details_original|mmlu:electrical_engineering|5_2023-08-28T19:52:01.926454.parquet'
    - '**/details_original|mmlu:elementary_mathematics|5_2023-08-28T19:52:01.926454.parquet'
    - '**/details_original|mmlu:formal_logic|5_2023-08-28T19:52:01.926454.parquet'
    - '**/details_original|mmlu:global_facts|5_2023-08-28T19:52:01.926454.parquet'
    - '**/details_original|mmlu:high_school_biology|5_2023-08-28T19:52:01.926454.parquet'
    - '**/details_original|mmlu:high_school_chemistry|5_2023-08-28T19:52:01.926454.parquet'
    - '**/details_original|mmlu:high_school_computer_science|5_2023-08-28T19:52:01.926454.parquet'
    - '**/details_original|mmlu:high_school_european_history|5_2023-08-28T19:52:01.926454.parquet'
    - '**/details_original|mmlu:high_school_geography|5_2023-08-28T19:52:01.926454.parquet'
    - '**/details_original|mmlu:high_school_government_and_politics|5_2023-08-28T19:52:01.926454.parquet'
    - '**/details_original|mmlu:high_school_macroeconomics|5_2023-08-28T19:52:01.926454.parquet'
    - '**/details_original|mmlu:high_school_mathematics|5_2023-08-28T19:52:01.926454.parquet'
    - '**/details_original|mmlu:high_school_microeconomics|5_2023-08-28T19:52:01.926454.parquet'
    - '**/details_original|mmlu:high_school_physics|5_2023-08-28T19:52:01.926454.parquet'
    - '**/details_original|mmlu:high_school_psychology|5_2023-08-28T19:52:01.926454.parquet'
    - '**/details_original|mmlu:high_school_statistics|5_2023-08-28T19:52:01.926454.parquet'
    - '**/details_original|mmlu:high_school_us_history|5_2023-08-28T19:52:01.926454.parquet'
    - '**/details_original|mmlu:high_school_world_history|5_2023-08-28T19:52:01.926454.parquet'
    - '**/details_original|mmlu:human_aging|5_2023-08-28T19:52:01.926454.parquet'
    - '**/details_original|mmlu:human_sexuality|5_2023-08-28T19:52:01.926454.parquet'
    - '**/details_original|mmlu:international_law|5_2023-08-28T19:52:01.926454.parquet'
    - '**/details_original|mmlu:jurisprudence|5_2023-08-28T19:52:01.926454.parquet'
    - '**/details_original|mmlu:logical_fallacies|5_2023-08-28T19:52:01.926454.parquet'
    - '**/details_original|mmlu:machine_learning|5_2023-08-28T19:52:01.926454.parquet'
    - '**/details_original|mmlu:management|5_2023-08-28T19:52:01.926454.parquet'
    - '**/details_original|mmlu:marketing|5_2023-08-28T19:52:01.926454.parquet'
    - '**/details_original|mmlu:medical_genetics|5_2023-08-28T19:52:01.926454.parquet'
    - '**/details_original|mmlu:miscellaneous|5_2023-08-28T19:52:01.926454.parquet'
    - '**/details_original|mmlu:moral_disputes|5_2023-08-28T19:52:01.926454.parquet'
    - '**/details_original|mmlu:moral_scenarios|5_2023-08-28T19:52:01.926454.parquet'
    - '**/details_original|mmlu:nutrition|5_2023-08-28T19:52:01.926454.parquet'
    - '**/details_original|mmlu:philosophy|5_2023-08-28T19:52:01.926454.parquet'
    - '**/details_original|mmlu:prehistory|5_2023-08-28T19:52:01.926454.parquet'
    - '**/details_original|mmlu:professional_accounting|5_2023-08-28T19:52:01.926454.parquet'
    - '**/details_original|mmlu:professional_law|5_2023-08-28T19:52:01.926454.parquet'
    - '**/details_original|mmlu:professional_medicine|5_2023-08-28T19:52:01.926454.parquet'
    - '**/details_original|mmlu:professional_psychology|5_2023-08-28T19:52:01.926454.parquet'
    - '**/details_original|mmlu:public_relations|5_2023-08-28T19:52:01.926454.parquet'
    - '**/details_original|mmlu:security_studies|5_2023-08-28T19:52:01.926454.parquet'
    - '**/details_original|mmlu:sociology|5_2023-08-28T19:52:01.926454.parquet'
    - '**/details_original|mmlu:us_foreign_policy|5_2023-08-28T19:52:01.926454.parquet'
    - '**/details_original|mmlu:virology|5_2023-08-28T19:52:01.926454.parquet'
    - '**/details_original|mmlu:world_religions|5_2023-08-28T19:52:01.926454.parquet'
- config_name: original_mmlu_abstract_algebra_5
  data_files:
  - split: 2023_08_28T19_52_01.926454
    path:
    - '**/details_original|mmlu:abstract_algebra|5_2023-08-28T19:52:01.926454.parquet'
  - split: latest
    path:
    - '**/details_original|mmlu:abstract_algebra|5_2023-08-28T19:52:01.926454.parquet'
- config_name: original_mmlu_anatomy_5
  data_files:
  - split: 2023_08_28T19_52_01.926454
    path:
    - '**/details_original|mmlu:anatomy|5_2023-08-28T19:52:01.926454.parquet'
  - split: latest
    path:
    - '**/details_original|mmlu:anatomy|5_2023-08-28T19:52:01.926454.parquet'
- config_name: original_mmlu_astronomy_5
  data_files:
  - split: 2023_08_28T19_52_01.926454
    path:
    - '**/details_original|mmlu:astronomy|5_2023-08-28T19:52:01.926454.parquet'
  - split: latest
    path:
    - '**/details_original|mmlu:astronomy|5_2023-08-28T19:52:01.926454.parquet'
- config_name: original_mmlu_business_ethics_5
  data_files:
  - split: 2023_08_28T19_52_01.926454
    path:
    - '**/details_original|mmlu:business_ethics|5_2023-08-28T19:52:01.926454.parquet'
  - split: latest
    path:
    - '**/details_original|mmlu:business_ethics|5_2023-08-28T19:52:01.926454.parquet'
- config_name: original_mmlu_clinical_knowledge_5
  data_files:
  - split: 2023_08_28T19_52_01.926454
    path:
    - '**/details_original|mmlu:clinical_knowledge|5_2023-08-28T19:52:01.926454.parquet'
  - split: latest
    path:
    - '**/details_original|mmlu:clinical_knowledge|5_2023-08-28T19:52:01.926454.parquet'
- config_name: original_mmlu_college_biology_5
  data_files:
  - split: 2023_08_28T19_52_01.926454
    path:
    - '**/details_original|mmlu:college_biology|5_2023-08-28T19:52:01.926454.parquet'
  - split: latest
    path:
    - '**/details_original|mmlu:college_biology|5_2023-08-28T19:52:01.926454.parquet'
- config_name: original_mmlu_college_chemistry_5
  data_files:
  - split: 2023_08_28T19_52_01.926454
    path:
    - '**/details_original|mmlu:college_chemistry|5_2023-08-28T19:52:01.926454.parquet'
  - split: latest
    path:
    - '**/details_original|mmlu:college_chemistry|5_2023-08-28T19:52:01.926454.parquet'
- config_name: original_mmlu_college_computer_science_5
  data_files:
  - split: 2023_08_28T19_52_01.926454
    path:
    - '**/details_original|mmlu:college_computer_science|5_2023-08-28T19:52:01.926454.parquet'
  - split: latest
    path:
    - '**/details_original|mmlu:college_computer_science|5_2023-08-28T19:52:01.926454.parquet'
- config_name: original_mmlu_college_mathematics_5
  data_files:
  - split: 2023_08_28T19_52_01.926454
    path:
    - '**/details_original|mmlu:college_mathematics|5_2023-08-28T19:52:01.926454.parquet'
  - split: latest
    path:
    - '**/details_original|mmlu:college_mathematics|5_2023-08-28T19:52:01.926454.parquet'
- config_name: original_mmlu_college_medicine_5
  data_files:
  - split: 2023_08_28T19_52_01.926454
    path:
    - '**/details_original|mmlu:college_medicine|5_2023-08-28T19:52:01.926454.parquet'
  - split: latest
    path:
    - '**/details_original|mmlu:college_medicine|5_2023-08-28T19:52:01.926454.parquet'
- config_name: original_mmlu_college_physics_5
  data_files:
  - split: 2023_08_28T19_52_01.926454
    path:
    - '**/details_original|mmlu:college_physics|5_2023-08-28T19:52:01.926454.parquet'
  - split: latest
    path:
    - '**/details_original|mmlu:college_physics|5_2023-08-28T19:52:01.926454.parquet'
- config_name: original_mmlu_computer_security_5
  data_files:
  - split: 2023_08_28T19_52_01.926454
    path:
    - '**/details_original|mmlu:computer_security|5_2023-08-28T19:52:01.926454.parquet'
  - split: latest
    path:
    - '**/details_original|mmlu:computer_security|5_2023-08-28T19:52:01.926454.parquet'
- config_name: original_mmlu_conceptual_physics_5
  data_files:
  - split: 2023_08_28T19_52_01.926454
    path:
    - '**/details_original|mmlu:conceptual_physics|5_2023-08-28T19:52:01.926454.parquet'
  - split: latest
    path:
    - '**/details_original|mmlu:conceptual_physics|5_2023-08-28T19:52:01.926454.parquet'
- config_name: original_mmlu_econometrics_5
  data_files:
  - split: 2023_08_28T19_52_01.926454
    path:
    - '**/details_original|mmlu:econometrics|5_2023-08-28T19:52:01.926454.parquet'
  - split: latest
    path:
    - '**/details_original|mmlu:econometrics|5_2023-08-28T19:52:01.926454.parquet'
- config_name: original_mmlu_electrical_engineering_5
  data_files:
  - split: 2023_08_28T19_52_01.926454
    path:
    - '**/details_original|mmlu:electrical_engineering|5_2023-08-28T19:52:01.926454.parquet'
  - split: latest
    path:
    - '**/details_original|mmlu:electrical_engineering|5_2023-08-28T19:52:01.926454.parquet'
- config_name: original_mmlu_elementary_mathematics_5
  data_files:
  - split: 2023_08_28T19_52_01.926454
    path:
    - '**/details_original|mmlu:elementary_mathematics|5_2023-08-28T19:52:01.926454.parquet'
  - split: latest
    path:
    - '**/details_original|mmlu:elementary_mathematics|5_2023-08-28T19:52:01.926454.parquet'
- config_name: original_mmlu_formal_logic_5
  data_files:
  - split: 2023_08_28T19_52_01.926454
    path:
    - '**/details_original|mmlu:formal_logic|5_2023-08-28T19:52:01.926454.parquet'
  - split: latest
    path:
    - '**/details_original|mmlu:formal_logic|5_2023-08-28T19:52:01.926454.parquet'
- config_name: original_mmlu_global_facts_5
  data_files:
  - split: 2023_08_28T19_52_01.926454
    path:
    - '**/details_original|mmlu:global_facts|5_2023-08-28T19:52:01.926454.parquet'
  - split: latest
    path:
    - '**/details_original|mmlu:global_facts|5_2023-08-28T19:52:01.926454.parquet'
- config_name: original_mmlu_high_school_biology_5
  data_files:
  - split: 2023_08_28T19_52_01.926454
    path:
    - '**/details_original|mmlu:high_school_biology|5_2023-08-28T19:52:01.926454.parquet'
  - split: latest
    path:
    - '**/details_original|mmlu:high_school_biology|5_2023-08-28T19:52:01.926454.parquet'
- config_name: original_mmlu_high_school_chemistry_5
  data_files:
  - split: 2023_08_28T19_52_01.926454
    path:
    - '**/details_original|mmlu:high_school_chemistry|5_2023-08-28T19:52:01.926454.parquet'
  - split: latest
    path:
    - '**/details_original|mmlu:high_school_chemistry|5_2023-08-28T19:52:01.926454.parquet'
- config_name: original_mmlu_high_school_computer_science_5
  data_files:
  - split: 2023_08_28T19_52_01.926454
    path:
    - '**/details_original|mmlu:high_school_computer_science|5_2023-08-28T19:52:01.926454.parquet'
  - split: latest
    path:
    - '**/details_original|mmlu:high_school_computer_science|5_2023-08-28T19:52:01.926454.parquet'
- config_name: original_mmlu_high_school_european_history_5
  data_files:
  - split: 2023_08_28T19_52_01.926454
    path:
    - '**/details_original|mmlu:high_school_european_history|5_2023-08-28T19:52:01.926454.parquet'
  - split: latest
    path:
    - '**/details_original|mmlu:high_school_european_history|5_2023-08-28T19:52:01.926454.parquet'
- config_name: original_mmlu_high_school_geography_5
  data_files:
  - split: 2023_08_28T19_52_01.926454
    path:
    - '**/details_original|mmlu:high_school_geography|5_2023-08-28T19:52:01.926454.parquet'
  - split: latest
    path:
    - '**/details_original|mmlu:high_school_geography|5_2023-08-28T19:52:01.926454.parquet'
- config_name: original_mmlu_high_school_government_and_politics_5
  data_files:
  - split: 2023_08_28T19_52_01.926454
    path:
    - '**/details_original|mmlu:high_school_government_and_politics|5_2023-08-28T19:52:01.926454.parquet'
  - split: latest
    path:
    - '**/details_original|mmlu:high_school_government_and_politics|5_2023-08-28T19:52:01.926454.parquet'
- config_name: original_mmlu_high_school_macroeconomics_5
  data_files:
  - split: 2023_08_28T19_52_01.926454
    path:
    - '**/details_original|mmlu:high_school_macroeconomics|5_2023-08-28T19:52:01.926454.parquet'
  - split: latest
    path:
    - '**/details_original|mmlu:high_school_macroeconomics|5_2023-08-28T19:52:01.926454.parquet'
- config_name: original_mmlu_high_school_mathematics_5
  data_files:
  - split: 2023_08_28T19_52_01.926454
    path:
    - '**/details_original|mmlu:high_school_mathematics|5_2023-08-28T19:52:01.926454.parquet'
  - split: latest
    path:
    - '**/details_original|mmlu:high_school_mathematics|5_2023-08-28T19:52:01.926454.parquet'
- config_name: original_mmlu_high_school_microeconomics_5
  data_files:
  - split: 2023_08_28T19_52_01.926454
    path:
    - '**/details_original|mmlu:high_school_microeconomics|5_2023-08-28T19:52:01.926454.parquet'
  - split: latest
    path:
    - '**/details_original|mmlu:high_school_microeconomics|5_2023-08-28T19:52:01.926454.parquet'
- config_name: original_mmlu_high_school_physics_5
  data_files:
  - split: 2023_08_28T19_52_01.926454
    path:
    - '**/details_original|mmlu:high_school_physics|5_2023-08-28T19:52:01.926454.parquet'
  - split: latest
    path:
    - '**/details_original|mmlu:high_school_physics|5_2023-08-28T19:52:01.926454.parquet'
- config_name: original_mmlu_high_school_psychology_5
  data_files:
  - split: 2023_08_28T19_52_01.926454
    path:
    - '**/details_original|mmlu:high_school_psychology|5_2023-08-28T19:52:01.926454.parquet'
  - split: latest
    path:
    - '**/details_original|mmlu:high_school_psychology|5_2023-08-28T19:52:01.926454.parquet'
- config_name: original_mmlu_high_school_statistics_5
  data_files:
  - split: 2023_08_28T19_52_01.926454
    path:
    - '**/details_original|mmlu:high_school_statistics|5_2023-08-28T19:52:01.926454.parquet'
  - split: latest
    path:
    - '**/details_original|mmlu:high_school_statistics|5_2023-08-28T19:52:01.926454.parquet'
- config_name: original_mmlu_high_school_us_history_5
  data_files:
  - split: 2023_08_28T19_52_01.926454
    path:
    - '**/details_original|mmlu:high_school_us_history|5_2023-08-28T19:52:01.926454.parquet'
  - split: latest
    path:
    - '**/details_original|mmlu:high_school_us_history|5_2023-08-28T19:52:01.926454.parquet'
- config_name: original_mmlu_high_school_world_history_5
  data_files:
  - split: 2023_08_28T19_52_01.926454
    path:
    - '**/details_original|mmlu:high_school_world_history|5_2023-08-28T19:52:01.926454.parquet'
  - split: latest
    path:
    - '**/details_original|mmlu:high_school_world_history|5_2023-08-28T19:52:01.926454.parquet'
- config_name: original_mmlu_human_aging_5
  data_files:
  - split: 2023_08_28T19_52_01.926454
    path:
    - '**/details_original|mmlu:human_aging|5_2023-08-28T19:52:01.926454.parquet'
  - split: latest
    path:
    - '**/details_original|mmlu:human_aging|5_2023-08-28T19:52:01.926454.parquet'
- config_name: original_mmlu_human_sexuality_5
  data_files:
  - split: 2023_08_28T19_52_01.926454
    path:
    - '**/details_original|mmlu:human_sexuality|5_2023-08-28T19:52:01.926454.parquet'
  - split: latest
    path:
    - '**/details_original|mmlu:human_sexuality|5_2023-08-28T19:52:01.926454.parquet'
- config_name: original_mmlu_international_law_5
  data_files:
  - split: 2023_08_28T19_52_01.926454
    path:
    - '**/details_original|mmlu:international_law|5_2023-08-28T19:52:01.926454.parquet'
  - split: latest
    path:
    - '**/details_original|mmlu:international_law|5_2023-08-28T19:52:01.926454.parquet'
- config_name: original_mmlu_jurisprudence_5
  data_files:
  - split: 2023_08_28T19_52_01.926454
    path:
    - '**/details_original|mmlu:jurisprudence|5_2023-08-28T19:52:01.926454.parquet'
  - split: latest
    path:
    - '**/details_original|mmlu:jurisprudence|5_2023-08-28T19:52:01.926454.parquet'
- config_name: original_mmlu_logical_fallacies_5
  data_files:
  - split: 2023_08_28T19_52_01.926454
    path:
    - '**/details_original|mmlu:logical_fallacies|5_2023-08-28T19:52:01.926454.parquet'
  - split: latest
    path:
    - '**/details_original|mmlu:logical_fallacies|5_2023-08-28T19:52:01.926454.parquet'
- config_name: original_mmlu_machine_learning_5
  data_files:
  - split: 2023_08_28T19_52_01.926454
    path:
    - '**/details_original|mmlu:machine_learning|5_2023-08-28T19:52:01.926454.parquet'
  - split: latest
    path:
    - '**/details_original|mmlu:machine_learning|5_2023-08-28T19:52:01.926454.parquet'
- config_name: original_mmlu_management_5
  data_files:
  - split: 2023_08_28T19_52_01.926454
    path:
    - '**/details_original|mmlu:management|5_2023-08-28T19:52:01.926454.parquet'
  - split: latest
    path:
    - '**/details_original|mmlu:management|5_2023-08-28T19:52:01.926454.parquet'
- config_name: original_mmlu_marketing_5
  data_files:
  - split: 2023_08_28T19_52_01.926454
    path:
    - '**/details_original|mmlu:marketing|5_2023-08-28T19:52:01.926454.parquet'
  - split: latest
    path:
    - '**/details_original|mmlu:marketing|5_2023-08-28T19:52:01.926454.parquet'
- config_name: original_mmlu_medical_genetics_5
  data_files:
  - split: 2023_08_28T19_52_01.926454
    path:
    - '**/details_original|mmlu:medical_genetics|5_2023-08-28T19:52:01.926454.parquet'
  - split: latest
    path:
    - '**/details_original|mmlu:medical_genetics|5_2023-08-28T19:52:01.926454.parquet'
- config_name: original_mmlu_miscellaneous_5
  data_files:
  - split: 2023_08_28T19_52_01.926454
    path:
    - '**/details_original|mmlu:miscellaneous|5_2023-08-28T19:52:01.926454.parquet'
  - split: latest
    path:
    - '**/details_original|mmlu:miscellaneous|5_2023-08-28T19:52:01.926454.parquet'
- config_name: original_mmlu_moral_disputes_5
  data_files:
  - split: 2023_08_28T19_52_01.926454
    path:
    - '**/details_original|mmlu:moral_disputes|5_2023-08-28T19:52:01.926454.parquet'
  - split: latest
    path:
    - '**/details_original|mmlu:moral_disputes|5_2023-08-28T19:52:01.926454.parquet'
- config_name: original_mmlu_moral_scenarios_5
  data_files:
  - split: 2023_08_28T19_52_01.926454
    path:
    - '**/details_original|mmlu:moral_scenarios|5_2023-08-28T19:52:01.926454.parquet'
  - split: latest
    path:
    - '**/details_original|mmlu:moral_scenarios|5_2023-08-28T19:52:01.926454.parquet'
- config_name: original_mmlu_nutrition_5
  data_files:
  - split: 2023_08_28T19_52_01.926454
    path:
    - '**/details_original|mmlu:nutrition|5_2023-08-28T19:52:01.926454.parquet'
  - split: latest
    path:
    - '**/details_original|mmlu:nutrition|5_2023-08-28T19:52:01.926454.parquet'
- config_name: original_mmlu_philosophy_5
  data_files:
  - split: 2023_08_28T19_52_01.926454
    path:
    - '**/details_original|mmlu:philosophy|5_2023-08-28T19:52:01.926454.parquet'
  - split: latest
    path:
    - '**/details_original|mmlu:philosophy|5_2023-08-28T19:52:01.926454.parquet'
- config_name: original_mmlu_prehistory_5
  data_files:
  - split: 2023_08_28T19_52_01.926454
    path:
    - '**/details_original|mmlu:prehistory|5_2023-08-28T19:52:01.926454.parquet'
  - split: latest
    path:
    - '**/details_original|mmlu:prehistory|5_2023-08-28T19:52:01.926454.parquet'
- config_name: original_mmlu_professional_accounting_5
  data_files:
  - split: 2023_08_28T19_52_01.926454
    path:
    - '**/details_original|mmlu:professional_accounting|5_2023-08-28T19:52:01.926454.parquet'
  - split: latest
    path:
    - '**/details_original|mmlu:professional_accounting|5_2023-08-28T19:52:01.926454.parquet'
- config_name: original_mmlu_professional_law_5
  data_files:
  - split: 2023_08_28T19_52_01.926454
    path:
    - '**/details_original|mmlu:professional_law|5_2023-08-28T19:52:01.926454.parquet'
  - split: latest
    path:
    - '**/details_original|mmlu:professional_law|5_2023-08-28T19:52:01.926454.parquet'
- config_name: original_mmlu_professional_medicine_5
  data_files:
  - split: 2023_08_28T19_52_01.926454
    path:
    - '**/details_original|mmlu:professional_medicine|5_2023-08-28T19:52:01.926454.parquet'
  - split: latest
    path:
    - '**/details_original|mmlu:professional_medicine|5_2023-08-28T19:52:01.926454.parquet'
- config_name: original_mmlu_professional_psychology_5
  data_files:
  - split: 2023_08_28T19_52_01.926454
    path:
    - '**/details_original|mmlu:professional_psychology|5_2023-08-28T19:52:01.926454.parquet'
  - split: latest
    path:
    - '**/details_original|mmlu:professional_psychology|5_2023-08-28T19:52:01.926454.parquet'
- config_name: original_mmlu_public_relations_5
  data_files:
  - split: 2023_08_28T19_52_01.926454
    path:
    - '**/details_original|mmlu:public_relations|5_2023-08-28T19:52:01.926454.parquet'
  - split: latest
    path:
    - '**/details_original|mmlu:public_relations|5_2023-08-28T19:52:01.926454.parquet'
- config_name: original_mmlu_security_studies_5
  data_files:
  - split: 2023_08_28T19_52_01.926454
    path:
    - '**/details_original|mmlu:security_studies|5_2023-08-28T19:52:01.926454.parquet'
  - split: latest
    path:
    - '**/details_original|mmlu:security_studies|5_2023-08-28T19:52:01.926454.parquet'
- config_name: original_mmlu_sociology_5
  data_files:
  - split: 2023_08_28T19_52_01.926454
    path:
    - '**/details_original|mmlu:sociology|5_2023-08-28T19:52:01.926454.parquet'
  - split: latest
    path:
    - '**/details_original|mmlu:sociology|5_2023-08-28T19:52:01.926454.parquet'
- config_name: original_mmlu_us_foreign_policy_5
  data_files:
  - split: 2023_08_28T19_52_01.926454
    path:
    - '**/details_original|mmlu:us_foreign_policy|5_2023-08-28T19:52:01.926454.parquet'
  - split: latest
    path:
    - '**/details_original|mmlu:us_foreign_policy|5_2023-08-28T19:52:01.926454.parquet'
- config_name: original_mmlu_virology_5
  data_files:
  - split: 2023_08_28T19_52_01.926454
    path:
    - '**/details_original|mmlu:virology|5_2023-08-28T19:52:01.926454.parquet'
  - split: latest
    path:
    - '**/details_original|mmlu:virology|5_2023-08-28T19:52:01.926454.parquet'
- config_name: original_mmlu_world_religions_5
  data_files:
  - split: 2023_08_28T19_52_01.926454
    path:
    - '**/details_original|mmlu:world_religions|5_2023-08-28T19:52:01.926454.parquet'
  - split: latest
    path:
    - '**/details_original|mmlu:world_religions|5_2023-08-28T19:52:01.926454.parquet'
- config_name: results
  data_files:
  - split: 2023_08_19T16_35_46.942696
    path:
    - results_2023-08-19T16:35:46.942696.parquet
  - split: 2023_08_21T17_55_50.567332
    path:
    - results_2023-08-21T17:55:50.567332.parquet
  - split: 2023_08_21T20_09_03.352670
    path:
    - results_2023-08-21T20:09:03.352670.parquet
  - split: 2023_08_21T20_15_29.093529
    path:
    - results_2023-08-21T20:15:29.093529.parquet
  - split: 2023_08_21T20_20_08.261679
    path:
    - results_2023-08-21T20:20:08.261679.parquet
  - split: 2023_08_24T09_19_51.585793
    path:
    - results_2023-08-24T09:19:51.585793.parquet
  - split: 2023_08_28T19_52_01.926454
    path:
    - results_2023-08-28T19:52:01.926454.parquet
  - split: 2023_08_29T17_54_59.197645
    path:
    - results_2023-08-29T17:54:59.197645.parquet
  - split: 2023_09_08T17_00_44.389859
    path:
    - results_2023-09-08T17-00-44.389859.parquet
  - split: 2023_09_09T12_32_30.613622
    path:
    - results_2023-09-09T12-32-30.613622.parquet
  - split: 2023_09_14T20_50_38.766533
    path:
    - results_2023-09-14T20-50-38.766533.parquet
  - split: 2023_09_15T08_35_01.075146
    path:
    - results_2023-09-15T08-35-01.075146.parquet
  - split: 2023_09_15T09_53_02.418861
    path:
    - results_2023-09-15T09-53-02.418861.parquet
  - split: 2023_09_20T14_39_46.791628
    path:
    - results_2023-09-20T14-39-46.791628.parquet
  - split: 2023_12_02T13_00_06.695936
    path:
    - results_2023-12-02T13-00-06.695936.parquet
  - split: 2023_12_02T13_00_54.924067
    path:
    - results_2023-12-02T13-00-54.924067.parquet
  - split: latest
    path:
    - results_2023-12-02T13-00-54.924067.parquet
---

# Dataset Card for Evaluation run of meta-llama/Llama-2-7b-hf

## Dataset Description

- **Homepage:** 
- **Repository:** https://huggingface.co/meta-llama/Llama-2-7b-hf
- **Paper:** 
- **Leaderboard:** https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard
- **Point of Contact:** clementine@hf.co

### Dataset Summary

Dataset automatically created during the evaluation run of model [meta-llama/Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf) on the [Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard).

The dataset is composed of 127 configuration, each one coresponding to one of the evaluated task.

The dataset has been created from 16 run(s). Each run can be found as a specific split in each configuration, the split being named using the timestamp of the run.The "train" split is always pointing to the latest results.

An additional configuration "results" store all the aggregated results of the run (and is used to compute and display the aggregated metrics on the [Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)).

To load the details from a run, you can for instance do the following:
```python
from datasets import load_dataset
data = load_dataset("open-llm-leaderboard/details_meta-llama__Llama-2-7b-hf",
	"harness_gsm8k_5",
	split="train")
```

## Latest results

These are the [latest results from run 2023-12-02T13:00:54.924067](https://huggingface.co/datasets/open-llm-leaderboard/details_meta-llama__Llama-2-7b-hf/blob/main/results_2023-12-02T13-00-54.924067.json)(note that their might be results for other tasks in the repos if successive evals didn't cover the same tasks. You find each in the results and the "latest" split for each eval):

```python
{
    "all": {
        "acc": 0.14480667172100076,
        "acc_stderr": 0.009693234799052708
    },
    "harness|gsm8k|5": {
        "acc": 0.14480667172100076,
        "acc_stderr": 0.009693234799052708
    }
}
```

### Supported Tasks and Leaderboards

[More Information Needed]

### Languages

[More Information Needed]

## Dataset Structure

### Data Instances

[More Information Needed]

### Data Fields

[More Information Needed]

### Data Splits

[More Information Needed]

## Dataset Creation

### Curation Rationale

[More Information Needed]

### Source Data

#### Initial Data Collection and Normalization

[More Information Needed]

#### Who are the source language producers?

[More Information Needed]

### Annotations

#### Annotation process

[More Information Needed]

#### Who are the annotators?

[More Information Needed]

### Personal and Sensitive Information

[More Information Needed]

## Considerations for Using the Data

### Social Impact of Dataset

[More Information Needed]

### Discussion of Biases

[More Information Needed]

### Other Known Limitations

[More Information Needed]

## Additional Information

### Dataset Curators

[More Information Needed]

### Licensing Information

[More Information Needed]

### Citation Information

[More Information Needed]

### Contributions

[More Information Needed]