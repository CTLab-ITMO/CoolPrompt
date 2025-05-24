## Data

CoolPrompt is working with custom NLP benchmark. You can download it from our minio storage using [downloading script](https://github.com/CTLab-ITMO/CoolPrompt/blob/stage/scripts/dataset_downloading.sh) (remember, you need to use credentials to access it).
All data will be downloaded to *~/autoprompting_data*. You can change prompt templates and basic prompts for each dataset manually by editing configuration files there. 
To work with data you need to create custom dataset classes ([SST2Dataset](https://github.com/CTLab-ITMO/CoolPrompt/blob/b088b72de5e9405a720bb4d7157afd9b42ce767f/src/data/classification/sst2_dataset.py#L6) and etc.) using split='train' or 'test' ('test' is by default). You can read about all other parameters in docstrings.
The examples of using benchmark can be found in this [notebook](https://github.com/CTLab-ITMO/CoolPrompt/blob/stage/notebooks/examples/datasets_usage.ipynb).