PROJECT_ROOT=.

echo "Downloading..."
$PROJECT_ROOT/.venv/bin/python3 $PROJECT_ROOT/../../utils/scripts/dataset_downloading.py --login sbolpro --password Sbolnginx1314! --save_path ~/autoprompting_datasets.zip

unzip ~/autoprompting_datasets.zip -d ~

rm ~/autoprompting_datasets.zip