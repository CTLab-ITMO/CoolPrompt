PROJECT_ROOT=~/CoolPrompt

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --login) login="$2"; shift ;;
        --password) password="$2"; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

if [[ -z "$login" || -z "$password" ]]; then
    echo "Usage: $0 --login <login> --password <password>"
    exit 1
fi

DATA_PATH=$PROJECT_ROOT/data

echo "Downloading..."
python3 $PROJECT_ROOT/src/utils/scripts/dataset_downloading.py --login $login --password $password --save_path $DATA_PATH/datasets.zip

unzip $DATA_PATH/datasets.zip -d $DATA_PATH

rm $DATA_PATH/data/*.json
rm $DATA_PATH/data/*.py

mv $DATA_PATH/data/* $DATA_PATH

rmdir $DATA_PATH/data

rm $DATA_PATH/datasets.zip
