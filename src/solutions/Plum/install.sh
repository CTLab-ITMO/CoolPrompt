conda create -n plum python=3.9 -y
conda activate plum
pip install -r requirements.txt
mkdir logs output
python -c "import nltk;nltk.download('punkt')"
