# Venv - Create
uv venv .venv_atari
uv pip install -r requirements.txt

# Venv - Create
source .venv_atari/Scripts/activate

# Venv - Update
uv pip freeze > requirements.txt
uv pip install -r requirements.txt

