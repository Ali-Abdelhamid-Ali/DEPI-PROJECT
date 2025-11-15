
1. `conda create -n  DEVTUNE python=3.11`
2. `conda activate DEVTUNE`

```
conda create --name new_environment_name --clone existing_environment_name
conda remove --name <environment_name> --all
```

requirements.txt

3. `pip install -r requirements.txt`

4. `uvicorn main:app --reload --host 0.0.0.0 --port 5000`