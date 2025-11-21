
1. `conda create -n  DEVTUNE python=3.11`
2. `conda activate DEVTUNE`

```
conda create --name new_environment_name --clone existing_environment_name
conda remove --name <environment_name> --all
```

requirements.txt

3. `pip install -r requirements.txt`

4. `uvicorn main:app --reload --host 0.0.0.0 --port 5555`



5.contrib

```
# Assuming you're starting from scratch
git clone https://github.com/Ali-Abdelhamid-Ali/DEPI-PROJECT.git
cd DEPI-PROJECT

# Create your feature branch
git checkout -b feature-chat-history

# Copy your code files to this directory
# Then add and commit
git add .
git commit -m "Implement chat history with conversation memory"

# Push to the specific branch
git push -u origin feature-chat-history
```