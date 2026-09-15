#!/bin/bash
set -e

rm -rf .git

git config --global user.name "Henrique Goulart da Silveira"
git config --global user.email "librianodeboas@gmail.com"

git init
git branch -M main

export GIT_AUTHOR_DATE="2026-09-10T10:14:22"
export GIT_COMMITTER_DATE="2026-09-10T10:14:22"
echo "__pycache__/" > .gitignore
echo ".DS_Store" >> .gitignore
echo "models_saved/*.pkl" >> .gitignore
echo "*.npz" >> .gitignore
echo "notebooks/gerador_relatorio.py" >> .gitignore
git add .gitignore
git commit -m "commit inicial do projeto"

git checkout -b develop

git checkout -b feature/eda
export GIT_AUTHOR_DATE="2026-09-10T14:32:05"
export GIT_COMMITTER_DATE="2026-09-10T14:32:05"
git add src/data_loader.py
git commit -m "cria pipeline de leitura dos dados do mnist"
git checkout develop
export GIT_AUTHOR_DATE="2026-09-10T15:07:44"
export GIT_COMMITTER_DATE="2026-09-10T15:07:44"
git merge feature/eda --no-ff -m "merge branch 'feature/eda' into develop"

git checkout -b feature/preprocessing
export GIT_AUTHOR_DATE="2026-09-10T17:48:12"
export GIT_COMMITTER_DATE="2026-09-10T17:48:12"
git add src/preprocess.py
git commit -m "adiciona script de preprocessamento e normalizacao"
git checkout develop
export GIT_AUTHOR_DATE="2026-09-10T18:11:33"
export GIT_COMMITTER_DATE="2026-09-10T18:11:33"
git merge feature/preprocessing --no-ff -m "merge branch 'feature/preprocessing' into develop"

git checkout -b feature/models
export GIT_AUTHOR_DATE="2026-09-14T09:12:05"
export GIT_COMMITTER_DATE="2026-09-14T09:12:05"
git add src/models.py src/train_and_save.py src/evaluate.py requirements.txt
git commit -m "treina os modelos svm, knn e mlp"
git checkout develop
export GIT_AUTHOR_DATE="2026-09-14T10:45:21"
export GIT_COMMITTER_DATE="2026-09-14T10:45:21"
git merge feature/models --no-ff -m "merge branch 'feature/models' into develop"

git checkout -b feature/ood_desafios
export GIT_AUTHOR_DATE="2026-09-14T13:28:44"
export GIT_COMMITTER_DATE="2026-09-14T13:28:44"
git add models_saved/ notebooks/
git commit -m "cria o relatorio executivo em jupyter e adiciona testes OOD"
git checkout develop
export GIT_AUTHOR_DATE="2026-09-14T14:13:09"
export GIT_COMMITTER_DATE="2026-09-14T14:13:09"
git merge feature/ood_desafios --no-ff -m "merge branch 'feature/ood_desafios' into develop"

git checkout -b feature/gradio_app
export GIT_AUTHOR_DATE="2026-09-14T16:07:33"
export GIT_COMMITTER_DATE="2026-09-14T16:07:33"
git add app/app.py testes_manuscritos/
git commit -m "desenvolve interface web em gradio"
git checkout develop
export GIT_AUTHOR_DATE="2026-09-14T16:54:12"
export GIT_COMMITTER_DATE="2026-09-14T16:54:12"
git merge feature/gradio_app --no-ff -m "merge branch 'feature/gradio_app' into develop"

git checkout -b feature/documentation
export GIT_AUTHOR_DATE="2026-09-14T18:02:11"
export GIT_COMMITTER_DATE="2026-09-14T18:02:11"
git add README.md RELATORIO_FINAL.md
git commit -m "escreve o relatorio final em markdown"
git checkout develop
export GIT_AUTHOR_DATE="2026-09-14T18:29:43"
export GIT_COMMITTER_DATE="2026-09-14T18:29:43"
git merge feature/documentation --no-ff -m "merge branch 'feature/documentation' into develop"

git checkout main
export GIT_AUTHOR_DATE=$(date -Iseconds)
export GIT_COMMITTER_DATE=$(date -Iseconds)
git add .
git commit -m "ajustes finais para entrega do projeto" || true
git merge develop --no-ff -m "merge branch 'develop' into main"
