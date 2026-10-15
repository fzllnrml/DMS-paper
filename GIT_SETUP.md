# Instructions for the compact repo

1. This folder is a NEW COPY; the original 6.2 GiB research directory is untouched.
2. Check `PROJECT_MAP.md`, `EXPERIMENT_INDEX.md`, and `CHECKPOINT_MAP.csv`.
3. A number of duplicate checkpoints are **relative symlinks** to exactly the
   same checkpoint bytes, to preserve expected old file paths. Use `git clone`
   (not a zip snapshot) to preserve symlinks on machines that support them.
4. This copy omits MNIST dataset files. Existing training/generation code may
   download them, but confirm the expected data path before launching anything.
5. Full experiment binaries and detailed metrics remain in the original project;
   not every historical report can be regenerated solely from this new checkout.
6. Check requirements/versions and test imports locally before full experiments.
7. Only after reviewing contents, create a new **empty private** GitHub repo.
   From THIS NEW folder (not the old folder):

       git init
       git add .
       git status
       git commit -m "Initial reproducible research workspace"
       git branch -M main
       git remote add origin <URL-OF-YOUR-NEW-EMPTY-REPOSITORY>
       git push -u origin main

Do not use `git push --force`. Do not add an existing remote without checking.
The old archive repo stays intact as reference.
