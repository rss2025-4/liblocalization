## run:

```bash

python3 -m venv path/of/your/choice

# code depending on liblocalization can only run inside the venv
source path/of/your/choice/bin/activate

# unisntall existing if updating
pip uninstall liblocalization -y

# latest
pip install --upgrade git+https://github.com/rss2025-4/liblocalization.git
# or from branch
pip install --upgrade git+https://github.com/rss2025-4/liblocalization.git@branch-xxxx
# or from revision
pip install --upgrade git+https://github.com/rss2025-4/liblocalization.git@xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

```
