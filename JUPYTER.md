### JupyterLab sidebar hiding – breadcrumbs

- Implemented project-local Lab settings (no ~/.jupyter)
  - Set env: `JUPYTER_CONFIG_DIR=config/jupyter`, `JUPYTERLAB_SETTINGS_DIR=config/jupyter/lab/user-settings`, `JUPYTERLAB_WORKSPACES_DIR=config/jupyter/lab/workspaces`.
  - Wrote `config/jupyter/lab/user-settings/@jupyterlab/application-extension/application.jupyterlab-settings` → `{ "mode": "single-document" }`.
  - Result: sidebar still visible.

- Embedded Lab UI instead of classic
  - Iframe → `/jupyter/lab/tree/<nb>`.
  - Result: sidebar still visible.

- Isolated + reset workspace
  - Iframe → `/jupyter/lab/workspaces/nbscribe-embed/tree/<nb>?reset=1`.
  - Result: sidebar still visible; `reset=1` had no effect and was removed.

- Reduced access-log noise (kept important events)
  - Custom filter suppresses repetitive GETs and 204 workspaces PUTs.

Hypothesis
- JupyterLab 4 controls single-document and sidebar via workspace layout; the user-setting key above is ignored. Restored/empty workspaces still default to showing the left sidebar.

Next steps (preferred → fallback)
- Seed workspace JSON at `config/jupyter/lab/workspaces/nbscribe-embed.jupyterlab-workspace` with: mode `single-document`, left area closed, no filebrowser widget. (Implemented.)
- (Fallback) App-local `config/jupyter/labconfig/page_config.json` with `disabledExtensions: ["@jupyterlab/filebrowser-extension"]` to hard-disable the file browser.

