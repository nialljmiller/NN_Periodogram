import nn_periodogram

project = "NN_Periodogram"
author = "Niall J. Miller"
copyright = "2022-2026, Niall J. Miller"
release = nn_periodogram.__version__
extensions = ["myst_parser", "sphinx.ext.autodoc", "sphinx.ext.napoleon", "sphinx.ext.viewcode"]
autodoc_mock_imports = []
autodoc_member_order = "bysource"
napoleon_numpy_docstring = True
myst_enable_extensions = ["colon_fence", "dollarmath"]
html_theme = "furo"
html_title = "NN_Periodogram"
exclude_patterns = ["_build"]
