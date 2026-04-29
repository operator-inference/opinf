# This in the configuration file needed to use Sphinx instead of Jupyter Notebook

# -- Project Information -----------------------------------------------------

project = "Operator Inference"      # Added this title here
author = "Willcox Research Group, Oden Institute for Computational Engineering and Sciences"

# -- General Configuration ---------------------------------------------------

extensions = [
    # MyST and notebook support (replaces JupyterBook's built-in handling)
    "myst_nb",                      # handles our .md files

    # Bibliography (was bundled in JupyterBook, now explicit)
    "sphinxcontrib.bibtex",

    # API documentation (same as before)
    "numpydoc",                     # Parses the Numpy-style docstrings
    "sphinx.ext.autodoc",           # Pulls the docstrings from Python Source code to create API docs
    "sphinx.ext.autosummary",       # Generates summary tables and individual pages for modules, classes, and functions
    "sphinx.ext.viewcode",          # Adds the "view source" to the docs
    "sphinx.ext.napoleon",          # Understand numpy and google style docstrings

    # Cross-project links
    "sphinx.ext.intersphinx",       # Allows cross-linking to other packages docs

    # UI components (admonitions, grids, tabs, etc.)
    "sphinx_design",                # Adds grids, carbs, tabs, and badges to your docs

    # Diagrams
    "sphinxcontrib.mermaid",        # Allows for the embedding of mermaid diagrams
]

# Templates for autosummary (same path as before)
templates_path = ["templates"]

# Static files (CSS, images, etc.)
html_static_path = ["_static"]

# Suppress specific warnings (carried over from _config.yml)
suppress_warnings = ["etoc.toctree"]

# -- MyST Configuration ------------------------------------------------------
# These replace the parse.myst_enable_extensions block in _config.yml

myst_enable_extensions = [
    "amsmath",          # Lets you use \begin{equation} and others
    "colon_fence",      # ::: instead of ```
    "dollarmath",       # $...$ and $$..$$ for inline block math
    "linkify",          # bare URLs become links
    "substitution",     # Lets you define reusable text snippets
    "tasklist",         # Github Style check boxes
]

# -- Notebook Execution (myst_nb) --------------------------------------------
# Replaces execute.execute_notebooks and execute.timeout in _config.yml

nb_execution_mode = "cache"         # auto, force, cache, or "off"
nb_execution_timeout = 120          # seconds before KeyboardInterrupt

# -- Bibliography ------------------------------------------------------------
# Replaces bibtex_bibfiles in _config.yml

bibtex_bibfiles = [
    "references.bib",
    "literature.bib",
]
bibtex_reference_style = "label"

# -- Autosummary / Autodoc ---------------------------------------------------
# Carried over from sphinx.config in _config.yml

add_function_parentheses = True
add_module_names = False            # Shorten function names in API docs

autosummary_generate = True
autosummary_filename_map = {        # Resolves lower/upper case ambiguities
    "opinf.post.Lp_error": "bigLp-error",
}

# -- Numpydoc ----------------------------------------------------------------

numpydoc_class_members_toctree = False
numpydoc_show_class_members = False
numpydoc_show_inherited_class_members = False

# -- Intersphinx -------------------------------------------------------------
# Cross-links to external package documentation

intersphinx_mapping = {
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "numpy":      ("https://numpy.org/doc/stable/", None),
    "python":     ("https://docs.python.org/3/", None),
    "scipy":      ("https://docs.scipy.org/doc/scipy/", None),
    "sklearn":    ("https://scikit-learn.org/stable/", None),
    "pandas":     ("https://pandas.pydata.org/docs/", None),
}

# -- MathJax -----------------------------------------------------------------
# Replaces mathjax_path and mathjax3_config in _config.yml
# All your custom LaTeX macros are carried over exactly

mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"

mathjax3_config = {
    "tex": {
        "macros": {
            "RR": "\\mathbb{R}",
            "NN": "\\mathbb{N}",
            "ZZ": "\\mathbb{Z}",
            "I":  "\\mathbf{I}",
            "0":  "\\mathbf{0}",
            "1":  "\\mathbf{1}",
            "q":  "\\mathbf{q}",
            "u":  "\\mathbf{u}",
            "z":  "\\mathbf{z}",
            "d":  "\\mathbf{d}",
            "f":  "\\mathbf{f}",
            "s":  "\\mathbf{s}",
            "Q":  "\\mathbf{Q}",
            "U":  "\\mathbf{U}",
            "Z":  "\\mathbf{Z}",
            "Op":    "\\mathbf{f}",
            "Ophat": "\\hat{\\mathbf{f}}",
            "c":  "\\mathbf{c}",
            "A":  "\\mathbf{A}",
            "H":  "\\mathbf{H}",
            "G":  "\\mathbf{G}",
            "B":  "\\mathbf{B}",
            "N":  "\\mathbf{N}",
            "v":  "\\mathbf{v}",
            "w":  "\\mathbf{w}",
            "V":  "\\mathbf{V}",
            "W":  "\\mathbf{W}",
            "Vr": "\\mathbf{V}_{\\!r}",
            "Wr": "\\mathbf{W}_{\\!r}",
            "qhat": "\\hat{\\mathbf{q}}",
            "zhat": "\\hat{\\mathbf{z}}",
            "fhat": "\\hat{\\mathbf{f}}",
            "Qhat": "\\hat{\\mathbf{Q}}",
            "Zhat": "\\hat{\\mathbf{Z}}",
            "chat": "\\hat{\\mathbf{c}}",
            "Ahat": "\\hat{\\mathbf{A}}",
            "Hhat": "\\hat{\\mathbf{H}}",
            "Ghat": "\\hat{\\mathbf{G}}",
            "Bhat": "\\hat{\\mathbf{B}}",
            "Nhat": "\\hat{\\mathbf{N}}",
            "D":    "\\mathbf{D}",
            "ohat": "\\hat{\\mathbf{o}}",
            "Ohat": "\\hat{\\mathbf{O}}",
            "bfmu":     "\\boldsymbol{\\mu}",
            "bfGamma":  "\\boldsymbol{\\Gamma}",
            "bfPhi":    "\\boldsymbol{\\Phi}",
            "bfSigma":  "\\boldsymbol{\\Sigma}",
            "bfPsi":    "\\boldsymbol{\\Psi}",
            "bfLambda": "\\boldsymbol{\\Lambda}",
            "bfxi":     "\\boldsymbol{\\xi}",
            "trp":   "{^{\\mathsf{T}}}",
            "ddt":   "\\frac{\\textrm{d}}{\\textrm{d}t}",
            "ddqhat": "\\frac{\\partial}{\\partial\\qhat}",
            "mean":   "\\operatorname{mean}",
            "std":    "\\operatorname{std}",
            "argmin": "\\operatorname{argmin}",
        }
    }
}

# -- Figure numbering --------------------------------------------------------
# Eq (1.1) instead of (1) — carried over from numfig_secnum_depth

numfig = True                       # Must enable numfig for numfig_secnum_depth to work
numfig_secnum_depth = 1             # JupyterBook set this but didn't require numfig=True explicitly

# -- HTML Output -------------------------------------------------------------
# Replaces the html and repository blocks in _config.yml

html_theme = "sphinx_book_theme"
html_logo = "_static/logo.svg"
html_favicon = "_static/favicon.svg"

html_css_files = ["properties.css"] # Your custom CSS from _static/

html_theme_options = {
    "repository_url": "https://github.com/Willcox-Research-Group/rom-operator-inference-Python3",
    "repository_branch": "main",
    "path_to_book": "docs",
    "use_edit_page_button": False,
    "use_issues_button": True,
    "use_repository_button": True,
    "home_page_in_navbar": False,
}

html_context = {
    # Required by themes that support edit/issue buttons
    "github_user": "Willcox-Research-Group",
    "github_repo": "rom-operator-inference-Python3",
    "github_version": "main",
    "doc_path": "docs",
}