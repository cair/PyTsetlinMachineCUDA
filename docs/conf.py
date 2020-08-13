import os
import sys
sys.path.insert(0, os.path.abspath('..'))

import sphinx_rtd_theme

needs_sphinx = '1.5'

extensions = ['sphinx.ext.autodoc', 'sphinx.ext.napoleon']

source_suffix = '.rst'

project = 'PyTsetlinMachineCUDA'
copyright = '2020, Ole-Christoffer Granmo'
author = 'Ole-Christoffer Granmo'

pygments_style = 'sphinx'

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True

html_theme = 'sphinx_rtd_theme'
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
