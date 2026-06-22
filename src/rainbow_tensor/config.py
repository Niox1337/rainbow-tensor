"""Single home for all mutable global configuration.

The theme, renderer, and explanation modules keep their own logic, but every
default a user can change lives here, so there is one place to read or reset the
whole configuration. The object defaults start as ``None`` sentinels meaning
"use the built in default", so this module imports nothing from the package and
can never cause an import cycle.
"""

# None -> theme.LIGHT
default_theme = None
# None -> each theme's own axis colour ramp
default_axis_colors = None
# None -> renderers.SVG
default_renderer = None
# explanation language; "en" is always the fallback
language = "en"
