API reference
=============

.. currentmodule:: rainbow_tensor

Shape and indexing
------------------

.. autofunction:: shape
.. autofunction:: index

Shape changing and combining
----------------------------

.. autofunction:: reshape
.. autofunction:: transpose
.. autofunction:: swapaxes
.. autofunction:: sum
.. autofunction:: mean
.. autofunction:: concatenate
.. autofunction:: stack
.. autofunction:: broadcast
.. autofunction:: einsum

Result object
-------------

.. autoclass:: TensorVisual
   :members:

Renderers
---------

.. autoclass:: SvgRenderer
   :members:

.. autofunction:: get_default_renderer
.. autofunction:: set_default_renderer
.. autofunction:: register_renderer
.. autofunction:: resolve_renderer

Themes
------

.. autoclass:: Theme
   :members:

.. autofunction:: get_default_theme
.. autofunction:: set_default_theme
.. autofunction:: register_theme
.. autofunction:: resolve_theme
