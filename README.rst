=======================================
Polynomial fusion layer
=======================================

Official implementation of the Polynomial fusion layer appearing in `"Speech-Driven Facial Animation Using Polynomial Fusion of Features" <https://ieeexplore.ieee.org/document/9054469>`_ (ICASSP'20).


Installation
=======================================

Prerequisites:

* NumPy https://numpy.org/
* PyTorch https://pytorch.org/
* TensorLy https://github.com/tensorly/tensorly

To install from source, run the following:

.. code-block::

  pip install -e .

Example usage
=======================================

.. code-block:: python

    import torch

    from polynomial_fusion import PolynomialFusion

    device = "cuda:0"

    # Toy example with 3 views (features/data sources)

    # Get inputs
    batch_size = 64
    d1 = 20
    d2 = 21
    d3 = 22

    X1 = torch.randn([batch_size, d1])
    X2 = torch.randn([batch_size, d2])
    X3 = torch.randn([batch_size, d3])

    # CP decomposition, multi-view
    f = PolynomialFusion(model_type = "CP",
                         input_dims = [d1, d2, d3],
                         output_dim = 32,
                         rank = 10,
                         concat_last_mode = False,
                         multiview = True)
    f.to(device)

    # Compute joint embedding
    Z = f([X1.to(device), X2.to(device), X3.to(device)]) # shape [batch_size, 32]

Citing
=======
If you use this code, please cite [1]_:

.. code-block:: bibtex

  @inproceedings{sdapf2020,
  author={Kefalas, Triantafyllos and Vougioukas, Konstantinos and Panagakis, Yannis and Petridis, Stavros and Kossaifi, Jean and Pantic, Maja},
  booktitle={ICASSP 2020 - 2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={Speech-Driven Facial Animation Using Polynomial Fusion of Features}, 
  year={2020},
  pages={3487-3491},
  doi={10.1109/ICASSP40776.2020.9054469}}


References
==========

.. [1] Triantafyllos Kefalas, Konstantinos Vougioukas, Yannis Panagakis, Stavros Petridis, Jean Kossaifi and Maja Pantic, **Speech-driven facial animation using polynomial fusion of features**, *International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, 2020.
