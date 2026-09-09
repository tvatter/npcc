"""Concrete distributional-regression backends for npcc.

Each backend implements :class:`npcc.core.margin.ConditionalMargin` (directly,
for native-evaluation backends) or
:class:`npcc.core.margin_quantile_table.QuantileTableDistribution1D`
(for quantile-based backends).

Backends are constructed through the registry
(:mod:`npcc.core.registry`), never imported eagerly here, so that the
optional-extra backends (NGBoost, TabICL, quantile-GBM) only import their
third-party dependency when actually requested.
"""
