"""
This is a skeleton file that can serve as a starting point for a Python
console script. To run this script uncomment the following lines in the
``[options.entry_points]`` section in ``setup.cfg``::

    console_scripts =
         fibonacci = biocpy_diffexp.skeleton:run

Then run ``pip install .`` (or ``pip install -e .`` for editable mode)
which will install the command ``fibonacci`` inside your current environment.

Besides console scripts, the header (i.e. until ``_logger``...) of this file can
also be used as template for Python modules.

Note:
    This file can be renamed depending on your needs or safely removed if not needed.

References:
    - https://setuptools.pypa.io/en/latest/userguide/entry_point.html
    - https://pip.pypa.io/en/stable/reference/pip_install
"""

import argparse
import logging
import sys
from functools import singledispatch
import numpy as np
import anndata
import summarizedexperiment
import pandas as pd
from formulaic_contrasts import FormulaicContrasts
from copy import copy, deepcopy

from biocpy_diffexp import __version__

__author__ = "Deijkstra"
__copyright__ = "Deijkstra"
__license__ = "MIT"

_logger = logging.getLogger(__name__)


# ---- Python API ----
# The functions defined in this section can be imported by users in their
# Python scripts/interactive interpreter, e.g. via
# `from biocpy_diffexp.skeleton import fib`,
# when using this Python module as a library.

from scipy.sparse import csr_matrix
from anndata2ri import scipy2ri

from functools import singledispatch
from scipy.sparse import csc_matrix

import rpy2.rinterface_lib.callbacks
from rpy2.robjects.packages import STAP
# import anndata2ri
import logging

from .conversion import _py_to_r, _r_to_py, _ad_to_rmat, _ad_to_dge


# rbase = importr("base")

r_string = '''
.get_class <- function(object) class(object)
.get_colnames <- function(mat) colnames(mat)
.get_rownames <- function(mat) rownames(mat)
.get_rownames_dge <- function(dge) rownames(dge$counts)
'''
r_pkg = STAP(r_string, "r_pkg")

# edger = importr("edgeR")




@singledispatch
def filter_by_expr(mat: np.ndarray, **kwargs):
    assert isinstance(mat, np.ndarray)
    r_string = '''
    .filter <- function(mat, ...) {
        keep <- edgeR::filterByExpr(mat, ...)
        return(keep)
    }
    '''
    r_pkg = STAP(r_string, "r_pkg")
    rmat = _py_to_r(mat)
    kwargs_keys = list(kwargs.keys())
    if len(kwargs_keys) > 0:
        for key in kwargs_keys:
            kwargs[key] = _py_to_r(kwargs[key])
    
    keep = r_pkg._filter(rmat, **kwargs)
    
    keep = _r_to_py(keep)
    keep = keep.astype(bool)
    return keep
@filter_by_expr.register
def _(adata: anndata.AnnData, layer = "X", **kwargs):
    mat = adata.X if layer =="X" else adata.layers[layer]
    return filter_by_expr(mat.T, **kwargs)

@singledispatch
def calc_norm_factors(mat: np.ndarray, **kwargs):
    assert isinstance(mat, np.ndarray)
    r_string = '''
    .calc_norm_factors <- function(mat, ...) {
        norm_factors <- edgeR::calcNormFactors(mat, ...)
        return(norm_factors)
    }
    '''
    r_pkg = STAP(r_string, "r_pkg")
    rmat = _py_to_r(mat)
    kwargs_keys = list(kwargs.keys())
    if len(kwargs_keys) > 0:
        for key in kwargs_keys:
            kwargs[key] = _py_to_r(kwargs[key])
    
    norm_factors = r_pkg._calc_norm_factors(rmat, **kwargs)
    
    norm_factors = _r_to_py(norm_factors)
    return norm_factors
@calc_norm_factors.register
def _(adata: anndata.AnnData, layer = "X", **kwargs):
    mat = adata.X if layer =="X" else adata.layers[layer]
    norm_factors = calc_norm_factors(mat.T, **kwargs)
    adata.obs["norm.factors"] = norm_factors


@calc_norm_factors.register
def _(se: summarizedexperiment.SummarizedExperiment, assay = "counts", in_place = False, **kwargs):
    # out = se._define_output(in_place = in_place)
    out = deepcopy(se)
    
    mat = summed.get_assays()[assay]
    norm_factors = calc_norm_factors(mat, **kwargs)
    out.set_column_data = copy(out.get_column_data())
    out.column_data.set_column("norm.factors", norm_factors, in_place = True)
    return out


# @wrap_non_picklable_objects
class EdgerFit:
    def __init__(self, r_fit, formulaic_contrast):
        self.r_fit = r_fit
        self.formulaic_contrast = formulaic_contrast
# @delayed
# @wrap_non_picklable_objects
def fit_edger(
    adata: anndata.AnnData, 
    design: str | np.ndarray | pd.DataFrame, 
    **kwargs
):

    if isinstance(design, str):
        formulaic_contrast = FormulaicContrasts(adata.obs, design)
        design_matrix = formulaic_contrast.design_matrix.values
    elif isinstance(design, np.ndarray):
        design_matrix = design
    elif isinstance(design, pd.DataFrame):
        design_matrix = design.values
    else:
        print("something is wrong with the design")

    dge = _ad_to_dge(adata)
    r_design = _py_to_r(design_matrix)

    kwargs_keys = list(kwargs.keys())
    if len(kwargs_keys) > 0:
        for key in kwargs_keys:
            kwargs[key] = _py_to_r(kwargs[key])
        
    r_string = '''
    .fit_edger <- function(y, design, ...) {
        fit <- glmQLFit(y, design)
        return(fit)
    }
    '''
    r_pkg = STAP(r_string, "r_pkg")
    
    edger_fit = r_pkg._fit_edger(dge, r_design)
    return EdgerFit(edger_fit, formulaic_contrast)

from typing import Sequence
# from typing_inspect import get_origin

# @delayed
# @wrap_non_picklable_objects
def test_edger(fit: EdgerFit, contrast: Sequence[float] | Sequence[str] | dict, **kwargs):
    kwargs_keys = list(kwargs.keys())
    if len(kwargs_keys) > 0:
        for key in kwargs_keys:
            kwargs[key] = _py_to_r(kwargs[key])

    if isinstance(contrast, dict):
        key_names = list(contrast.keys())
        assert "column" in key_names
        assert "baseline" in key_names
        assert "group_to_compare" in key_names
        _contrast = np.asarray(fit.formulaic_contrast.contrast(**contrast))
    elif (isinstance(contrast, Sequence)) & (all(isinstance(n, int) for n in contrast)):
        _contrast = np.asarray(contrast)
    elif (isinstance(contrast, Sequence)) & (all(isinstance(n, str) for n in contrast)):
        _contrast = np.asarray(fit.formulaic_contrast.contrast(*contrast))
        
    
    r_string = '''
    .test_edger <- function(fit, contrast, ...) {
        test <- glmQLFTest(fit, contrast = contrast)
        res <- topTags(test, n = Inf)
        return(as.data.frame(res))
    }
    '''
    r_pkg = STAP(r_string, "r_pkg")

    r_contrast = _py_to_r(_contrast)
    res = r_pkg._test_edger(fit.r_fit, r_contrast)
    res = _r_to_py(res)
    return res

def _return_null_df(adata):
    null_df = pd.DataFrame(index = adata.var_names)
    null_df["logFC"] = np.nan
    null_df["logCPM"] = np.nan
    null_df["F"] = np.nan
    null_df["PValue"] = np.nan
    null_df["FDR"] = np.nan
    null_df = null_df.reset_index(names = ["variable"])
    return null_df
# @delayed
# @wrap_non_picklable_objects
def test_contrast(adata, design, contrast):
    
    try: 
        
        fit = fit_edger(adata, design)

        res = test_edger(fit, contrast)
        return res
    except:
        print("testing edger failed")
        return _return_null_df(adata)






# ---- CLI ----
# The functions defined in this section are wrappers around the main Python
# API allowing them to be called directly from the terminal as a CLI
# executable/script.