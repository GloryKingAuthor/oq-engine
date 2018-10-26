# -*- coding: utf-8 -*-
# vim: tabstop=4 shiftwidth=4 softtabstop=4
#
# Copyright (C) 2012-2018 GEM Foundation
#
# OpenQuake is free software: you can redistribute it and/or modify it
# under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# OpenQuake is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with OpenQuake. If not, see <http://www.gnu.org/licenses/>.

"""
:mod:`openquake.hazardlib.calc.stochastic` contains
:func:`stochastic_event_set`.
"""
import sys
import time
import numpy
from openquake.baselib.general import AccumDict
from openquake.baselib.performance import Monitor
from openquake.baselib.python3compat import raise_
from openquake.hazardlib.source.rupture import EBRupture
from openquake.hazardlib.contexts import ContextMaker, FarAwayRupture

TWO32 = 2 ** 32  # 4,294,967,296
F64 = numpy.float64
U64 = numpy.uint64
U32 = numpy.uint32
U16 = numpy.uint16
event_dt = numpy.dtype([('eid', U64), ('grp_id', U16), ('ses', U32),
                        ('sample', U32)])


def source_site_noop_filter(srcs):
    for src in srcs:
        yield src, None


source_site_noop_filter.integration_distance = {}


# this is used in acceptance/stochastic_test.py, not in the engine
def stochastic_event_set(sources, source_site_filter=source_site_noop_filter):
    """
    Generates a 'Stochastic Event Set' (that is a collection of earthquake
    ruptures) representing a possible *realization* of the seismicity as
    described by a source model.

    The calculator loops over sources. For each source, it loops over ruptures.
    For each rupture, the number of occurrence is randomly sampled by
    calling
    :meth:`openquake.hazardlib.source.rupture.BaseProbabilisticRupture.sample_number_of_occurrences`

    .. note::
        This calculator is using random numbers. In order to reproduce the
        same results numpy random numbers generator needs to be seeded, see
        http://docs.scipy.org/doc/numpy/reference/generated/numpy.random.seed.html

    :param sources:
        An iterator of seismic sources objects (instances of subclasses
        of :class:`~openquake.hazardlib.source.base.BaseSeismicSource`).
    :param source_site_filter:
        The source filter to use (default noop filter)
    :returns:
        Generator of :class:`~openquake.hazardlib.source.rupture.Rupture`
        objects that are contained in an event set. Some ruptures can be
        missing from it, others can appear one or more times in a row.
    """
    for source, s_sites in source_site_filter(sources):
        try:
            for rupture in source.iter_ruptures():
                [n_occ] = rupture.sample_number_of_occurrences()
                for _ in range(n_occ):
                    yield rupture
        except Exception as err:
            etype, err, tb = sys.exc_info()
            msg = 'An error occurred with source id=%s. Error: %s'
            msg %= (source.source_id, str(err))
            raise_(etype, msg, tb)


# ######################## rupture calculator ############################ #

def sample_ruptures(sources, src_filter=source_site_noop_filter,
                    gsims=(), param=(), monitor=Monitor()):
    """
    :param sources:
        a sequence of sources of the same group
    :param src_filter:
        a source site filter
    :param gsims:
        a list of GSIMs for the current tectonic region model (can be empty)
    :param param:
        a dictionary of additional parameters (by default
        ses_per_logic_tree_path=1 and filter_distance=1000)
    :param monitor:
        monitor instance
    :returns:
        a dictionary with eb_ruptures, num_events, num_ruptures, calc_times
    """
    if not param:
        param = dict(ses_per_logic_tree_path=1, filter_distance=1000)
    eb_ruptures = []
    # AccumDict of arrays with 3 elements weight, nsites, calc_time
    calc_times = AccumDict(accum=numpy.zeros(3, numpy.float32))
    # Compute and save stochastic event sets
    cmaker = ContextMaker(gsims, src_filter.integration_distance,
                          param, monitor)
    num_ses = param['ses_per_logic_tree_path']
    for src, sites in src_filter(sources):
        t0 = time.time()
        # NB: the number of occurrences is very low, << 1, so it is
        # more efficient to filter only the ruptures that occur, i.e.
        # to call sample_ruptures *before* the filtering
        ebrs = build_eb_ruptures(src, num_ses, cmaker, sites)
        n_evs = sum(ebr.multiplicity for ebr in ebrs)
        eb_ruptures.extend(ebrs)
        dt = time.time() - t0
        calc_times[src.id] += numpy.array([n_evs, src.nsites, dt])
    dic = dict(eb_ruptures=eb_ruptures, calc_times=calc_times)
    return dic


def build_eb_ruptures(src, num_ses, cmaker, s_sites, rup_n_occ=(), sam_ses=()):
    """
    :param src: a source object
    :param num_ses: number of stochastic event sets
    :param cmaker: a ContextMaker instance
    :param s_sites: a (filtered) site collection
    :param rup_n_occ: (rup, n_occ) pairs [inferred from the source]
    :param sam_ses: a (sample index, ses index) pair [optional]
    :returns: a list of EBRuptures
    """
    # NB: s_sites can be None if cmaker.maximum_distance is False, then
    # the contexts are not computed and the ruptures not filtered
    ebrs = []
    for rup, n_occ in rup_n_occ or src.sample_ruptures(num_ses, cmaker.ir_mon):
        if cmaker.maximum_distance:
            with cmaker.ctx_mon:
                try:
                    rup.sctx, rup.dctx = cmaker.make_contexts(s_sites, rup)
                    indices = rup.sctx.sids
                except FarAwayRupture:
                    continue
        else:
            indices = ()

        # creating EBRuptures
        events = []
        if sam_ses:  # this happens in UCERF, when n_occ is a scalar
            sam_idx, ses_idx = sam_ses
            for _ in range(n_occ):
                events.append((0, src.src_group_id, ses_idx + 1, sam_idx))
        else:  # regular case, n_occ is a matrix (num_samples, num_ses)
            for (sam_idx, ses_idx), num_occ in numpy.ndenumerate(n_occ):
                for _ in range(num_occ):
                    events.append((0, src.src_group_id, ses_idx + 1, sam_idx))
        E = len(events)
        # setting event IDs based on the rupture serial
        if E:
            assert E < TWO32, len(events)
            evs = numpy.array(events, event_dt)
            ebr = EBRupture(rup, src.id, indices, evs)
            eids = U64(TWO32 * ebr.serial) + numpy.arange(E, dtype=U64)
            ebr.events['eid'] = eids
            ebrs.append(ebr)

    return ebrs
