#  -*- coding: utf-8 -*-
#  vim: tabstop=4 shiftwidth=4 softtabstop=4

#  Copyright (c) 2018, GEM Foundation

#  OpenQuake is free software: you can redistribute it and/or modify it
#  under the terms of the GNU Affero General Public License as published
#  by the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.

#  OpenQuake is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Affero General Public License for more details.

#  You should have received a copy of the GNU Affero General Public License
#  along with OpenQuake.  If not, see <http://www.gnu.org/licenses/>.

import operator
import numpy
from openquake.baselib.general import block_splitter
from openquake.hazardlib.geo.utils import get_bounding_box
from openquake.hazardlib.source.base import ParametricSeismicSource
from openquake.hazardlib import mfd

MINWEIGHT = 100


class RuptureCollectionSource(ParametricSeismicSource):
    """
    A parametric source obtained from the splitting of a ComplexFaultSource
    """
    MODIFICATIONS = set()
    RUPTURE_WEIGHT = 4.0  # the same as ComplexFaultSources

    def __init__(self, source_id, name, tectonic_region_type, mfd, ruptures):
        self.source_id = source_id
        self.name = name  # used in disagg
        self.tectonic_region_type = tectonic_region_type
        self.mfd = mfd
        self.ruptures = ruptures
        self.num_ruptures = len(ruptures)

    def count_ruptures(self):
        return self.num_ruptures

    def iter_ruptures(self):
        return iter(self.ruptures)

    def get_bounding_box(self, maxdist):
        """
        Bounding box containing all the hypocenters, enlarged by the
        maximum distance
        """
        locations = [rup.hypocenter for rup in self.ruptures]
        return get_bounding_box(locations, maxdist)


def split(src, chunksize=MINWEIGHT):
    """
    Split a complex fault source in chunks
    """
    for i, block in enumerate(block_splitter(src.iter_ruptures(), chunksize,
                                             kind=operator.attrgetter('mag'))):
        rup = block[0]
        source_id = '%s:%d' % (src.source_id, i)
        amfd = mfd.ArbitraryMFD([rup.mag], [rup.mag_occ_rate])
        yield RuptureCollectionSource(
            source_id, src.name, src.tectonic_region_type, amfd, block)


def sample(srcs, num_ses):
    """
    :param srcs: a list of sources (before splitting)
    :returns: a list of sampled RuptureCollectionSources, possibly empty
    """
    new = []
    for src in srcs:
        numpy.random.seed(src.serial)
        tom = getattr(src, 'temporal_occurrence_model')
        if tom:
            ruptures = list(src.iter_ruptures())
            rates = numpy.array([rup.occurrence_rate for rup in ruptures])
            n_occ = tom.sample_number_of_occurrences(
                rates * num_ses * src.samples)
            rups = []
            for i, rup in enumerate(ruptures):
                if n_occ[i]:
                    rup.n_occ = n_occ[i]
                    rups.append(rup)
            if rups:
                rcs = RuptureCollectionSource(
                    src.source_id, src.name,
                    src.tectonic_region_type,
                    src.mfd, rups)
                rcs.id = src.id
                rcs.serial = src.serial
                rcs.src_group_id = src.src_group_id
                new.append(rcs)
        else:
            new.append(src)
    return new
