"""Script to perform regression tests.

Usage:
  regression_test <report> <db_path> <params> [--num_cpu=<n>]

Arguments:
  <report>      Name of the report.
  <db_path>     Absolute path to database. 
  <params>      Parameters to query database on.

Options:
  -h --help         Show this screen.
  --version         Show version.
  --num_cpu=<n>     number of cpus to use [default: 2]
"""

#!/usr/bin/env python
from __future__ import print_function

import argparse
from astropy.io import fits
import crds
import datetime
from dask import compute, delayed
from dask.diagnostics import ProgressBar
from docopt import docopt
import jwst
from jwst.pipeline import (Detector1Pipeline, DarkPipeline, 
                           Image2Pipeline,Spec2Pipeline)
from jwst.datamodels import RampModel
import logging
import os
from sqlalchemy import create_engine
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import Session

# Get rid of python 2 stuff in the future.
try:
    from cStringIO import StringIO  # Python 2
except ImportError:
    from io import StringIO

pipelines = {
    'MIR_IMAGE': [Detector1Pipeline, Image2Pipeline],
    'MIR_LRS-SLITLESS': [Detector1Pipeline, Spec2Pipeline],
    'MIR_FLATMRS': [Detector1Pipeline],
    'MIR_MRS': [Detector1Pipeline, Spec2Pipeline],
    'MIR_LYOT': [Detector1Pipeline, Image2Pipeline],
    'MIR_4QPM': [Detector1Pipeline, Image2Pipeline],
    'MIR_CORONCAL': [Detector1Pipeline, Image2Pipeline],
    'MIR_DARK': [DarkPipeline],
    'MIR_FLATIMAGE': [Detector1Pipeline],
    'MIR_LRS-FIXEDSLIT': [Detector1Pipeline, Spec2Pipeline],
    'NRC_IMAGE': [Detector1Pipeline, Image2Pipeline],
    'NRC_FOCUS': [Detector1Pipeline, Image2Pipeline],
    'NRC_LED': [Detector1Pipeline],
    'NRC_DARK': [DarkPipeline],
    'NRC_CORON': [Detector1Pipeline, Image2Pipeline],
    'NRS_DARK':[DarkPipeline],
    'NRS_FIXEDSLIT': [Detector1Pipeline, Spec2Pipeline],
    'NRS_AUTOWAVE': [Detector1Pipeline, Spec2Pipeline],
    'NRS_IFU': [Detector1Pipeline, Spec2Pipeline],
    'NRS_IMAGE': [Detector1Pipeline, Image2Pipeline],
    'NRS_CONFIRM': [Detector1Pipeline, Image2Pipeline],
    'NRS_TACONFIRM': [Detector1Pipeline, Image2Pipeline],
    'NRS_TACQ' : [Detector1Pipeline, Image2Pipeline],
    'NRS_TASLIT': [Detector1Pipeline, Image2Pipeline],
    'NRS_AUTOFLAT': [Detector1Pipeline],
    'NRS_MSASPEC': [Detector1Pipeline, Spec2Pipeline],
    'NRS_WATA': [Detector1Pipeline, Image2Pipeline],
    'NRS_MSATA': [Detector1Pipeline, Image2Pipeline],
    'NRS_FOCUS': [Detector1Pipeline],
    'NRS_LAMP': [Detector1Pipeline],
    'NRS_MSASPEC': [Detector1Pipeline, Spec2Pipeline],
    'NRS_MIMF': [Detector1Pipeline, Image2Pipeline],
    'NRS_BRIGHTOBJ': [Detector1Pipeline, Spec2Pipeline],
    'FGS_FOCUS': [Detector1Pipeline],
    'FGS_INTFLAT': [Detector1Pipeline],
    'FGS_IMAGE': [Detector1Pipeline, Image2Pipeline],
    'NIS_FOCUS': [Detector1Pipeline],
    'NIS_IMAGE': [Detector1Pipeline, Image2Pipeline],
    'NIS_WFSS': [Detector1Pipeline, Spec2Pipeline],
    'NIS_SOSS': [Detector1Pipeline, Spec2Pipeline],
    'NIS_AMI': [Detector1Pipeline, Image2Pipeline],
    'NIS_LAMP': [Detector1Pipeline],
    'NIS_DARK': [DarkPipeline]
}

skip_list = [
    '/grp/jwst/ins/mary/b7.1rc3_sic_dil/jw84600036001_02101_00001_nrs2_uncal.fits',
    '/grp/jwst/ins/mary/b7.1rc3_sic_dil/jw84600039001_02101_00001_nrs2_uncal.fits',
    '/grp/jwst/ins/mary/b7.1rc3_sic_dil/jw80600042001_02101_00001_mirimage_uncal.fits',
    '/grp/jwst/ins/mary/b7.1rc3_sic_dil/jw80600005001_02101_00001_mirimage_uncal.fits',
    '/grp/jwst/ins/mary/b7.1rc3_sic_dil/jw80600052001_02102_00001_mirimage_uncal.fits',
    '/grp/jwst/ins/mary/b7.1rc3_sic_dil/jw80600041001_02101_00001_mirimage_uncal.fits',
    '/grp/jwst/ins/mary/b7.1rc3_sic_dil/jw80600015001_02101_00001_mirimage_uncal.fits',
    '/grp/jwst/ins/mary/b7.1rc3_sic_dil/jw87600025001_02101_00001_nis_uncal.fits'
]


def get_keyword(keyword, header):
    if keyword in header:
        return str(header[keyword])
    else:
        return str(None)


def run_pipeline(fname, report):
    # redirect pipeline log from sys.stderr to a string
    log_stream = StringIO()
    stpipe_log = logging.Logger.manager.loggerDict['stpipe']
    stpipe_log.handlers[0].stream = log_stream

    if fname in skip_list:
        return

    base = fname.split('uncal')[0]

    header = fits.getheader(fname)
    date = get_keyword('DATE-OBS', header)
    time = get_keyword('TIME-OBS', header)
    instrument = get_keyword('INSTRUME', header)
    exp_type = get_keyword('EXP_TYPE', header)
    detector = get_keyword('DETECTOR', header)
    readpatt = get_keyword('READPATT', header)
    nints = get_keyword('NINTS', header)
    ngroups = get_keyword('NGROUPS', header)
    filter = get_keyword('FILTER', header)
    subarray = get_keyword('SUBARRAY', header)
    substrt1 = get_keyword('SUBSTRT1', header)
    subsize1 = get_keyword('SUBSIZE1', header)
    substrt2 = get_keyword('SUBSTRT2', header)
    subsize2 = get_keyword('SUBSIZE2', header)
    pupil = get_keyword('PUPIL', header)
    grating = get_keyword('GRATING', header)

    output = RampModel(os.path.abspath(fname))
    # output = correct_subarray(output, header)

    steps = pipelines[exp_type]

    try:
        for Step in steps:
            output = Step.call(output, save_results=True)

        with open(report, 'a') as f:
            f.write('\t'.join(
                [os.path.abspath(fname), date, time, instrument, exp_type,
                 detector, readpatt, filter, pupil, grating, subarray,
                 # substrt1, subsize1, substrt2, subsize2,
                 str(substrt1), str(subsize1), str(substrt2), str(subsize2),
                 nints, ngroups, '"SUCCESS"', '" "', '\n']))

    except Exception as err:

        # find the last pipeline step mentioned in the log
        for entry in log_stream.getvalue().split(' - '):
            if 'Pipeline.' in entry:
                last_step = entry.split('Pipeline.')[-1]
            else:
                last_step = '?????'

        error = '{} - "{}"'.format(last_step, str(err))

        with open(report, 'a') as f:
            f.write('\t'.join(
                [os.path.abspath(fname), date, time, instrument, exp_type,
                 detector, readpatt, filter, pupil, grating, subarray,
                 # substrt1, subsize1, substrt2, subsize2,
                 str(substrt1), str(subsize1), str(substrt2), str(subsize2),
                 nints, ngroups, '"FAILED"', '"{}"'.format(str(error)), '\n']))

    finally:
        # write the pipeline log to a file
        with open(os.path.basename(fname).replace('fits', 'log'), 'w') as f:
            f.write(log_stream.getvalue())

def build_dask_delayed_list(function, data, args):
    """Build list of dask delayed objects for functions with single arguments.
    May want to expand for more arguments in the future.
    Parameters
    ----------
    function: func
        Function to multiprocess
    data: list-like
        List of data passed to function.
    
    Returns
    -------
    dask_delayed_list: list
        List of dask delayed objects to run in parallel.
    """
    
    dask_delayed_list = []

    for item in data:
        dask_delayed_list.append(delayed(function)(item, args))

    return dask_delayed_list

def main(args):
    import ast

    os.environ['PASS_INVALID_VALUES'] = '1'

    _, crds_context = crds.heavy_client.get_processing_mode("jwst")
    with open(args['<report>'], 'a') as f:
        f.write("# CRDS_CONTEXT = '{}'\n".format(crds_context))
        f.write("# run date = {}\n".format(datetime.datetime.now().isoformat()))
        f.write("# cal version = {}\n".format(jwst.__version__))
        f.write('\t'.join(
            ['filename', 'date', 'time', 'instrument', 'exp_type', 'detector',
             'readpatt', 'filter', 'pupil', 'grating',
             'subarray', 'SUBSTRT1', 'SUBSIZE1', 'SUBSTRT2', 'SUBSIZE2',
             # 'REGSTRT1', 'REGSIZE1', 'REGSTRT2', 'REGSIZE2',
             'nints', 'ngroups', 'status', 'message\n']))

    Base = automap_base()
    # engine, suppose it has two tables 'user' and 'address' set up
    engine = create_engine("sqlite:///{}".format(args['<db_path>']))

    # reflect the tables
    Base.prepare(engine, reflect=True)
    # TestData = Base.classes.test_data
    TestData = Base.classes.regression_data 
    session = Session(engine)

    params = {}
    if args['<params>']:
        params = ast.literal_eval(args['<params>'])
    query = session.query(TestData).filter_by(**params)
    print('found {} matching files'.format(query.count()))
    
    files = [os.path.join(data.path,data.filename) for data in query]
    results = build_dask_delayed_list(run_pipeline, files, args['<report>'])
    
    num_cpu = int(args['--num_cpu'])
    with ProgressBar():
        compute(results, num_workers=num_cpu)

def regression_test():
    args = docopt(__doc__, version='0.1')
    main(args)
