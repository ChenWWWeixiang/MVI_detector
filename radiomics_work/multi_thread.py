# !/usr/bin/env python

from __future__ import print_function
from radiomics_work.utils_sitk import *
from collections import OrderedDict
import csv,h5py,six
from datetime import datetime
import logging
from multiprocessing import cpu_count, Pool
import os,glob
import shutil
import threading
import numpy as np

import SimpleITK as sitk

import radiomics
from radiomics.featureextractor import RadiomicsFeatureExtractor

threading.current_thread().name = 'Main2'
mask_path = ['/mnt/data1/mvi2/pred_train2','/mnt/data1/mvi2/pred_test2']
OTT=['sa_results_train.csv','sa_results_test.csv']
data_path = '/mnt/data1/mvi2/nrrd_set_new3'
PARAMS = 'small.yaml'# Parameter file
LOG = os.path.join('log2.txt')  # Location of output log file

ROOT=''
# Parallel processing variables
TEMP_DIR = '_TEMP_2'
REMOVE_TEMP_DIR = True  # Remove temporary directory when results have been successfully stored into 1 file
NUM_OF_WORKERS = 16  # Number of processors to use, keep one processor free for other work
if NUM_OF_WORKERS < 1:  # in case only one processor is available, ensure that it is used
    NUM_OF_WORKERS = 1
HEADERS = None  # headers of all extracted features

# Assumes the input CSV has at least 2 columns: "Image" and "Mask"
# These columns indicate the location of the image file and mask file, respectively
# Additionally, this script uses 2 additonal Columns: "Patient" and "Reader"
# These columns indicate the name of the patient (i.e. the image), the reader (i.e. the segmentation), if
# these columns are omitted, a value is automatically generated ("Patient" = "Pt <Pt_index>", "Reader" = "N/A")

# Assumes the following relative paths to this script:
# - Same folder (ROOT): Params.yaml (settings), input.csv (input csv file)
# Creates a log file in the root folder

# Set up logging
################

rLogger = radiomics.logger
logHandler = logging.FileHandler(filename=LOG, mode='a')
logHandler.setLevel(logging.INFO)
logHandler.setFormatter(logging.Formatter('%(levelname)-.1s: (%(threadName)s) %(name)s: %(message)s'))
rLogger.addHandler(logHandler)

# Define filter that allows messages from specified filter and level INFO and up, and level WARNING and up from other
# loggers.
class info_filter(logging.Filter):
    def __init__(self, name):
        super(info_filter, self).__init__(name)
        self.level = logging.WARNING

    def filter(self, record):
        if record.levelno >= self.level:
            return True
        if record.name == self.name and record.levelno >= logging.INFO:
            return True
        return False


# Adding the filter to the first handler of the radiomics logger limits the info messages on the output to just those
# from radiomics.batch, but warnings and errors from the entire library are also printed to the output. This does not
# affect the amount of logging stored in the log file.
outputhandler = rLogger.handlers[0]  # Handler printing to the output
outputhandler.setFormatter(logging.Formatter('[%(asctime)-.19s] (%(threadName)s) %(name)s: %(message)s'))
outputhandler.setLevel(logging.INFO)  # Ensures that INFO messages are being passed to the filter
outputhandler.addFilter(info_filter('radiomics.batch'))

logging.getLogger('radiomics.batch').debug('Logging init')

MOD=['dwi_ADC','t1_PRE','t1_A','t1_V','t1_POST','T2']
def run(case):
    global PARAMS, ROOT, TEMP_DIR
    ptLogger = logging.getLogger('radiomics.batch')
    feature_vector = OrderedDict(case)
    try:


        # set thread name to patient name
        threading.current_thread().name = case['Patient']
        #filename = r'features_' + str(case['Reader']) + '_' + str(case['Patient']) + '.csv'
        #output_filename = os.path.join(ROOT, TEMP_DIR, filename)
        t = datetime.now()
        imageFilepath = case['Image']  # Required
        maskFilepath = case['Mask']  # Required
        # Instantiate Radiomics Feature extractor
        extractor = RadiomicsFeatureExtractor(PARAMS)
        imageFilepath.sort()
        # Extract features
        for i in range(6):
            #break
            row = dict()
            data=sitk.ReadImage(imageFilepath[i])
            maskFilepath.CopyInformation(data)
            result=extractor.execute(data, maskFilepath)
            for j, (key, val) in enumerate(six.iteritems(result)):
                if j < 11:
                    continue

                if not isinstance(val, (float, int, np.ndarray)):
                    continue
                if np.isnan(val):
                    val = 0
                row[MOD[i] + ':' + key]=val
            result = extractor.execute(data, get_margin(maskFilepath))
            # result = extractor.execute(imageName, maskName)
            for j, (key, val) in enumerate(six.iteritems(result)):
                if j < 11:
                    continue
                if not isinstance(val, (float, int, np.ndarray)):
                    continue
                if np.isnan(val):
                    val = 0
                # print(val)
                row[MOD[i] + '-margin:' + key]=val
                #row_next.append(val)
            feature_vector.update(row)
        #maskFilepath = sitk.ReadImage(case['liverseg'])[:,:,:,0]
        # for i in range(6):
        #
        #     row = dict()
        #     data=sitk.ReadImage(imageFilepath[i])
        #     maskFilepath.CopyInformation(data)
        #     result=extractor.execute(data, maskFilepath)
        #     for j, (key, val) in enumerate(six.iteritems(result)):
        #         if j < 11:
        #             continue
        #
        #         if not isinstance(val, (float, int, np.ndarray)):
        #             continue
        #         if np.isnan(val):
        #             val = 0
        #         row[MOD[i] + 'liver:' + key]=val
        #         #row_next.append(val)
        #     feature_vector.update(row)
        # Display message
        delta_t = datetime.now() - t
        ptLogger.info('Patient %s processed in %s', case['Patient'], delta_t)
    except Exception:
        ptLogger.error('Feature extraction failed!', exc_info=True)
    feature_vector.pop('Image')
    feature_vector.pop('Mask')
    feature_vector.pop('liverseg')
    return feature_vector


def _writeResults(featureVector):
    global HEADERS, OUTPUTCSV

    # Use the lock to prevent write access conflicts
    try:
        with open(OUTPUTCSV, 'a') as outputFile:
            writer = csv.writer(outputFile, lineterminator='\n')
            if HEADERS is None:
                HEADERS = list(featureVector.keys())
                writer.writerow(HEADERS)

            row = []
            for h in HEADERS:
                row.append(featureVector.get(h, "N/A"))
            writer.writerow(row)
    except Exception:
        logging.getLogger('radiomics.batch').error('Error writing the results!', exc_info=True)


if __name__ == '__main__':
    logger = logging.getLogger('radiomics.batch')

    # Ensure the entire extraction is handled on 1 thread
    #####################################################

    sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(1)

    # Set up the pool processing
    ############################

    logger.info('pyradiomics version: %s', radiomics.__version__)
    logger.info('Loading Dataset...')
    #extractor = RadiomicsFeatureExtractor('small.yaml')
    for outfile,inpath in zip(OTT,mask_path):
        OUTPUTCSV = os.path.join(outfile)
        files = glob.glob(os.path.join(inpath, '*.nrrd'))
        patients = list(set([f.split('/')[-1].split('_')[1] for f in files]))
        # Extract List of cases
        cases = []
        try:
            for i, name in enumerate(patients):
                row=dict()
                #cr = csv.DictReader(inFile, lineterminator='\n')
                datas = glob.glob(os.path.join(data_path, name + '_*.nrrd'))
                seg = glob.glob(os.path.join(inpath, '*'+ name + '_predictions.nrrd'))
                ismvi = seg[0].split('/')[-1].split('_')[0]
               # datas = f['raw']
                datas.remove(glob.glob(os.path.join(data_path,  name + '_segs_*.nrrd'))[0])
                liverseg = glob.glob(os.path.join('/mnt/data1/mvi2/liverseg', name + '.nrrd'))
                row['Image']=datas
                row['label'] = np.array(int(ismvi))
                row['Patient'] = name
                row['Mask'] =sitk.ReadImage(seg[0])
                row['liverseg']=liverseg
                #f.close()
                cases.append(row)

        except Exception:
            logger.error('CSV READ FAILED', exc_info=True)

        logger.info('Loaded %d jobs', len(cases))

        # Make output directory if necessary
        if not os.path.isdir(os.path.join(ROOT, TEMP_DIR)):
            logger.info('Creating temporary output directory %s', os.path.join(ROOT, TEMP_DIR))
            os.mkdir(os.path.join(ROOT, TEMP_DIR))

        # Start parallel processing
        ###########################

        logger.info('Starting parralel pool with %d workers out of %d CPUs', NUM_OF_WORKERS, cpu_count())
        # Running the Pool
        pool = Pool(NUM_OF_WORKERS)
        results = pool.map(run, cases)

        try:
            # Store all results into 1 file
            with open(OUTPUTCSV, mode='w') as outputFile:
                writer = csv.DictWriter(outputFile,
                                        fieldnames=list(results[0].keys()),
                                        restval='',
                                        extrasaction='raise',
                                        # raise error when a case contains more headers than first case
                                        lineterminator='\n')
                writer.writeheader()
                writer.writerows(results)

            if REMOVE_TEMP_DIR:
                logger.info('Removing temporary directory %s (contains individual case results files)',
                            os.path.join(ROOT, TEMP_DIR))
                shutil.rmtree(os.path.join(ROOT, TEMP_DIR))
        except Exception:
            logger.error('Error storing results into single file!', exc_info=True)

