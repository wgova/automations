from ecomplexity import ecomplexity,proximity
from pyDataverse.api import NativeApi, DataAccessApi
from pyDataverse.models import Dataverse
import pandas as pd
import os
from zipfile import ZipFile
from urllib.request import urlretrieve
from tempfile import mktemp
import zipfile
from io import BytesIO
import requests

def check_gpu_usage():
  ram_gb = virtual_memory().total / 1e9
  if ram_gb < 20:
    print('Not using a high-RAM runtime')

def dataverse_sitc_file_loader(file_doi,base_url='https://dataverse.harvard.edu/',desired_file_keywords=None):
  api = NativeApi(base_url)
  data_api = DataAccessApi(base_url)
  dataset = api.get_dataset(doi)
  files_list = dataset.json()['data']['latestVersion']['files']
  if desired_file_keywords is not None:
    file = [[f['dataFile']['filename'],f["dataFile"]["id"]] for f in files_list if f['dataFile']['filename'].endswith(desired_file_keywords)]
    return DataAccessApi(base_url),file[0][0] , file[0][1]
  else:
    return {item[0]: item[1:] for item in [[f['dataFile']['filename'],f["dataFile"]["id"]] for f in files_list]}

def dataverse_sitc_file_downloader(target_directory,file_doi,desired_file_keywords=None):
  if desired_file_keywords is not None:
    data_api,filename,file_id = dataverse_sitc_file_loader(file_doi=file_doi,desired_file_keywords=desired_file_keywords)
    response = data_api.get_datafile(file_id)
    completeName = os.path.join(target_directory, filename)
    if not os.path.exists(completeName):
      print(f'Downloading data from https://dataverse.harvard.edu/{file_doi}')
      with open(completeName, "wb") as f:
        f.write(response.content)
    else:
      print(f'Reading file from disk located in {target_directory}')
      completeName = [os.path.join(target_directory, f) for f in os.listdir(target_directory) if f.endswith('.dta')] 
    return completeName
  else:
    print(f'\nToo many large files to write to disk. \nScan list of files in the output to select desired_file_keywords \nThen re-run with these as parameters')
    #TODO: dictionary comprehension to store all files
    return list(dataverse_sitc_file_loader(file_doi).keys())

def update_name(self):
  return self.assign(name = self.location_code + '_' + self.sitc_product_code)

def load_dta(path):
  cols = ['year', 'location_code', 'sitc_product_code', 'export_value']
  return pd.read_stata(path)[cols]

def calc_sitc_econ_complexity(self):
  trade_cols = {'time':'year', 'loc':'location_code', 'prod':'sitc_product_code', 'val':'export_value'}
  return pd.concat([ecomplexity(self[self.year==t], trade_cols) for t in self.year.unique()])

def calculate_rca_metrics(save_path,rca_name='sitc_rca_2020.csv',sitc_dta_path=None):
  doi = 'doi:10.7910/DVN/H8SFD2'
  file_keywords = '4digit_year.dta'
  rca_path = os.path.join(save_path,rca_name)
  if os.path.exists(rca_path):
    print('Complexity metrics already calculated. \nLoading from disk')
    sitc_rca = pd.read_csv(rca_path,low_memory=False)\
    .dropna(subset=['rca'])
  else:
    print('Calculating complexity metrics')
    with tf.device('/device:GPU:0'):
      sitc_dta_path = dataverse_sitc_file_downloader(file_doi=doi,target_directory=save_path,desired_file_keywords=file_keywords)[0]
      sitc_rca = load_dta(sitc_dta_path)\
      .pipe(calc_sitc_econ_complexity)\
      .dropna(subset=['rca'])
      sitc_rca.to_csv(rca_path)
  return sitc_rca

def exclude_service_products(df):
  return df[df['sitc_product_code'].str.contains('^[0-9]')]