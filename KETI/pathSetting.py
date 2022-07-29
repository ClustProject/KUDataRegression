import sys

sys.path.append("../../")
sys.path.append("../../..")

from KETIPreDataIngestion.KETI_setting import influx_setting_KETI as ins
from KETIPreDataIngestion.data_influx import influx_Client_v2 as influx_Client
db_client = influx_Client.influxClient(ins.CLUSTDataServer2)

DataMetaPath = "./integratedData.json"
csvDataFileRootDir ='./data/'
scalerRootDir ='./scaler/'
trainModelMetaFilePath ="./model.json"