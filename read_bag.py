import bagpy
from bagpy import bagreader
import pandas as pd

b=bagreader('<path of the bag file>') # if it is in the same folder, just the name
b.topic_table        # what are the topics we have inside the bag
IMU_data=b.message_by_topic(topic='<topic>')
IMU_csv=pd.read_csv(IMU_data)
