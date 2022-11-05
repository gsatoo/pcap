######################################
# NSL-KDD Anomaly detection
# https://www.kaggle.com/code/avk256/nsl-kdd-anomaly-detection/notebook

import numpy as np 
import pandas as pd 
import argparse

class NSL_KDD():
    def __init__(self,
                 fnameTrain,
                 fnameTest,
                 numOfLabels=2,
                 inclFlgFG2 = True,
                 inclFlgFG3 = True,
                 inclFlgFG4 = True,
                 inclFlgFG5 = True,
                 inclFlgFG6 = True):
        self.numOfLabels = numOfLabels
        self.fNameTrain = fnameTrain
        self.fNameTest = fnameTest
        self.dfTrain = pd.read_csv(fnameTrain)
        self.dfTest = pd.read_csv(fnameTest)
        # set the column labels
        self.columns = (['duration'
                         ,'protocol_type'
                         ,'service'
                         ,'flag'
                         ,'src_bytes'
                         ,'dst_bytes'
                         ,'land'
                         ,'wrong_fragment'
                         ,'urgent'
                         ,'hot'
                         ,'num_failed_logins'
                         ,'logged_in'
                         ,'num_compromised'
                         ,'root_shell'
                         ,'su_attempted'
                         ,'num_root'
                         ,'num_file_creations'
                         ,'num_shells'
                         ,'num_access_files'
                         ,'num_outbound_cmds'
                         ,'is_host_login'
                         ,'is_guest_login'
                         ,'count'
                         ,'srv_count'
                         ,'serror_rate'
                         ,'srv_serror_rate'
                         ,'rerror_rate'
                         ,'srv_rerror_rate'
                         ,'same_srv_rate'
                         ,'diff_srv_rate'
                         ,'srv_diff_host_rate'
                         ,'dst_host_count'
                         ,'dst_host_srv_count'
                         ,'dst_host_same_srv_rate'
                         ,'dst_host_diff_srv_rate'
                         ,'dst_host_same_src_port_rate'
                         ,'dst_host_srv_diff_host_rate'
                         ,'dst_host_serror_rate'
                         ,'dst_host_srv_serror_rate'
                         ,'dst_host_rerror_rate'
                         ,'dst_host_srv_rerror_rate'
                         ,'attack_name'
                         ,'level'])

        self.all_features = ['duration',
                             'protocol_type', 'service', 'flag',
                             'src_bytes', 'dst_bytes',
                             'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised',
                             'root_shell', 'su_attempted',
                             'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds', 
                             'is_host_login', 'is_guest_login', 
                             'count', 'srv_count', 
                             'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 
                             'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
                             'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 
                             'attack_name', 'level']

        # 
        # Basic features of individual TCP connections in NSL-KDD dataset.
        # # feature name | description | type 
        # duration | length (number of seconds) of the connection | continuous 
        # protocol_type | type of the protocol, e.g. tcp, udp, etc | discrete 
        # service | network service on the destination, e.g., http, telnet, etc. | discrete 
        # src_bytes | number of data bytes from source to destination | continuous 
        # dst_bytes | number of data bytes from destination to source | continuous 
        # flag | normal or error status of the connection | discrete 
        # land | 1 if connection is from/to the same host/port | discrete 
        # wrong_fragment | 0 otherwise; number of ``wrong'' fragments | continuous 
        # urgent | number of urgent packets | continuous
        self.categorical_features = ['protocol_type', 'service', 'flag']
        self.numeric_features_grp1 = ['duration', 'src_bytes', 'dst_bytes',
                                      'land', 'wrong_fragment', 'urgent']
        
        # Content features within a connection suggested by domain knowledge in NSL-KDD dataset.
        # # feature name | description | type
        # hot | number of ``hot'' indicators | continuous
        # num_failed_logins | number of failed login attempts | continuous
        # logged_in | 1 if successfully logged in; 0 otherwise | discrete
        # num_compromised | number of ``compromised'' conditions | continuous
        # root_shell | 1 if root shell is obtained; 0 otherwise | discrete
        # su_attempted | 1 if ``su root'' command attempted; 0 otherwise | discrete
        # num_root | number of ``root'' accesses | continuous
        # num_file_creations | number of file creation operations | continuous
        # num_shells | number of shell prompts | continuous
        # num_access_files | number of operations on access control files | continuous
        # num_outbound_cmds | number of outbound commands in an ftp session | continuous
        # is_hot_login | 1 if the login belongs to the ``hot'' list; 0 otherwise | discrete
        # is_guest_login | 1 if the login is a ``guest''login; 0 otherwise | discrete
        self.numeric_features_grp2 = ['hot', 'num_failed_logins', 'logged_in', 'num_compromised',
                                      'root_shell', 'su_attempted',
                                      'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds', 
                                      'is_host_login', 'is_guest_login']
        # Traffic features computed using a two-second time window in NSL-KDD dataset.
        # # feature name | description | type
        # count | number of connections to the same host as the current connection in the past two seconds | continuous
        self.numeric_features_grp3 = ['count']
        # Note: The following  features refer to these same-host connections.
        # serror_rate | % of connections that have ``SYN'' errors | continuous
        # rerror_rate | % of connections that have ``REJ'' errors | continuous
        # same_srv_rate | % of connections to the same service | continuous
        # diff_srv_rate | % of connections to different services | continuous
        # srv_count | number of connections to the same service as the current connection in the past two seconds | continuous
        self.numeric_features_grp4 = ['serror_rate', 'rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_count']
        # Note: The following features refer to these same-service connections.
        # srv_serror_rate | % of connections that have ``SYN'' errors | continuous
        # srv_rerror_rate | % of connections that have ``REJ'' errors | continuous
        # srv_diff_host_rate | % of connections to different hosts | continuous
        self.numeric_features_grp5 = ['srv_serror_rate', 'srv_rerror_rate', 'srv_diff_host_rate']
        # 
        self.numeric_features_grp6 = ['dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
                                      'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate']
        self.numeric_features = self.numeric_features_grp1
        if (inclFlgFG2 == True) :
            self.numeric_features += self.numeric_features_grp2
        if (inclFlgFG3 == True) :
            self.numeric_features += self.numeric_features_grp3
        if (inclFlgFG4 == True) :
            self.numeric_features += self.numeric_features_grp4
        if (inclFlgFG5 == True) :
            self.numeric_features += self.numeric_features_grp5
        if (inclFlgFG6 == True) :
            self.numeric_features += self.numeric_features_grp6

        print(f"# Numeric Features are as follows:{self.numeric_features}")

        # lists to hold our attack classifications
        self.attack_categories = ['Normal','DoS','Probe','Privilege','Access']
        self.attacks_dos = ['apache2','back','land','neptune','mailbomb','pod',
                            'processtable','smurf','teardrop','udpstorm','worm']
        self.attacks_probe = ['ipsweep','mscan','nmap','portsweep','saint','satan']
        self.attacks_privilege = ['buffer_overflow','loadmdoule','perl','ps','rootkit','sqlattack','xterm']
        self.attacks_access = ['ftp_write','guess_passwd','http_tunnel','imap','multihop','named', 'phf',
                               'sendmail','snmpgetattack','snmpguess','spy','warezclient','warezmaster','xclock','xsnoop']

        self.labels = ['attack_class']

        # add the column name
        self.dfTrain.columns = self.columns
        self.dfTest.columns = self.columns
        #

        # helper function to map an attack_name to a label (2-class)
        def map_attack2(attack):
            if attack in self.attacks_dos:
                # dos_attacks map to 1
                attack_type = 1
            elif attack in self.attacks_probe:
                # probe_attacks mapt to 1
                attack_type = 1
            elif attack in self.attacks_privilege:
                # privilege escalation attacks map to 1
                attack_type = 1
            elif attack in self.attacks_access:
                # remote access attacks map to 1
                attack_type = 1
            else:
                # normal maps to 0
                attack_type = 0
            return attack_type

        # helper function to map an attack_name to a label (5-class)
        def map_attack5(attack):
            if attack in self.attacks_dos:
                # dos_attacks map to 1
                attack_type = 1
            elif attack in self.attacks_probe:
                # probe_attacks mapt to 2
                attack_type = 2
            elif attack in self.attacks_privilege:
                # privilege escalation attacks map to 3
                attack_type = 3
            elif attack in self.attacks_access:
                # remote access attacks map to 4
                attack_type = 4
            else:
                # normal maps to 0
                attack_type = 0
            return attack_type
        
        # map an attack name to a label
        if (self.numOfLabels == 2):
            self.dfTrain['attack_class'] = self.dfTrain.attack_name.apply(map_attack2)
            self.dfTest['attack_class'] = self.dfTest.attack_name.apply(map_attack2)
        elif (self.numOfLabels == 5):
            self.dfTrain['attack_class'] = self.dfTrain.attack_name.apply(map_attack5)
            self.dfTest['attack_class'] = self.dfTest.attack_name.apply(map_attack5)
        else:
            raise Exception("unsupported number of labels.")

        # get the intial set of encoded features and encode them
        encodedTrain = pd.get_dummies(self.dfTrain[self.categorical_features])
        encodedTest = pd.get_dummies(self.dfTest[self.categorical_features])
        # not all of the features are in the test set, so we need to account for diffs
        indexTest = np.arange(len(self.dfTest.index))
        column_diffs = list(set(encodedTrain.columns.values)-set(encodedTest.columns.values))
        diff_df = pd.DataFrame(0, index=indexTest, columns=column_diffs)
        # we'll also need to reorder the columns to match, so let's get those
        column_order = encodedTrain.columns.to_list()
        # append the new columns
        encodedTest_temp = encodedTest.join(diff_df)
        # reorder the columns
        encodedTestFinal = encodedTest_temp[column_order].fillna(0)

        # final dataset
        self.dfTrainData = encodedTrain.join(self.dfTrain[self.numeric_features])
        self.dfTestData = encodedTestFinal.join(self.dfTest[self.numeric_features])
        # self.dfTrainData = pd.merge(encodedTrain,self.dfTrain[self.numeric_features],left_index=True, right_index=True)
        # self.dfTestData = pd.merge(encodedTestFinal,self.dfTest[self.numeric_features],left_index=True, right_index=True)

        # set features from dataframe column names
        self.features = list(self.dfTrainData.columns)

        # add label columns
        self.dfTrainData = self.dfTrainData.join(self.dfTrain[self.labels])
        self.dfTestData = self.dfTestData.join(self.dfTest[self.labels])

    def getDf(self):
        return self.dfTrainData, self.dfTestData
    
    def getFeatures(self):
        return self.features

    def getLabels(self):
        return self.labels

    def showDfInputFile(self):
        print("###############################")
        print("Train Dataframe")
        print(self.dfTrain.columns)
        print(self.dfTrain)
        vc = self.dfTrain['attack_name'].value_counts()
        print(vc)
        vc = self.dfTrain['duration'].value_counts()
        print(vc)
        print("Test Dataframe")
        print(self.dfTest.columns)
        print(self.dfTest)
        vc = self.dfTest['attack_name'].value_counts()
        vc = self.dfTrain['duration'].value_counts()
        print(vc)
        print(vc)

    def show(self):
        print("##############################################################")
        print(f"train file name is {self.fNameTrain}.")
        print(f"test file name is {self.fNameTest}.")
        print("Features")
        print(self.features)
        print("Labels")
        print(self.labels)
        print("##########################")
        print(self.dfTrainData.columns)
        print(f"len = {len(self.dfTrainData.columns)}")
        print("Train Dataset")
        print(self.dfTrainData)
        print("Train Attack Class")
        print(self.dfTrainData['attack_class'])
        vc = self.dfTrainData['attack_class'].value_counts()
        print(vc)
        #
        print(self.dfTestData.columns)
        print(f"len = {len(self.dfTestData.columns)}")
        print("Test Dataset")
        print(self.dfTestData)
        print("Test Attack Name")
        print(self.dfTest['attack_name'])
        print("Test Attack Class")
        print(self.dfTestData['attack_class'])
        vc = self.dfTestData['attack_class'].value_counts()
        print(vc)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-od", "--dirDataset", type=str, default="~/Research2/dataset")  
    parser.add_argument("-n", "--numOfLabels", type=int, choices=[2, 5], default=2)  

    args = parser.parse_args()
    dirDataset = args.dirDataset+'/NSL-KDD'
    numOfLabels = args.numOfLabels
    if (numOfLabels != 2) and (numOfLabels != 5):
        raise Exception("unsupported number of labels.")
    #
    file_path_20_percent = dirDataset+'/KDDTrain+_20Percent.txt'
    file_path_full_training_set = dirDataset+'/KDDTrain+.txt'
    file_path_test = dirDataset+'/KDDTest+.txt' 
    #
    dataset = NSL_KDD(fnameTrain = file_path_20_percent,
                      fnameTest = file_path_test,
                      numOfLabels = numOfLabels)
    dataset.show()
    dataset.showDfInputFile()
    #
    dataset = NSL_KDD(fnameTrain = file_path_full_training_set,
                      fnameTest = file_path_test,
                      numOfLabels = numOfLabels)
    dataset.show()
    dataset.showDfInputFile()

