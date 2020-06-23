import yaml
import yapl

class Config:
    '''
        Extend this Base class to avoid Re-writing most of the things

        example:
            class config(Config):
                def __init__(self):
                    super().__init__()
                    self.EXPERIMENT = {
                        'NAME':"Exp1",
                        'DATE' : datetime.datetime.now(),
                    }
                    self.DATA_FILE = '../hsjdhfs/sdfjsk.zip'
                    self.DEVICE = 'cpu'

            config = config()
            config.DEVICE
            config.BATCHSIZE
    '''

    def __init__(self):
        # General Experiment Params
        self.EXPERIMENT_NAME = 'img_class_resner50_trail25'
        self.DATATYPE = 'img' #img: if images; text: if textual; tabular: if csv tabular
        self.PROBLEM_TYPE = 'classification' #classification, regression
        
        self.GCS_DS_PATH = KaggleDatasets().get_gcs_path('<dataset-name>')
        self.TRAIN_CSV = '../input/<dataset-name>/train.csv'
        self.TEST_CSV = '../input/<dataset-name>/test.csv'
        # self.TRAIN_FILES = tf.io.gfile.glob(GCS_DS_PATH + '<files>*')
        # self.TEST_FILES = tf.io.gfile.glob(GCS_DS_PATH + '<files>')
        self.VALIDATION_CSV = ""
        
        self.TOTAL_TRAIN_IMG = 0
        self.TOTAL_TEST_IMG = 0
        
        #Data specific information
        self.INPUT_SHAPE = (128,128,3) #exam: image shape
        self.NUM_CLASSES = 1 #one for binary classfication and regreesion
        self.IMG_SIZE = [1024, 1024]
        self.IMG_SHAPE = (512, 512, 3)

        #model specific
        self.DO_FINETUNE = True
        
        
        #Training information
        self.BATCH_SIZE = 8
        self.BUFFER_SIZE = 100
        self.EPOCHES = 10 
        
        # self.LOSS = tf.keras.losses.BinaryCrossentropy()
        # self.OPTIMIZER = tf.keras.optimizers.Adam(learning_rate=0.01)
        self.ACCURACY = ['accuracy']
        
        self.STRATEGY = None
        
        self.LOG_DIR = './log'
        self.CHECKPOINT_DIR = './log/checkpoint/cp.cpkt'


    def dumpconfig(self, location):
        with open(location, 'w') as dumpfile:
            yaml.dump(self.__dict__, dumpfile)

        return "you file has successfully saved"

    def make_global(self):
        yapl.config = self