import tensorflow as tf
import torch 

import yalp

class BasicBinaryClassifier:
    def __init__(self, is_sequential=True):
        self.is_sequential = is_sequential
        self.backend = yalp.backend


    def _model(self):
        if self.backend == 'tf':
            if self.is_sequential:
                #Simple binary classification model for 1D input
                model = tf.keras.Sequential()
                model.add(tf.keras.Input(shape=yalp.config.INPUT_SHAPE))

                if yalp.config.DATATYPE == 'img':
                    model.add(tf.keras.layers.Conv2D(32, 3, activation="relu"))
                    model.add(tf.keras.layers.Conv2D(64, 3, activation="relu"))
                    model.add(tf.keras.layers.MaxPooling2D(3))
                    model.add(tf.keras.layers.Flatten())

                if yalp.config.DATATYPE == 'text':
                    #TODO
                
                if yalp.config.DATATYPE == 'tabular':
                    model.add(tf.keras.layers.Dense(32))
                    model.add(tf.keras.layers.Dense(64))
                    model.add(tf.keras.layers.Dense(128))

                if yalp.config.PROBLEM_TYPE == "classification":
                    model.add(tf.keras.layers.Dense(yalp.config.NUM_CLASSES, activation='sigmoid'))
                
                elif yalp.config.PROBLEM_TYPE == "regression":
                    model.add(tf.keras.layers.Dense(yalp.config.NUM_CLASSES))
                

                return model








class BasicTF(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.input_layer = tf.keras.layers.InputLayer(input_shape = config.IMG_SHAPE)
        self.dense1 = tf.keras.layers.Dense(100, activation='sigmoid')
        self.out = tf.keras.layers.Dense(1, activation='sigmoid')
        
    def call(self, input_batch):
        x = self.input_layer(input_batch)
        x = self.dense1(x)
        x = self.out(x)
        
        return x

class SampleModel:
    def __init__(self):
        pass
    
    def tf(self):
        
        return BasicTF()


def samplerunTF():
    #Creating Dataset
    dataset = (
        tf.data.TFRecordDataset(
            config.TRAIN_FILES,  
            num_parallel_reads=tf.data.experimental.AUTOTUNE
        ).map(
            process_training_data,
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        ).repeat(
        ).shuffle(
            buffer_size=config.BUFFER_SIZE
        ).batch(
            config.BATCH_SIZE
        ).prefetch(
            tf.data.experimental.AUTOTUNE
        )
    )
    
    #Setup model and train
    if config.STRATEGY is not None:
        with strategy.scope():
            model = get_model(is_sequential = True)
    else:
        model = get_model(is_sequential = True)
        
    history = fit_engine(model, dataset)
        
    return model, history



def samplerunTorch():
    ## Reading Data files
    df = pd.read_csv(config.TRAIN_CSV).values
    
    dataset = torch.utils.data.DataLoader(
        ImageLoader(
            image_files = df[:,0],
            targets = df[:,1]
        ),
        batch_size = config.BATCH_SIZE,
        num_workers = 4
    )
    
    #Model prepration
    model = Resnet50()
    model.to(config.DEVICE)
    config.OPTIMIZER = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    #Epoches
    for epoch in range(config.EPOCHES):
        print("Epoch {}:".format(epoch))
        loss, acc = train_engine(dataset, model)
        print("EPOCH {}- LOSS: {} | Accuracy: {}".format(epoch, loss, mean(acc)))
        
    return model