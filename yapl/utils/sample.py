import tensorflow as tf
import torch 

import yalp

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