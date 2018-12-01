def ensembleModels(ymodel):
    model_input = Input(shape=ymodel[0].input_shape[1:])
    yModels=[model(model_input) for model in ymodel] 
    yAvg=layers.average(yModels)
    modelEns = Model(inputs=model_input, outputs=yAvg, name='ensemble')
    print (modelEns.summary())
    modelEns.save('model.h5')

m1 = load_model('./cnnModel/model1/model-00179-0.70949.h5')
m2 = load_model('./cnnModel/model1/model-00149-0.70616.h5')
m3 = load_model('./cnnModel/model2/model-00398-0.70469.h5')
m4 = load_model('./cnnModel/model3/model-00388-0.70764.h5')
m5 = load_model('./cnnModel/model4/model-00346-0.70543.h5')

if __name__ == '__main__':
   ensembleModels(ymodel=[m1, m2, m3, m4, m5])