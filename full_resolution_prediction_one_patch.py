fullimage = n
output_preds = list()
output_preds_final = []


for x in tqdm(range(0, 224)):
    #print(x)
    patchforprediction_batch = []
    for i in range(len(poles)):
        #print(i)
        
        #intervals = [poles[i], poles[i]+32]
        for y in range(poles[i], poles[i]+32):
        #for y in intervals:
            #print(y)
            patchforprediction = fullimage[x:x+224, y:y+224]
            patchforprediction_batch.append(patchforprediction)
            
            X_train = np.array(patchforprediction_batch)

        preds = predict_batch_from_model(X_train, model)
        
        #print(preds)
            
        #output_preds.extend(preds)

    #output_preds_final.append(output_preds)
    output_preds_final.append(preds)

output_preds_final = np.array(output_preds_final)
    #print(output_preds_final)

#predfullresolution = np.vstack(output_preds_final)
#np.save('predfullresolution_patch_%d_%d' % (fullimage.shape[0], fullimage.shape[1]), predfullresolution)
np.save('output_preds_final_%d_%d' % (fullimage.shape[0], fullimage.shape[1]), output_preds_final)
