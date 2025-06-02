import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

class Sample_Split_Scaler:
    def __init__(self):
        self.sections =None
        self.shuffle =None
        self.x =None
        self.y =None
        self.netz = None
        self.scaler_method =None
        self.scalerX =None
        self.scalerY =None
        self.method = None
# Samples aufspalten in Training, Validierung, Test
    def split_sample(self,x,y,shuffle=True,sections=(.6,.25,.15)):
        self.sections = sections
        self.shuffle = shuffle
        tmp= train_test_split(x, y, test_size=np.sum(sections[1:]),shuffle=shuffle)
        # zusätzliche Unterteilung für Testdaten
        tmp2 = train_test_split(tmp[1], tmp[3], test_size=sections[-1],shuffle=shuffle)

        # 0 = train, 1= val, 2= test
        if tmp[0].ndim ==1:
            self.x=list();self.x.append(tmp[0].reshape(-1,1));self.x.append(tmp2[0].reshape(-1,1));self.x.append(tmp2[1].reshape(-1,1))
            self.y=list();self.y.append(tmp[0].reshape(-1,1));self.y.append(tmp2[0].reshape(-1,1));self.y.append(tmp2[1].reshape(-1,1))

        else:
            self.x=list();self.x.append(tmp[0]);self.x.append(tmp2[0]);self.x.append(tmp2[1])    
            self.y=list();self.y.append(tmp[2]);self.y.append(tmp2[2]);self.y.append(tmp2[3])    
    def scaler(self,method):
        self.scaler_method =method
        if self.scaler_method =="MinMax":
            self.scalerX = MinMaxScaler()
            self.scalerY = MinMaxScaler()
        elif self.scaler_method =="Standard":
            self.scalerX = StandardScaler()
            self.scalerY = StandardScaler()
        # 0 = train, 1= val, 2= test
        self.x_norm = list();
	# Fitting nur auf Trainingsdaten
        self.x_norm.append(self.scalerX.fit_transform(self.x[0]))
        [self.x_norm.append(self.scalerX.transform(self.x[i])) for i in range(1,3)]
        self.y_norm = list();self.y_norm.append(self.scalerY.fit_transform(self.y[0]));
        [self.y_norm.append(self.scalerY.fit_transform(self.y[i])) for i in range(1,3)];
        


def sample_generator_lags(x,y,timesteps,horizon,dict_input_lags,lag_max,inputs_future,batch_size=500,epochs=10):
    # Auswahl der Features (inkl. lags)
    while True:
        list=[]
        # Schleife über alle zu verwendenden inputs
        
        for input in dict_input_lags.keys():
        # Schleife über alle zu betrachtenden lags
            for lagi in dict_input_lags[input]:
                # Matrix wird zugeschnitten, erster Datensatz beginnt mit t =lag_max - min[lag[i]] 
                if lagi==0:
                    list.append(x[lag_max-lagi:,input])
                else:
                    list.append(x[lag_max-lagi:-lagi,input])    
            # Umwandlung in numPy        
        xx = np.array(list).T
        yy =y[lag_max:]
    # xx,yy sind nun die Rohdaten aus denen die Samples abgeleitet werden können 
        
    #* Schleife über Epochen *********************
    #******** je Batchsize werden Daten zurückgeliefert
        batch_count=0
        x_,xfuture_,y_=[],[],[] 
        for epo in range(epochs):
    # ************ Erstellung der Samples aus den Rohdaten*******************************
        
            # Anzahl der verkürzen sich noch um die notwendigen timesteps (nach hinten) 
            #sowie die Anzahl der Prognoseschritte (nach vorne)
            idx_range = range(len(xx) - (timesteps + horizon))
    
            # Schleife für jedes Samples
    
            # Für jeden Tag liegt ein Beobachtungswert vor
            for idx in idx_range:
                # erster Beobachtungswert(e) liegt am ende der "ersten Zeitserie
                y_.append(np.squeeze(yy[idx+timesteps-1:idx+timesteps+horizon-1]))
                # zurodnung der Zeitserie [timesteps x features]
                # features welche nciht im LSTM-Netzwerk sondern erst nachfolgend betrachtet werden
                if not(inputs_future == []):
                    xfuture_.append(np.squeeze(x[idx+timesteps-1:idx+timesteps+horizon-1,inputs_future]))
                    
                x_.append(xx[idx:idx+timesteps])
                batch_count+=1
                if (batch_count==batch_size or idx ==idx_range[-1]):
                    if inputs_future == []:
                        yield np.array(x_),np.array(y_) 
                    else:
                        if np.ndim(xfuture_)==1:
                            yield (np.array(x_),np.array(xfuture_).reshape(-1,1)),np.array(y_) 
                        else:
                            yield (np.array(x_),np.array(xfuture_)),np.array(y_) 
                                
                        
                    # Rückgabe des Batch       
                    x_=[]
                    y_=[]
                    xfuture_=[]
                    batch_count=0
