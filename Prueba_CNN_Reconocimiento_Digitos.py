# 2. Configuración e importaciones
import os
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    print('Backend: TensorFlow')
except ImportError:
    os.environ['KERAS_BACKEND']='torch'
    import keras
    from keras import layers
    print('Backend:', keras.backend.backend())

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

SEED=42
keras.utils.set_random_seed(SEED)

# 3. Carga del archivo Excel
RUTA='07. Apoyo desafío - digitos_mnist_simple.xlsx'
if not os.path.exists(RUTA):
    try:
        from google.colab import files
        uploaded=files.upload()
        RUTA=next(iter(uploaded.keys()))
    except ImportError:
        raise FileNotFoundError('Ajusta RUTA a la ubicación del archivo Excel')

df=pd.read_excel(RUTA)
print('Dimensiones:',df.shape)
display(df.head())
print(df['label'].value_counts().sort_index())

# 4. Reshape, normalización y split
X=df.drop(columns=['label']).to_numpy(dtype='float32').reshape(-1,8,8,1)/16.0
y=df['label'].to_numpy(dtype='int64')
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.20,random_state=SEED,stratify=y)
X_tr,X_val,y_tr,y_val=train_test_split(X_train,y_train,test_size=.20,random_state=SEED,stratify=y_train)
print(X_tr.shape,X_val.shape,X_test.shape)

# 5. Visualizar ejemplos
fig,axs=plt.subplots(2,5,figsize=(9,4))
for i,ax in enumerate(axs.ravel()):
    ax.imshow(X[i,:,:,0],cmap='gray_r')
    ax.set_title(f'Dígito {y[i]}')
    ax.axis('off')
plt.tight_layout();plt.show()

# 6. CNN baseline
def crear_baseline():
    model=keras.Sequential([
        layers.Input((8,8,1)),
        layers.Conv2D(16,3,activation='relu',padding='same'),
        layers.MaxPooling2D(2),
        layers.Flatten(),
        layers.Dense(32,activation='relu'),
        layers.Dense(10,activation='softmax')
    ])
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    return model

baseline=crear_baseline()
baseline.summary()
hist_base=baseline.fit(X_tr,y_tr,validation_data=(X_val,y_val),epochs=12,batch_size=32,verbose=1)
base_loss,base_acc=baseline.evaluate(X_test,y_test,verbose=0)
print(f'Baseline accuracy={base_acc:.4f} loss={base_loss:.4f}')

# 7. CNN optimizada
def crear_optimizada():
    model=keras.Sequential([
        layers.Input((8,8,1)),
        layers.Conv2D(32,3,activation='relu',padding='same'),
        layers.MaxPooling2D(2),
        layers.Conv2D(64,3,activation='relu',padding='same'),
        layers.MaxPooling2D(2),
        layers.Flatten(),
        layers.Dense(64,activation='relu'),
        layers.Dropout(.30),
        layers.Dense(10,activation='softmax')
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=.001),
                  loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    return model

optimizada=crear_optimizada()
early=keras.callbacks.EarlyStopping(monitor='val_loss',patience=4,restore_best_weights=True)
reduce=keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=.5,patience=2,min_lr=1e-5)
hist_opt=optimizada.fit(X_tr,y_tr,validation_data=(X_val,y_val),epochs=30,batch_size=32,callbacks=[early,reduce],verbose=1)
opt_loss,opt_acc=optimizada.evaluate(X_test,y_test,verbose=0)
print(f'Optimizada accuracy={opt_acc:.4f} loss={opt_loss:.4f}')
print(f'Cambio accuracy={(opt_acc-base_acc)*100:+.2f} puntos porcentuales')

# 8. Curvas accuracy y loss
fig,axs=plt.subplots(1,2,figsize=(12,4))
axs[0].plot(hist_base.history['val_accuracy'],label='Baseline val')
axs[0].plot(hist_opt.history['val_accuracy'],label='Optimizada val')
axs[0].set_title('Validation accuracy');axs[0].legend();axs[0].grid(alpha=.3)
axs[1].plot(hist_base.history['val_loss'],label='Baseline val')
axs[1].plot(hist_opt.history['val_loss'],label='Optimizada val')
axs[1].set_title('Validation loss');axs[1].legend();axs[1].grid(alpha=.3)
plt.tight_layout();plt.show()

# 9. Matriz de confusión y métricas
probs=optimizada.predict(X_test,verbose=0)
pred=probs.argmax(axis=1)
cm=confusion_matrix(y_test,pred)
print(classification_report(y_test,pred,zero_division=0))
plt.figure(figsize=(7,6));plt.imshow(cm,cmap='Blues');plt.colorbar();plt.xticks(range(10));plt.yticks(range(10));
plt.xlabel('Predicción');plt.ylabel('Real');plt.title('Matriz de confusión')
for i in range(10):
    for j in range(10): plt.text(j,i,cm[i,j],ha='center',va='center',fontsize=8)
plt.tight_layout();plt.show()

