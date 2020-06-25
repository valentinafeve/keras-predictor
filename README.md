```python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
```


```python
!pip install -q -U tensorflow
```


```python
import itertools
import os
import math
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
layers = keras.layers

```


```python
!wget https://storage.googleapis.com/sara-cloud-ml/wine_data.csv
```


```python
path = "wine_data.csv"
```


```python
data = pd.read_csv(path)
data = data.sample(frac=1)
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>country</th>
      <th>description</th>
      <th>designation</th>
      <th>points</th>
      <th>price</th>
      <th>province</th>
      <th>region_1</th>
      <th>region_2</th>
      <th>variety</th>
      <th>winery</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>13618</th>
      <td>13618</td>
      <td>Italy</td>
      <td>Bright and refined, this conveys aromas of dar...</td>
      <td>NaN</td>
      <td>91</td>
      <td>26.0</td>
      <td>Tuscany</td>
      <td>Vino Nobile di Montepulciano</td>
      <td>NaN</td>
      <td>Red Blend</td>
      <td>Bindella</td>
    </tr>
    <tr>
      <th>93481</th>
      <td>93481</td>
      <td>US</td>
      <td>Oaky-sweet and simple, with jammy pineapple, t...</td>
      <td>Appellation Series</td>
      <td>84</td>
      <td>15.0</td>
      <td>California</td>
      <td>Russian River Valley</td>
      <td>Sonoma</td>
      <td>Chardonnay</td>
      <td>Healdsburg Ranches</td>
    </tr>
    <tr>
      <th>143346</th>
      <td>143346</td>
      <td>South Africa</td>
      <td>Clean, fresh flavors, good body and a balance ...</td>
      <td>NaN</td>
      <td>87</td>
      <td>11.0</td>
      <td>Western Cape</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Sauvignon Blanc</td>
      <td>Douglas Green</td>
    </tr>
    <tr>
      <th>28167</th>
      <td>28167</td>
      <td>France</td>
      <td>This is a ripe, fresh and fruity wine that's f...</td>
      <td>NaN</td>
      <td>89</td>
      <td>21.0</td>
      <td>Loire Valley</td>
      <td>Sancerre</td>
      <td>NaN</td>
      <td>Sauvignon Blanc</td>
      <td>Domaine de Rome</td>
    </tr>
    <tr>
      <th>52757</th>
      <td>52757</td>
      <td>US</td>
      <td>This full-bodied Chardonnay begins with aromas...</td>
      <td>Golden Glen</td>
      <td>85</td>
      <td>20.0</td>
      <td>New York</td>
      <td>Finger Lakes</td>
      <td>Finger Lakes</td>
      <td>Chardonnay</td>
      <td>Glenora</td>
    </tr>
  </tbody>
</table>
</div>




```python
data = data[pd.notnull(data['country'])]
data = data[pd.notnull(data['price'])]
data = data.drop(data.columns[0], axis=1)
variety_threshold = 500
value_counts = data['variety'].value_counts()
to_remove = value_counts[value_counts <= variety_threshold].index
data.replace(to_remove , np.nan, inplace=True)
data = data[pd.notnull(data['variety'])]
```


```python
train_size= int(len(data) * .8)
print("Train size: %d" % train_size)
print("Test size: %d" % (len(data) - train_size))
```

    Train size: 95646
    Test size: 23912



```python
# Train inputs
description_train = data['description'][:train_size]
variety_train = data['variety'][:train_size]

# Train labels
labels_train = data['price'][:train_size]

# Test inputs
description_test = data['description'][train_size:]
variety_test = data['variety'][train_size:]

# Test labels
labels_test = data['price'][train_size:]
```


```python
# Create a tokenizer
vocab_size = 5000
tokenize = keras.preprocessing.text.Tokenizer(num_words=vocab_size, char_level=False)
tokenize.fit_on_texts(description_train)
```


```python
description_bow_train = tokenize.texts_to_matrix(description_train)
description_bow_test = tokenize.texts_to_matrix(description_test)
```


```python
encoder = LabelEncoder()
encoder.fit(variety_train)
variety_train = encoder.transform(variety_train)
variety_test = encoder.transform(variety_test)
num_classes = np.max(variety_train) + 1

#To one-hot
variety_train = keras.utils.to_categorical(variety_train, num_classes)
variety_test = keras.utils.to_categorical(variety_test, num_classes)
```


```python
bow_inputs = layers.Input(shape=(vocab_size,))
variety_inputs = layers.Input(shape=(num_classes,))
merged_layer= layers.concatenate([bow_inputs, variety_inputs])
merget_layer = layers.Dense(256, activation="relu")(merged_layer)
predictions = layers.Dense(1)(merged_layer)
wide_model = keras.Model(inputs=[bow_inputs, variety_inputs], outputs=predictions)
```


```python
wide_model.compile(loss="mse", optimizer='adam', metrics=['accuracy'])
print(wide_model.summary())
```

    Model: "model"
    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to                     
    ==================================================================================================
    input_1 (InputLayer)            [(None, 5000)]       0                                            
    __________________________________________________________________________________________________
    input_2 (InputLayer)            [(None, 40)]         0                                            
    __________________________________________________________________________________________________
    concatenate (Concatenate)       (None, 5040)         0           input_1[0][0]                    
                                                                     input_2[0][0]                    
    __________________________________________________________________________________________________
    dense_1 (Dense)                 (None, 1)            5041        concatenate[0][0]                
    ==================================================================================================
    Total params: 5,041
    Trainable params: 5,041
    Non-trainable params: 0
    __________________________________________________________________________________________________
    None



```python
train_embed = tokenize.texts_to_sequences(description_train)
test_embed = tokenize.texts_to_sequences(description_test)

max_seq_length = 170
train_embed = keras.preprocessing.sequence.pad_sequences(train_embed, maxlen=max_seq_length, padding = "post")
test_embed = keras.preprocessing.sequence.pad_sequences(test_embed, maxlen=max_seq_length, padding = "post")
```


```python
deep_inputs = layers.Input(shape=(max_seq_length,))
embedding = layers.Embedding(vocab_size, 8, input_length=max_seq_length)(deep_inputs)
embedding = layers.Flatten()(embedding)
embed_out = layers.Dense(1)(embedding)
deep_model = keras.Model(inputs=deep_inputs, outputs=embed_out)
print(deep_model.summary())

```

    Model: "model_1"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_3 (InputLayer)         [(None, 170)]             0         
    _________________________________________________________________
    embedding (Embedding)        (None, 170, 8)            40000     
    _________________________________________________________________
    flatten (Flatten)            (None, 1360)              0         
    _________________________________________________________________
    dense_2 (Dense)              (None, 1)                 1361      
    =================================================================
    Total params: 41,361
    Trainable params: 41,361
    Non-trainable params: 0
    _________________________________________________________________
    None



```python
deep_model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
```


```python
merged_out = layers.concatenate([wide_model.output, deep_model.output])
merged_out = layers.Dense(1)(merged_out)
combined_model = keras.Model(wide_model.input + [deep_model.input], merged_out)
print(combined_model.summary())
```

    Model: "model_2"
    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to                     
    ==================================================================================================
    input_3 (InputLayer)            [(None, 170)]        0                                            
    __________________________________________________________________________________________________
    input_1 (InputLayer)            [(None, 5000)]       0                                            
    __________________________________________________________________________________________________
    input_2 (InputLayer)            [(None, 40)]         0                                            
    __________________________________________________________________________________________________
    embedding (Embedding)           (None, 170, 8)       40000       input_3[0][0]                    
    __________________________________________________________________________________________________
    concatenate (Concatenate)       (None, 5040)         0           input_1[0][0]                    
                                                                     input_2[0][0]                    
    __________________________________________________________________________________________________
    flatten (Flatten)               (None, 1360)         0           embedding[0][0]                  
    __________________________________________________________________________________________________
    dense_1 (Dense)                 (None, 1)            5041        concatenate[0][0]                
    __________________________________________________________________________________________________
    dense_2 (Dense)                 (None, 1)            1361        flatten[0][0]                    
    __________________________________________________________________________________________________
    concatenate_1 (Concatenate)     (None, 2)            0           dense_1[0][0]                    
                                                                     dense_2[0][0]                    
    __________________________________________________________________________________________________
    dense_3 (Dense)                 (None, 1)            3           concatenate_1[0][0]              
    ==================================================================================================
    Total params: 46,405
    Trainable params: 46,405
    Non-trainable params: 0
    __________________________________________________________________________________________________
    None



```python
combined_model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
```


```python
combined_model.fit([description_bow_train, variety_train] + [ train_embed ], labels_train, epochs=50, batch_size=128)
combined_model.evaluate([description_bow_test, variety_test] + [ test_embed ], labels_test, batch_size=128)
```

    Epoch 1/50
    748/748 [==============================] - 4s 5ms/step - loss: 941.3827 - accuracy: 0.0000e+00
    Epoch 2/50
    748/748 [==============================] - 3s 4ms/step - loss: 930.1476 - accuracy: 0.0000e+00
    Epoch 3/50
    748/748 [==============================] - 3s 4ms/step - loss: 918.2635 - accuracy: 0.0000e+00
    Epoch 4/50
    748/748 [==============================] - 3s 4ms/step - loss: 905.2264 - accuracy: 0.0000e+00
    Epoch 5/50
    748/748 [==============================] - 3s 4ms/step - loss: 889.9504 - accuracy: 0.0000e+00
    Epoch 6/50
    748/748 [==============================] - 3s 4ms/step - loss: 873.7689 - accuracy: 0.0000e+00
    Epoch 7/50
    748/748 [==============================] - 3s 4ms/step - loss: 854.2842 - accuracy: 0.0000e+00
    Epoch 8/50
    748/748 [==============================] - 3s 4ms/step - loss: 833.3561 - accuracy: 0.0000e+00
    Epoch 9/50
    748/748 [==============================] - 3s 4ms/step - loss: 811.6473 - accuracy: 0.0000e+00
    Epoch 10/50
    748/748 [==============================] - 3s 4ms/step - loss: 789.2904 - accuracy: 0.0000e+00
    Epoch 11/50
    748/748 [==============================] - 3s 4ms/step - loss: 766.7440 - accuracy: 0.0000e+00
    Epoch 12/50
    748/748 [==============================] - 3s 4ms/step - loss: 745.5654 - accuracy: 0.0000e+00
    Epoch 13/50
    748/748 [==============================] - 3s 4ms/step - loss: 724.9681 - accuracy: 0.0000e+00
    Epoch 14/50
    748/748 [==============================] - 3s 4ms/step - loss: 705.7652 - accuracy: 0.0000e+00
    Epoch 15/50
    748/748 [==============================] - 3s 4ms/step - loss: 687.3585 - accuracy: 0.0000e+00
    Epoch 16/50
    748/748 [==============================] - 3s 4ms/step - loss: 671.9308 - accuracy: 0.0000e+00
    Epoch 17/50
    748/748 [==============================] - 3s 4ms/step - loss: 654.9420 - accuracy: 0.0000e+00
    Epoch 18/50
    748/748 [==============================] - 3s 4ms/step - loss: 640.7798 - accuracy: 0.0000e+00
    Epoch 19/50
    748/748 [==============================] - 3s 4ms/step - loss: 627.2385 - accuracy: 0.0000e+00
    Epoch 20/50
    748/748 [==============================] - 3s 4ms/step - loss: 614.8297 - accuracy: 0.0000e+00
    Epoch 21/50
    748/748 [==============================] - 3s 4ms/step - loss: 602.9612 - accuracy: 0.0000e+00
    Epoch 22/50
    748/748 [==============================] - 3s 4ms/step - loss: 591.8800 - accuracy: 0.0000e+00
    Epoch 23/50
    748/748 [==============================] - 3s 4ms/step - loss: 581.5754 - accuracy: 0.0000e+00
    Epoch 24/50
    748/748 [==============================] - 3s 4ms/step - loss: 571.9762 - accuracy: 0.0000e+00
    Epoch 25/50
    748/748 [==============================] - 3s 4ms/step - loss: 562.5278 - accuracy: 0.0000e+00
    Epoch 26/50
    748/748 [==============================] - 3s 4ms/step - loss: 553.9205 - accuracy: 0.0000e+00
    Epoch 27/50
    748/748 [==============================] - 3s 4ms/step - loss: 545.8486 - accuracy: 0.0000e+00
    Epoch 28/50
    748/748 [==============================] - 3s 4ms/step - loss: 538.1983 - accuracy: 0.0000e+00
    Epoch 29/50
    748/748 [==============================] - 3s 4ms/step - loss: 530.8666 - accuracy: 0.0000e+00
    Epoch 30/50
    748/748 [==============================] - 3s 4ms/step - loss: 523.5403 - accuracy: 0.0000e+00
    Epoch 31/50
    748/748 [==============================] - 3s 4ms/step - loss: 516.7404 - accuracy: 0.0000e+00
    Epoch 32/50
    748/748 [==============================] - 3s 4ms/step - loss: 510.3697 - accuracy: 0.0000e+00
    Epoch 33/50
    748/748 [==============================] - 3s 4ms/step - loss: 504.5469 - accuracy: 0.0000e+00
    Epoch 34/50
    748/748 [==============================] - 3s 4ms/step - loss: 498.6843 - accuracy: 0.0000e+00
    Epoch 35/50
    748/748 [==============================] - 3s 4ms/step - loss: 493.1948 - accuracy: 0.0000e+00
    Epoch 36/50
    748/748 [==============================] - 3s 4ms/step - loss: 488.0857 - accuracy: 0.0000e+00
    Epoch 37/50
    748/748 [==============================] - 3s 4ms/step - loss: 482.2809 - accuracy: 0.0000e+00
    Epoch 38/50
    748/748 [==============================] - 3s 4ms/step - loss: 477.8959 - accuracy: 0.0000e+00
    Epoch 39/50
    748/748 [==============================] - 3s 4ms/step - loss: 473.2823 - accuracy: 0.0000e+00
    Epoch 40/50
    748/748 [==============================] - 3s 4ms/step - loss: 468.5371 - accuracy: 0.0000e+00
    Epoch 41/50
    748/748 [==============================] - 3s 4ms/step - loss: 464.2664 - accuracy: 0.0000e+00
    Epoch 42/50
    748/748 [==============================] - 3s 4ms/step - loss: 460.0229 - accuracy: 0.0000e+00
    Epoch 43/50
    748/748 [==============================] - 3s 4ms/step - loss: 455.9980 - accuracy: 0.0000e+00
    Epoch 44/50
    748/748 [==============================] - 3s 4ms/step - loss: 452.2878 - accuracy: 0.0000e+00
    Epoch 45/50
    748/748 [==============================] - 3s 4ms/step - loss: 448.5306 - accuracy: 0.0000e+00
    Epoch 46/50
    748/748 [==============================] - 3s 4ms/step - loss: 444.7486 - accuracy: 0.0000e+00
    Epoch 47/50
    748/748 [==============================] - 3s 4ms/step - loss: 441.3842 - accuracy: 0.0000e+00
    Epoch 48/50
    748/748 [==============================] - 3s 4ms/step - loss: 437.7628 - accuracy: 0.0000e+00
    Epoch 49/50
    748/748 [==============================] - 3s 4ms/step - loss: 434.3597 - accuracy: 0.0000e+00
    Epoch 50/50
    748/748 [==============================] - 3s 4ms/step - loss: 431.1888 - accuracy: 0.0000e+00
    187/187 [==============================] - 1s 3ms/step - loss: 996.4149 - accuracy: 0.0000e+00





    [996.4148559570312, 0.0]




```python
predictions = combined_model.predict([ description_bow_test, variety_test] + [ test_embed])
```


```python
num_predictions = 40
diff = 0
for i in range(num_predictions):
  val = predictions[i]
  print(description_test.iloc[i])
  print('Predicted: ', val[0], 'Actual: ', labels_test.iloc[i], '\n')
  diff += abs(val[0] - labels_test.iloc[i])
```

    Here's another exceptional Oregon Riesling to add to the growing ranks of top producers. Lemon-drop fruit meets peaches and cream in the mouth, as this off-dry (20g/L) wine displays a spot-on balance between acid, sugar, fruit and honey. This is delicious already, and built to age nicely over a decade or longer.
    Predicted:  106.493416 Actual:  18.0 
    
    Made to fit a standard mold, but made well. Aromas are of plum and cinnamon, with similar notes on the palate. Slender in body, with smooth tannins and a nut-laden finish. Imported by Southern Starz, Inc.
    Predicted:  32.487404 Actual:  23.0 
    
    Young and tart in cool-climate acidity, this Pinot needs time in the cellar. It's an exotic wine, spicy and peppery, almost briary, like a Zinfandel, except with flavors of wild forest raspberries, cherries, orange zest and a hint of pine cone. It's as cellarable a Pinot Noir as exists in California. Best after 2015.
    Predicted:  26.840513 Actual:  60.0 
    
    Compact aromas of red berries get a boost from oak-based coconut, cedar and graphite notes. This is a blend of three Malbec vineyards of varying elevations; it's bursting with acidity, while high-toned plum and currant flavors are a touch salty. A lively, fiery finish is fueled by latent acidity. Drink through 2020.
    Predicted:  44.705788 Actual:  30.0 
    
    This powerful Pinot, from one of the three distinct blocks within Black Kite's estate vineyard, offers earthy plum compote and blueberry with a lingering background note of vanilla. Bright, grippy and layered, the finish offers toasty oak.
    Predicted:  47.285618 Actual:  55.0 
    
    This 2003 Riserva emphasizes oak-related aromas of black pepper, spice and cedar thanks to 36 months of French barrique. It has an inherently nervous quality, which will no doubt unwind with a few more years of cellar aging.
    Predicted:  63.148815 Actual:  60.0 
    
    A little on the sharp, green and minty side, with tart acids framing cherry, black raspberry and smoky wood flavors. A good wine that could use more lushness and richness.
    Predicted:  79.07275 Actual:  40.0 
    
    A Lambrusco-like petillant that's neutral and dry as a bone on the nose except for a pinch of green. The palate has a fine, controlled spritz and lemon-lime flavors. Crisp, citrusy and refreshing, and fairly well executed.
    Predicted:  -15.0146055 Actual:  12.0 
    
    A soft, off-dry wine, full of lightweight currant fruits. It is gentle, in a fruity, apéritif style.
    Predicted:  13.10156 Actual:  15.0 
    
    This blend of Nero d'Avola, Cabernet Sauvignon and Merlot offers aromas of dark fruit, espresso bean, leather, tobacco and savory spice. It's a bold, structured wine, with firm tannins and a bitter chocolate aftertaste.
    Predicted:  34.19325 Actual:  40.0 
    
    A fine, food-friendly, medium-sized Chard whose nose and finish both show a fair amount of wood. Fruit in the center is not very demonstrative, some peach skin holding down the fort. Imported by Foster's Wine Estates Americas.
    Predicted:  13.055821 Actual:  14.0 
    
    Nice and rich in toasty oak, framing mouthfilling flavors of pineapples, papaya, peach and cinnamon spice. Feels creamy-smooth, with a good tang of acidity.
    Predicted:  0.86690366 Actual:  15.0 
    
    A deep and dense wine, with coconut and blackberry dominating the weighty, powerful bouquet. This is an organic wine with an explosive palate of berry fruit and plenty of juicy acidity to keep it fresh and healthy. Toast, chocolate and cola are the finishing notes, and overall it's a strapping wine to drink now thru 2013. Imported by Organic Vintners.
    Predicted:  64.09779 Actual:  60.0 
    
    Easily the best Merlot from Livermore Valley in memory, which is not a back-handed compliment. It's a bone dry, tannic wine stuffed with complex, interesting fruit, currant, herb, tobacco and cedar flavors. Really very fine, and drinkable now despite the tannins. Its longterm future is controversial, though, so open over the next 3 years.
    Predicted:  42.810608 Actual:  80.0 
    
    A pretty good wine, with a firmly tannic, clean structure. Shows jammy or pie-filling cherry, blackberry and cocoa flavors, spiced with anise and pepper.
    Predicted:  37.360027 Actual:  32.0 
    
    The 80-acre Domaine de la Moussière is Alphonse Mellot's principal vineyard in Sancerre. This entry-level cuvée from the vineyard is already sumptuous enough, with fine, concentrated grassy flavors, laced with green apples, kiwi and citrus.
    Predicted:  58.277275 Actual:  39.0 
    
    With plenty of consumer-friendly vanilla/tobacco flavors, this value Cabernet also brings in dark fruits—blackberry and cassis. The tannins have been smoothed out a bit, though they still carry a little bit of a stemmy edge. A good everyday choice for that burger.
    Predicted:  12.539212 Actual:  11.0 
    
    There is an austere initial feel to this impressive wine, driven by its firm tannins and tight texture. It is dense and concentrated while also having black currant fruits that underly this intense structure. It certainly needs aging for several years.
    Predicted:  33.024666 Actual:  30.0 
    
    There are brambly, wild berry or crabapple notes here that are reinforced by the wine's naturally high acidity. This Barbera shows a pretty ruby color with dense thickness and a smooth, berry-filled finish.
    Predicted:  10.724262 Actual:  20.0 
    
    A powerful berry-based wine with no oak is what's on offer. The nose exudes rubber and leather at first, and then comes a wave of rambunctious black fruit. The palate is bright and packed, with ripe plum and cherry flavors. Shows good mouthfeel and chewy tannins. And once again: no oak!
    Predicted:  25.749783 Actual:  10.0 
    
    Fresh and easy, this approachable Chard exhibits bright aromas of fresh red apple, light citrus and dainty yellow flowers. The palate is slightly richer and fuller-bodied, with lush flavors of wood-grilled apple skins and toasted brioche that transition into the baking-spice laden finish.
    Predicted:  79.10085 Actual:  21.0 
    
    This is a nice, easy-to-like PG. It's crisp in acidity, with forward melon, lemon and lime, and date flavors, and while it's dry, it finishes with a honeyed sweetness. Try with avocado and crab salad or a rich baked ham.
    Predicted:  57.11621 Actual:  20.0 
    
    Tight and reductive, the nose of this Bordeaux blend shows tar, rubber and smoke. The palate follows with nicely concentrated black cherry and cassis, detailed with leather and tobacco. A far better effort than the winery's pricier Two Generations, this is a cellar-worthy Bordeaux blend that is just hinting at its long-term potential.
    Predicted:  58.343666 Actual:  50.0 
    
    Ponzi's mainstream Pinot Noir really brings the fruit in 2008, a wealth of blueberry and black raspberry flavors. From aging in 30% new French oak come streaks of clove and anise, with just a hint of moist loam. Tannins are light and ripe.
    Predicted:  18.303734 Actual:  35.0 
    
    Red berry fruit, anise seed, mint tea and dried flowers are attractive components to a bouquet with a strong fruit compote or jammy quality. Flavors include dark plum, vanilla and raisin and there's a nice smoothness to the tannins. Imported by Vias Imports.
    Predicted:  55.80226 Actual:  75.0 
    
    This wine is harsh in texture, with a green stem, puckery mouthfeel that frames unripe herb and mint flavors barely suggesting cherries.
    Predicted:  13.830893 Actual:  18.0 
    
    This reserve-level wine, from the estate vineyard, is certified biodynamic. Smooth and sweetly spicy, it compounds raspberry pastry with hints of crumb cake and brown sugar. The finish seems to hit a bit of a wall, and resonates with a touch of metal.
    Predicted:  26.914127 Actual:  55.0 
    
    A clumsy wine. It's semisweet in vanilla and lemon yogurt flavors, and soft, with a funky, soiled finish.
    Predicted:  -5.989372 Actual:  30.0 
    
    A nice blush wine. Made from 100% Pinot Noir, it's dryish and crisp, with pretty flavors of cherries, raspberries and vanilla. Nice with chicken pot pie or chicken enchiladas.
    Predicted:  20.596931 Actual:  18.0 
    
    This wine overdelivers for its pale garnet color. There's loads of sweet cherry fruit along with mocha and caramel character, all set in a bed of soft tannins. A long finish trails off to an extended earth note. Try with leg of lamb. Fully mature, drink up in the next year or two.
    Predicted:  36.54485 Actual:  25.0 
    
    This herbaceous wine has a steely edge to its citrus and apple flavors. Exuberant and fragrant, this has a crisp texture that jumps out of the glass.
    Predicted:  38.50559 Actual:  14.0 
    
    Upon opening, this wine showed some funky low tones that blew off after a while to reveal an unusual mix of aromas, including cotton candy, stewed prunes and fresh nuts. The light-to-medium-bodied palate is more classic, with razor-like acidity and raspberry and cherry flavors. Delicious cherry-berry finish. Try with beef stew. Drink now.
    Predicted:  0.50154936 Actual:  18.0 
    
    This is just the second vintage for this winery, whose inviting labels look like handwritten notes. The rosé captures the bright color, fresh fruit and silky texture of its Pinot Noir grape; it's a perfect salmon or turkey wine, and one you could happily sip as an apéritif.
    Predicted:  4.611366 Actual:  24.0 
    
    A very good effort with this tricky grape. Along with the stone-fruit flavors are sweeter streaks of cotton candy and marshmallow. But the candied fruit is lively and not cloying or artificial. The finish is long and clean, with a twist of buttered nuts and creamy vanilla.
    Predicted:  7.049846 Actual:  18.0 
    
    A wonderfully aromatic wine that's bone dry, fresh and textured. A steely edge gives a mineral, tight character that emphasizes the youth of the wine. With a dominance of orange and lemon zest, plus juicy acidity, the wine does need a year or two; drink after 2016.
    Predicted:  23.66328 Actual:  22.0 
    
    If you drink this wine too cold, you'll miss the nuances behind the pineapple, orange and spice flavors. There's a minerally undertow, and some interesting herbal notes.
    Predicted:  14.213644 Actual:  16.0 
    
    A very good Merlot except for some sharpness. The wine is dry and finely tannic, with black and red cherry, mocha, dried herb and tobacco flavors.
    Predicted:  33.95843 Actual:  22.0 
    
    If you want a more earthy style of red Burgundy, this is the wine for you. It has a definite funky edge, lying against the ripe tannins and sweet soft, jammy fruit. The acidity comes through to finish a wine with weight and character.
    Predicted:  17.046572 Actual:  50.0 
    
    Initially, this is a fruit-dominated wine, with a cherries-in-syrup flavor. Only gradually do the tannins and the stalky but rich texture come in to suggest aging potential. The wine finishes with the fresh edge typical of 2007.
    Predicted:  33.00124 Actual:  61.0 
    
    In the heat of the 2008 vintage, it may have been a mistake to further dry 20% of the grapes for this wine. The result has turned out chocolaty-ripe, but also with a touch of raisining evident, and while there's volume, there's also a certain lack of midpalate richness.
    Predicted:  35.908936 Actual:  23.0 
    



```python

```
