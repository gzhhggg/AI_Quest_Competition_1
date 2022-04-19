# AI_Quest_Competition_1

タイトル：「PBL_02 不良個所自動検出による検品作業効率化（製造業）AI課題」  

ストーリー  
ABC基板株式会社では基板の不良個所の検品作業に工数がかかっている  
データセットを用いて基盤の合否判定が出来るAIモデルを構築・分析して結果を提出する。

課題内容    
<img src="https://user-images.githubusercontent.com/93046615/163923287-c95cd307-7cf3-47b2-ae1a-30d89eb6b6ad.png" width="900px">  

# ファイル名    

### 前処理    
1. Random_prepro.ipynb  
2. Trim_Mask_prepro.ipynb  
3. Erasing_prepro.ipynb  
4. split_img.ipynb  

### 学習  
1. learning.ipynb

### 画像
1. Original_Images
2. Random_Images
3. Trim_Images
4. Mask_Images
5. Erasing_Images

# 前処理について  

### 1.ImageDataGeneratorで学習画像を増やす  
ファイル名：Random_prepro.ipynb  


```bash
from keras.preprocessing.image import load_img,img_to_array,ImageDataGenerator
```


CNN の学習を行う場合にオーグメンテーション (augmentation) を行い、学習データのバリエーションを増やすことで精度向上を狙う  
<img src="https://user-images.githubusercontent.com/93046615/163938862-6f3f238e-4283-4364-b134-31494acf1a9d.jpg" width="200px"><img src="https://user-images.githubusercontent.com/93046615/163938863-b8ca4002-9804-4a43-bf6b-b42eb3d71dff.jpg" width="200px"><img src="https://user-images.githubusercontent.com/93046615/163938865-96addad0-22cc-4530-a1cb-3f0df214b362.jpg" width="200px"><img src="https://user-images.githubusercontent.com/93046615/163938867-9d29cd6c-9f82-4f39-9e9e-8a38caf8254f.jpg" width="200px"><img src="https://user-images.githubusercontent.com/93046615/163939298-8f0c8cb9-4b07-4ad3-9e32-b20a8ca2dbba.jpg" width="200px">

結果  
オリジナル画像のデータ数が少なかったため、効果は大きかったが枚数を増やしすぎると過学習の原因に繋がることが分かった。  
<br>
<br>
<br>
<br>
### 2.基盤の外枠をトリミングする処理  
ファイル名：Trim_Mask_prepro.ipynb  

特徴量の強調を行うために画像をトリミングをする  
基板の色の下限・上限をHSVで指定して、2値化してfindContoursを使いオブジェクト輪郭検出する  
検出後　Coordinateを使って座標のx,y座標の最大値最小値を取得する
```bash
def Trim_precessing(path):
    img = cv2.imread(path)
    hsvLower = np.array([30, 80, 0])    # 抽出する色の下限(HSV)
    hsvUpper = np.array([90, 255, 255])    # 抽出する色の上限(HSV)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) # 画像をHSVに変換
    hsv_mask = cv2.inRange(hsv, hsvLower, hsvUpper)    # HSVからマスクを作成
    contours = cv2.findContours(hsv_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
    x1_min,y1_min,x2_max,y2_max = Coordinate(contours)
```
<img src="https://user-images.githubusercontent.com/93046615/163941009-84f35b40-1458-48d9-9186-181fbd87a31a.png" width="200px"><img src="https://user-images.githubusercontent.com/93046615/163941026-2d943e23-75cd-43ed-b01c-a86b724ee7c4.png" width="200px"><img src="https://user-images.githubusercontent.com/93046615/163941044-5bfe3c3b-e78a-41d1-bffa-ad928a3a2b12.png" width="200px"><img src="https://user-images.githubusercontent.com/93046615/163941291-25001656-f1d2-4347-8825-a9b23030a557.jpeg" width="200px">

結果  
特徴の強調はかなり効果が大きく制度が向上した  
<br>
<br>
<br>
<br>

### 3.基盤のはんだ部分をトリミングする処理  
ファイル名：Trim_Mask_prepro.ipynb  

特徴量の強調を行うためにはんだ部分のみトリミングする  
OpenCVでゴミケシや白膨張を繰り返し行いはんだ以外の場所を黒塗りする  
```bash
    #Opening ごみ消し
    kernel = np.ones((10,10),np.uint8)
    opening = cv2.morphologyEx(hsv_mask, cv2.MORPH_OPEN, kernel)
    #白を膨張させる
    kernel = np.ones((80,80),np.uint8)
    dilation = cv2.dilate(img3,kernel,iterations = 1)
```
<img src="https://user-images.githubusercontent.com/93046615/163942669-d7923ee3-6b5a-40dd-b136-35cac376f9e2.png" width="200px"><img src="https://user-images.githubusercontent.com/93046615/163942677-aedf9f39-aeb3-4aba-a6d4-0bd4f9922b20.png" width="200px"><img src="https://user-images.githubusercontent.com/93046615/163942687-1684d083-a02b-41d2-b4a8-cd27232067f7.png" width="200px"><img src="https://user-images.githubusercontent.com/93046615/163942697-05c67b07-0724-4e4b-a507-6378001d0ea8.png" width="200px"><img src="https://user-images.githubusercontent.com/93046615/163943219-cf704fc8-743e-4a55-9c68-30ebcc064adc.jpeg" width="200px">

結果  
トライ＆エラーを繰り返して一番時間をかけた前処理だが精度は向上しなかった  
はんだ部分のみ完全にトリミングは難しい  
<br>
<br>
<br>
<br>
### 4.ランダムに短形を重ねて学習画像を増やす
ファイル名：Erasing_prepro.ipynb

ランダムに短形を作成して元画像に重ねていく
短形はべた塗でも良かったが、モザイク調のほうが精度が高かった  
```bash
def random_erasing(file_path, p=0.5, sl=0.02, sh=0.4, r1=0.3, r2=3.3):
    img = cv2.imread(file_path)
    target_img = img.copy()

    if p < np.random.rand():
        # RandomErasingを実行しない
        return target_img 

    H, W, _ = target_img.shape
    S = H * W

    while True:
        Se = np.random.uniform(sl, sh) * S # 画像に重畳する矩形の面積
        re = np.random.uniform(r1, r2) # 画像に重畳する矩形のアスペクト比

        He = int(np.sqrt(Se * re)) # 画像に重畳する矩形のHeight
        We = int(np.sqrt(Se / re)) # 画像に重畳する矩形のWidth

        xe = np.random.randint(0, W) # 画像に重畳する矩形のx座標
        ye = np.random.randint(0, H) # 画像に重畳する矩形のy座標

        if xe + We <= W and ye + He <= H:
            # 画像に重畳する矩形が画像からはみ出していなければbreak
            break

    mask = np.random.randint(0, 255, (He, We, 3)) # 矩形がを生成 矩形内の値はランダム値
    target_img[ye:ye + He, xe:xe + We, :] = mask # 画像に矩形を重畳
    cv2.imwrite(file_path,target_img)
```
<img src="https://user-images.githubusercontent.com/93046615/163944891-5b6170bf-0e61-470c-adfc-4c5299db01f6.jpg" width="200px"><img src="https://user-images.githubusercontent.com/93046615/163944895-9ad3489a-0e27-47bc-9993-063c83566df5.jpg" width="200px"><img src="https://user-images.githubusercontent.com/93046615/163944899-477d90e6-c838-4e5f-acda-07a364c20e4b.jpg" width="200px"><img src="https://user-images.githubusercontent.com/93046615/163944904-7c69ea88-c6cb-47a8-8238-0986526d44f5.jpg" width="200px"><img src="https://user-images.githubusercontent.com/93046615/163944910-24ce402d-e258-443e-8d27-ff86642cd02e.jpg" width="200px">

結果  
一見効果がなさそうに思えたが、かなり精度が向上した  
精度が上がったメカニズムがいまいち理解できないのでモヤモヤする。。。
<br>
<br>
1.2.4の手法を掛け合わせて前処理を行った


# 学習モデルについて
ファイル名：learning.ipynb

VGG16・VGG19・Xceptionで精度がVGG16が高かったため採用した。  
コンペ優勝者はアンサンブル学習を採用していた。
パラメータの設定で入力画像サイズの高さと幅をもっと大きくしていればより精度が高くなることが分かった。  
エポック数を多くしすぎると過学習の原因になってしまう。
```bash
IMG_WIDTH, IMG_HEIGHT = 224, 224
TARGET_SIZE = (IMG_WIDTH, IMG_HEIGHT)
# 判定分類数（4分類を判定するモデルを構築し、そのモデルの判別結果を最後に良品、不良品の2分類に変換する前提）
NB_CLASSES = 4
# 学習時のエポック数
# ※エポック数は多すぎると過学習の原因になる
EPOCHS = 30
# バッチサイズ
BATCH_SIZE = 5
```
エポック数30の場合、TrainとValidでロス値が大きくなることが分かる  
グラフからエポック数5～10くらいが適正と判断した。  
<img src="https://user-images.githubusercontent.com/93046615/163947891-e7de68ef-8a5a-4777-ad27-5fb9d2d80291.png" width="500px"><img src="https://user-images.githubusercontent.com/93046615/163948604-9679ecf9-27af-4a3d-8bae-ae472284a789.png" width="500px">  

作成したモデルでテスト画像を分類する  
```bash
# テストデータに対して1つずつ予測し、テストデータのファイル名と判定結果をリストに保存
file_list = []
pred_list = []
for file in glob.glob(test_data_dir + '/*'):
    image_data = file
    filename = file.split('/')[-1]
    img = image.load_img(image_data, target_size=(IMG_WIDTH, IMG_HEIGHT))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255
    pred = model.predict(x)[0]
    judge = np.argmax(pred)

    # *bridge, horn, potatoを不良（'1'）に、regularを良品（'0'）に変換。if文の条件分岐は上の「分類とラベルの対応確認」セルの結果を参考に変更すること*
    if judge==0:
        judge=1
    elif judge==1:
        judge=1
    elif judge==2:
        judge=1
    else:
        judge=0

    pred_list.append(judge)
    file_list.append(filename)
```

# Note  
tensorflowはCPUで処理するため学習に時間がかかる、画像枚数が増えれば増えるほどとてつもなく時間がかかる！！  
tensorflow-gpuで学習させたら処理時間は1/1000になり衝撃的だった  
朝起きて学習開始して仕事終わって帰ってきても終わらない状況だったのにGPUで処理することで1時間もかからず処理が終わるようになった    
特にコンペの場合、締切直前になればなるほど時間が惜しくなる  
もっと早くに知ることが出来ていたら、、、と悔いが残る  
# Author

* 作成者 KeiichiAdachi
* 所属 Japan/Aichi
* E-mail keiichimonoo@gmail.com
 
# License
なし  




