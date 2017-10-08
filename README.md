# 敵対的生成ネットワーク（GAN）による手書きカタカナ文字の生成（Kerasでの実装）

## 

## 手書きカタカナ文字のデータについて
手書きのカタカナ文字の画像データはETLデータベースに公開されているデータを利用している。このデータは、OCR機器で収集された、「1,383人の筆記者・157,662の合計文字数」の手書き文字の画像データとなっている。

114文字の中から、カタカナの46文字のみを利用している。

### 詳細（[ETL6ページにより](http://etlcdb.db.aist.go.jp/etlcdb/etln/etl6/etl6.htm)）
```
ＯＣＲシート仕様
    文字枠      : 横 5mm、縦 6mm
    文字枠ピッチ : 横 6.35mm、縦 12.7mm
    文字枠数    : 26 x 17 = 442

対象文字 (計 114文字)
    数字     : 10
    英大文字 : 26
    カタカナ : 46
    特殊文字 : 32

ＯＣＲシート収集
    筆記者数     : 1,383人
    全サンプル数 : 157,662

観測装置
    濃度レベル : 16 (4bit)
    標本点数   : 64 x 63 = 4,032 pixels

データベース作成
    観測期間   : 1976年12月～1977年5月
```

### 前処理
* 各画像を28x28にリサイズ（MNISTと同じサイズ）
* ピクセル値を[0, 255]から[-1, 1]に正規化

### 実画像のサンプル
![real_images_sample](illustrations/real_images_sample.png "Real images sample")

## GANについて
敵対的生成ネットワーク([Generative Adversarial Networks (GAN), Goodfellow et al. 2014](https://arxiv.org/abs/1406.2661))とは、偽造画像を生成するGeneratorネットワーク（G）と、実画像と偽造画像を判別するDiscriminatorネットワーク（D）を戦わせる生成モデルの学習パラダイムである。GはDをだませるような偽造画像を生成できるように学習し、Dは実画像と偽造画像をできるだけ判別できるように学習します。

本リポジトリでは、手書きのカタカナ文字の画像を生成するために、畳み込み敵対的生成ネットワーク（[DCGAN, Radford et al. 2015](https://arxiv.org/abs/1511.06434)）の１種を実装している。

### GAN構造のあれこれ
* [How to Train a GAN? Tips and tricks to make GANs work](https://github.com/soumith/ganhacks)による下記のアドバイスを反映している：
  * スパースな勾配を避けるために、Fully convolutionalネットワーク構造とLeakyReLUを採用
  * 画像を[-1, 1]に正規化し、tanhの活性化関数で生成
  * 実画像と偽造画像を別々のバッチでDiscriminatorを学習させる
  * **z**は、Uniform分布ではなく、正規分布から生成
* hyperparameterチューニングのために参考になったリンク
  * [GAN by Example using Keras on Tensorflow Backend](https://medium.com/towards-data-science/gan-by-example-using-keras-on-tensorflow-backend-1a6d515a60d0)
  * [MNIST Generative Adversarial Model in Keras](https://oshearesearch.com/index.php/2016/07/01/mnist-generative-adversarial-model-in-keras/)

## 使い方
notebooks/main_ETL6_katakana.ipynbに使い方の例を示している。

### DCGANの定義
指定可能なDCGAN hyperparameterに関してはkerasdcgan/models.pyをご参照ください。
```python
from kerasdcgan.models import DCGAN

dcgan = DCGAN()
dcgan.build()
dcgan.summary()
```

### バッチで学習（Dを１ステップ、Gを１ステップ学習する）
```python
# x_train_batch is a batch of real images (numpy.ndarray of shape (samples, height, width, 1))
d_metrics, stacked_metrics = dcgan.train_on_batch(x_train_batch, freeze_discriminator=True)
```

### **z**の生成
```python
# Generate 25 noise samples
noise_samples = dcgan.generate_noise(25)
```

### 偽造画像の生成
```python
noise_samples = 25
# noise_samples = dcgan.generate_noise(25)  # This also works
generated_images = dcgan.generate(noise_samples)
```

### 画像データの作成
```python
from itertools import product
from kerasdcgan.etl import read_etl6_data, data2array

data = read_etl6_data('/path/to/etl6_files')
x_all, y_all = data2array(data, new_shape=(28, 28))

# Get katakana images
katakana = [''.join(e) for e in product(' KSTNHMYRW', 'AIUEO')] + [' N']
katakana_idx = [i for i, label in enumerate(y_all) if label in katakana]
x_train = x_all[katakana_idx]
```

## 結果サンプル
### 偽造画像のサンプル
![fake_images_026500](illustrations/fake_images_026500.png "Fake images sample")

### 偽造画像の進化のアニメーション
![100_256_512_30000](illustrations/100_256_512_30000.gif "Fake images animation")

## Repository info
### Requirements
* tensorflow-gpu==1.2.1
* Keras==2.0.8
* numpy==1.13.3
* Pillow==4.3.0
* pandas==0.20.3

### データ
* [ETL文字データベース](http://etlcdb.db.aist.go.jp/?lang=ja) (ETL6を利用)
