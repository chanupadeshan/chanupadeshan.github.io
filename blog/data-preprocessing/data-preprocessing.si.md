---
title: "Data Preprocessing: මොඩලයකට ගැළපෙන ලෙස දත්ත සකස් කිරීම"
description: "දත්ත පෙර සැකසුම යනු කුමක්ද, එය වැදගත් ඇයි, සහ වගු, පෙළ, රූප හා කාල ශ්‍රේණි සඳහා Python උදාහරණ."
author: "Chanupa Deshan"
language: si
date: 2026-09-05
---

# Data Preprocessing: මොඩලයකට ගැළපෙන ලෙස දත්ත සකස් කිරීම

මොඩලයක් ගනුදෙනුකරුවෙකු, වාක්‍යයක් හෝ ඡායාරූපයක් දකින්නේ අප දකින ආකාරයට නොවෙයි. එයට ලැබෙන්නේ ඒ තොරතුරු නිරූපණය කරන ආදානයක්. අමු දත්ත මොඩලයට භාවිත කළ හැකි, එකම රීතිවලට අනුව සකස් කළ ආදාන බවට පත් කිරීම දත්ත පෙර සැකසුම (Data Preprocessing) ලෙස හැඳින්වෙනවා.

[පෙර ලිපියේදී](../supervisedVSunsupervised/supervised-vs-unsupervised-learning.si.html) අපි Supervised සහ Unsupervised Learning සැසඳුවා. ඒ දෙකටම ආදානවල ගුණාත්මකභාවය වැදගත්. මේ ලිපියේදී දත්ත වර්ග කිහිපයක් සකස් කරන ආකාරය Python කේත සමඟ බලමු.

## Preprocessing යනු කුමක්ද? එය වැදගත් ඇයි?

දත්ත වර්ග පරීක්ෂා කිරීම, හඳුනාගත් දෝෂ නිවැරදි කිරීම, නොමැති අගයන් හැසිරවීම සහ ආදාන සුදුසු නිරූපණයකට මාරු කිරීම preprocessing තුළට අයත් වෙනවා. Feature Engineering මීට සම්බන්ධ ක්‍රියාවලියක්. එහිදී ගනුදෙනුවක පැය හෝ ඊයේ උෂ්ණත්වය වැනි ප්‍රයෝජනවත් ලක්ෂණ නිර්මාණය කරනවා.

මිල තීරුවක `1200`, `"1,200"`, හිස් කොටුවක් සහ වෙනත් මුදල් ඒකකයක අගයක් තිබෙනවා යැයි සිතන්න. ඒ සියල්ල සංඛ්‍යා බවට හැරවීම පමණක් ප්‍රමාණවත් නැහැ. ඒකක, හිස් කොටුවක අර්ථය සහ වලංගු අගයන් මොනවාද යන්නත් තේරුම් ගත යුතුයි.

හොඳ preprocessing ක්‍රියාවලියක් ආදානවල අනුකූලතාව රකිනවා, අදාළ මොඩලවලට කාර්යක්ෂමව ඉගෙනීමට උපකාරී වෙනවා, සහ පුහුණුවේදීත් සැබෑ භාවිතයේදීත් එකම රීති යොදා ගැනීමට ඉඩ දෙනවා. දත්ත රැස් කිරීමේ දෝෂ කලින් හඳුනා ගැනීමටත් එය උපකාරීයි. නමුත් වැරදි ලේබල, දත්තවල නොමැති ජන කණ්ඩායම් හෝ පැහැදිලි නැති ගැටලුවක් එයින් පමණක් විසඳෙන්නේ නැහැ.

| දත්ත වර්ගය | සාමාන්‍ය සැකසුම් | වළක්වා ගත යුතු වැරැද්ද |
| --- | --- | --- |
| වගු | හිස් අගයන් පිරවීම, encoding, scaling | Test දත්ත මත සංඛ්‍යාන ඉගෙනීම |
| පෙළ | Unicode සැකසුම, tokenization, vectorization | අධික පිරිසිදු කිරීමෙන් අර්ථය නැති කිරීම |
| රූප | දිශානතිය, වර්ණ ආකාරය, ප්‍රමාණය, pixel පරිවර්තනය | පෙර පුහුණු මොඩලයට නොගැළපෙන සැකසුම් |
| කාල ශ්‍රේණි | වේලාවන් ගැළපීම, lag ලක්ෂණ, කාලය අනුව බෙදීම | අනාගත තොරතුරු වර්තමාන ලක්ෂණවලට යොදා ගැනීම |
| හඬ සහ වීඩියෝ | Sample/frame rates, සමමුහුර්තකරණය, ලක්ෂණ උකහා ගැනීම | එකම පටිගත කිරීමේ කොටස් train සහ test දෙකටම ඇතුළත් කිරීම |

## විශ්වාසදායක ඇගයීමක් සඳහා ක්‍රියාවලියක්

මුලින් දත්ත ව්‍යුහය පරීක්ෂා කර පුරෝකථනය කළ යුතු කාර්යය නිර්වචනය කරන්න. පුරෝකථනය කරන මොහොතේ ඇත්තටම තිබෙන තොරතුරු තීරණය කරන්න. හඳුනාගත් අනුපිටපත් නිවැරදි කර, එකම පුද්ගලයා, ලේඛනය හෝ පටිගත කිරීම දත්ත කොටස් අතර කාන්දු නොවන ලෙස සම්බන්ධ උදාහරණ කණ්ඩායම් කරන්න.

ඊළඟට කාර්යයට ගැළපෙන training, validation සහ test කොටස් සාදන්න. ස්වාධීන උදාහරණ සඳහා අහඹු බෙදීමක් ගැළපිය හැකියි. සම්බන්ධ වාර්තා හෝ කාලයට අනුව ඇති දත්ත සඳහා කණ්ඩායම් හෝ කාලය පදනම් කරගත් බෙදීම් අවශ්‍ය වෙනවා.

**දත්තවලින් ඉගෙනගන්නා preprocessing පරාමිතීන් training දත්ත මත පමණක් fit කරන්න.** මධ්‍යස්ථ අගයන්, සාමාන්‍ය අගයන්, කාණ්ඩ ලැයිස්තු සහ ලක්ෂණ තේරීමේ රීති දත්තවලින් ඉගෙනගන්නවා. Validation, test සහ සැබෑ භාවිතයේදී නැවත භාවිත කළ යුත්තේ ඒ fitted transformations මයි. Cross-validation හි සෑම training fold එකක් තුළම මේවා නැවත fit කළ යුතුයි. Pipeline එකක් මඟින් සැකසුම් සහ මොඩලය එකට තබාගත හැකියි. [scikit-learn හි data leakage මාර්ගෝපදේශය](https://scikit-learn.org/stable/common_pitfalls.html) බලන්න.

උදාහරණ සඳහා Python සහ කුඩා ඉගෙනුම් දත්ත කට්ටල භාවිත කරනවා. [requirements.txt](requirements.txt) බාගත කර virtual environment එකක packages ස්ථාපනය කරන්න:

```bash
python -m venv .venv
# Linux / macOS:
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

අදාළ Python ගොනුව බාගත කර, උදාහරණයක් ලෙස `python tabular.py` ක්‍රියාත්මක කරන්න. මේවා සැකසුම් පෙන්වන උදාහරණයි; සැබෑ මොඩලයක ගුණාත්මකභාවය මනින පරීක්ෂණ නොවෙයි.

## වගු දත්ත: හිස් අගයන්, කාණ්ඩ සහ පරිමාණය

දත්ත වර්ග, ඒකක, අගය පරාස සහ නැවත නැවත ඇති හඳුනාගැනීමේ අංක පරීක්ෂා කරන්න. Outliers මකා දැමීමට පෙර විමර්ශනය කරන්න. විශාල ගනුදෙනුවක් ඔබට හඳුනා ගැනීමට අවශ්‍ය සිදුවීමම විය හැකියි. හිස් අගයක් සඳහා මධ්‍යස්ථ අගය, “unknown” කාණ්ඩයක්, අගයක් නොමැති බව දක්වන ලක්ෂණයක් හෝ එවැනි දත්ත සෘජුව හසුරුවන මොඩලයක් භාවිත කළ හැකියි. තේරීම කළ යුත්තේ හිස් වීමේ අර්ථය අනුවයි.

බොහෝ රේඛීය සහ දුර මැනීම් භාවිත කරන මොඩලවලට scaling වැදගත්. Standardization හිදී training සාමාන්‍යය අඩු කර training සම්මත අපගමනයෙන් බෙදනවා. Min–max scaling මඟින් training අගය පරාසය තෝරාගත් පරාසයකට මාරු කරනවා. Robust scaling මධ්‍යස්ථ අගය සහ අන්තර් චතුර්ථක පරාසය වැනි සංඛ්‍යාන භාවිත කරනවා. Tree-based මොඩලවලට සාමාන්‍යයෙන් එවැනි scaling අවශ්‍ය නැහැ. මේවා විකල්ප ක්‍රමයි; සියල්ල එකට යෙදිය යුතු නැහැ. [scikit-learn preprocessing මාර්ගෝපදේශය](https://scikit-learn.org/stable/modules/preprocessing.html) බලන්න.

One-hot encoding මඟින් කාණ්ඩවලට කෘත්‍රිම සංඛ්‍යාත්මක අනුපිළිවෙළක් නොදී වෙනම දර්ශක තීරු සාදනවා. Ordinal encoding යොදා ගන්නේ අනුපිළිවෙළට සැබෑ අර්ථයක් තිබෙන විටයි. ඉතා වැඩි කාණ්ඩ ගණනක් තිබේ නම් වෙනත් නිරූපණ අවශ්‍ය විය හැකියි. Target encoding හිදී leakage වළක්වා ගැනීමට විශේෂ සැලකිල්ලක් අවශ්‍යයි.

පහත උදාහරණය සංඛ්‍යාත්මක තීරුවල හිස් අගයන් පුරවා scale කරනවා, නගර encode කරනවා, සහ classifier එකක් පුහුණු කරනවා. අලුත් නගරයක් ලැබුණත් pipeline එක දෝෂයක් නොදී ක්‍රියා කරනවා. නමුත් නොදන්නා කාණ්ඩයක් නොසලකා හැරීම හොඳ පුරෝකථනයක් සහතික කරන්නේ නැහැ. මෙය [ColumnTransformer සහ Pipeline](https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html) භාවිත කරන ආකාරයක්.

[tabular.py බාගත කරන්න](examples/tabular.py)

```python
"""Small teaching example; predictions are not a quality benchmark."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

X = pd.DataFrame({
    "age": [22, 45, np.nan, 31, 52, 27, 39, 24, 48, 35, 29, 56],
    "income": [30, 80, 45, np.nan, 95, 35, 65, 28, 88, 55, 40, 100],
    "city": ["Kandy", "Colombo", "Galle"] * 4,
})
y = [0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

numeric = Pipeline([
    ("fill", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
])
categorical = Pipeline([
    ("fill", SimpleImputer(strategy="most_frequent")),
    ("encode", OneHotEncoder(handle_unknown="ignore")),
])
prepare = ColumnTransformer([
    ("numeric", numeric, ["age", "income"]),
    ("category", categorical, ["city"]),
])
model = Pipeline([
    ("prepare", prepare),
    ("predict", LogisticRegression(max_iter=1000)),
])
model.fit(X_train, y_train)  # Learn every statistic on training data only.
print("Test predictions:", model.predict(X_test))
new_customer = pd.DataFrame({
    "age": [33], "income": [50], "city": ["Jaffna"]
})
print("New category prediction:", model.predict(new_customer))
```

## පෙළ දත්ත: සංඛ්‍යා සෑදීමට පෙර අර්ථය රකින්න

පෙළ සැකසුම කාර්යය අනුව වෙනස් වෙනවා. නිවැරදිව decode කිරීම, අනවශ්‍ය markup ඉවත් කිරීම, Unicode normalization සහ tokenization ප්‍රචලිත පියවරයි. කුඩා අකුරු බවට හැරවීම, stop words ඉවත් කිරීම, stemming සහ lemmatization අනිවාර්ය නැහැ. “not” ඉවත් කිරීමෙන් හැඟීමේ අර්ථය පෙරළිය හැකියි. විරාම ලකුණු මකා දැමීමෙන් වැදගත් සංඥා නැති විය හැකියි. ඉංග්‍රීසි නොවන සියලු අක්ෂර ඉවත් කළොත් සිංහල පෙළ විනාශ වෙනවා.

සාම්ප්‍රදායික මොඩල සඳහා වචන ගණන් හෝ TF-IDF ලක්ෂණ භාවිත කළ හැකියි. මෙහි character n-grams මඟින් ඉංග්‍රීසි සහ සිංහල සඳහා සරල ආරම්භයක් ලබා ගන්නවා. ඉංග්‍රීසි word tokenizer එකක් භාෂා දෙකටම ගැළපේ යැයි උපකල්පනය කරන්නේ නැහැ. TF-IDF පුහුණු පෙළෙන් vocabulary එක සහ inverse-document-frequency බර ඉගෙනගන්නවා. Sparse matrix එකක් බොහෝ දුරට ශූන්‍ය නොවන අගයන් කාර්යක්ෂමව ගබඩා කරනවා. විකල්ප සඳහා [TfidfVectorizer ලේඛනය](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) බලන්න.

[text.py බාගත කරන්න](examples/text.py)

```python
import unicodedata
from sklearn.feature_extraction.text import TfidfVectorizer


def clean_text(text):
    # Keep Sinhala characters, punctuation, and words such as "not".
    text = unicodedata.normalize("NFC", text)
    return " ".join(text.split())


train_texts = [
    "This service is good!", "This service is not good.",
    "මෙම සේවාව හොඳයි", "මෙම සේවාව හොඳ නැහැ",
]
test_texts = ["  The service\n is good. ", "සේවාව හොඳයි"]
vectorizer = TfidfVectorizer(
    analyzer="char_wb", ngram_range=(3, 5), lowercase=True
)
X_train = vectorizer.fit_transform([clean_text(t) for t in train_texts])
X_test = vectorizer.transform([clean_text(t) for t in test_texts])
print("Training shape:", X_train.shape)
print("Test shape:", X_test.shape)
# Both matrices use the vocabulary and IDF learned from train_texts.
```

Test matrix එකේ ලක්ෂණ තීරු ගණන training matrix එකට සමානයි. Test පණිවිඩවලින් අලුත් vocabulary එකක් ඉගෙනගන්නේ නැහැ. Character n-grams සම්පූර්ණ භාෂාමය tokenizer එකක් නොවෙයි. Transformer මොඩල සඳහා අදාළ checkpoint එකේ tokenizer එක සහ එහි padding, truncation හා attention-mask රීති භාවිත කරන්න. ඒ tokenizer එකට අවශ්‍ය තොරතුරු පිරිසිදු කිරීමේදී ඉවත් නොකරන්න.

## රූප: ප්‍රමාණය, වර්ණය සහ pixel අගයන්

මුලින් රූප විවෘත කළ හැකිද සහ ලේබල නිවැරදිදැයි බලන්න. කැමරාවේ දිශානතිය නිවැරදි කර, එකම වර්ණ ආකාරයක් තෝරා, resize, crop හෝ pad කළ යුතුදැයි තීරණය කරන්න. ඇද විශාල කිරීමෙන් හැඩය වෙනස් වෙනවා. Crop කිරීමෙන් ප්‍රධාන වස්තුව කැපී යා හැකියි. Padding මඟින් මායිම් එකතු වෙනවා. සුදුසු තේරීම කාර්යය මත රඳා පවතිනවා.

උදාහරණය දිග-පළල අනුපාතය රකිමින් රූපය වර්ගයකට pad කරනවා, RGB බවට පත් කරනවා, සහ 8-bit රූපය `[0, 1]` පරාසයේ float array එකක් බවට හරවනවා. ප්‍රතිදානයේ අනුපිළිවෙළ උස × පළල × channels. සමහර මොඩලවලට channels මුලින් තිබීම හෝ අමතර normalization අවශ්‍යයි. පෙර පුහුණු checkpoint එකක නියමිත සැකසුම් අනුගමනය කරන්න. [Pillow image operations](https://pillow.readthedocs.io/en/stable/reference/ImageOps.html) සහ [Torchvision transforms](https://docs.pytorch.org/vision/stable/transforms.html) බලන්න.

[image.py බාගත කරන්න](examples/image.py)

```python
import numpy as np
from PIL import Image, ImageOps


def prepare_image(image):
    image = ImageOps.exif_transpose(image)  # Respect camera orientation.
    image = image.convert("RGB")
    image = ImageOps.pad(
        image, (224, 224), method=Image.Resampling.BILINEAR,
        color=(0, 0, 0),
    )  # Preserve aspect ratio and add padding.
    return np.asarray(image, dtype=np.float32) / 255.0


# A synthetic input keeps the example runnable without downloading a photo.
example = Image.new("RGB", (320, 180), color=(80, 160, 240))
pixels = prepare_image(example)
print("Shape:", pixels.shape)  # (224, 224, 3): height, width, channels
print("Type:", pixels.dtype)  # float32
print("Range:", pixels.min(), pixels.max())  # Within [0, 1]
# For your photo:
# with Image.open("photo.jpg") as photo:
#     pixels = prepare_image(photo)
```

මෙය සාමාන්‍ය 8-bit RGB රූප සඳහායි. වෛද්‍ය ස්කෑන් හෝ වැඩි bit depth සහිත sensor දත්ත සඳහා සෑමවිටම ගැළපෙන්නේ නැහැ. 255 න් බෙදීම pixel scaling පියවරක්. Channel අනුව mean/std normalization වෙනම පියවරක්.

Training augmentation මඟින් යථාර්ථයට ගැළපෙන crops, දීප්තියේ වෙනස්කම් හෝ flips එකතු කරනවා. පැහැදිලිව නිර්වචනය කළ test-time augmentation ක්‍රමයක් ඇගයීමට භාවිත කරන අවස්ථාවක් හැර, validation සහ test සැකසුම් අහඹු නොවන ලෙස තබන්න. බළලෙකුගේ රූපයක් හරස් අතට හැරවීමෙන් ලේබලය වෙනස් නොවුණත්, ලියූ පෙළක් එසේ හැරවීමෙන් එය විකෘති වෙනවා. Segmentation සහ detection සඳහා masks හෝ boxes ද රූපයට ගැළපෙන ලෙස වෙනස් කළ යුතුයි.

## කාල ශ්‍රේණි: අතීතයෙන් ඉගෙන ගන්න

Timestamps කියවා, time zones එකම ආකාරයකට සකස් කර, දත්ත කාලය අනුව පෙළගස්වන්න. හිස් කාල පරාස සහ අසමාන නියැදි පරතර පරීක්ෂා කරන්න. Resampling මඟින් කියවීම් එකම කාල පරතරයකට ගෙන ආ හැකියි. මිනුමට ගැළපෙන එකතු කිරීමක් තෝරාගන්න. නොමැති කියවීමක් අනිවාර්යයෙන් බිංදුවක් නොවෙයි.

Forecasting සඳහා prediction කරන මොහොතේ තිබෙන තොරතුරු පමණක් යොදා lag values සහ rolling statistics සාදන්න. පසුව ලැබෙන අගයන් භාවිත කරන backward fill හෝ interpolation මඟින් අනාගතය කාන්දු විය හැකියි. උදාහරණය කෙටි feature හිඩැස් පමණක් අතීතයෙන් පුරවනවා, නොමැති target අගයන් පුරවන්නේ නැහැ, සහ කාලය අනුව train/test වෙන් කරනවා. අදාළ API සඳහා [pandas time-series මාර්ගෝපදේශය](https://pandas.pydata.org/docs/user_guide/timeseries.html) බලන්න.

[time_series.py බාගත කරන්න](examples/time_series.py)

```python
import numpy as np
import pandas as pd

readings = pd.Series(
    [10, 12, np.nan, 14, 13, 15, 16, 18, 17, 19, 20, 21],
    index=pd.date_range("2026-01-01", periods=12, freq="h", tz="UTC"),
    name="value",
)
# Keep the measured target separate. Do not invent evaluation labels.
features = pd.DataFrame({
    "lag_1": readings.shift(1),
    "past_mean_3": readings.shift(1).rolling(3, min_periods=1).mean(),
    "hour": readings.index.hour,
})
features = features.ffill(limit=1)  # Only past values; fill short gaps.
data = features.join(readings.rename("target")).dropna()
cutoff = pd.Timestamp("2026-01-01 08:00", tz="UTC")
train = data.loc[data.index < cutoff]
test = data.loc[data.index >= cutoff]
print("Training rows:", len(train), "Test rows:", len(test))
print(test.head())
# This is rolling one-step prediction: earlier observations become available.
# For a fixed multi-step forecast, future actual values cannot supply lags.
```

ප්‍රතිදානයේ training පේළි හයක් සහ test පේළි හතරක් ලැබෙනවා. මෙය **rolling one-step** ක්‍රමයක්: ඊළඟ කියවීම පුරෝකථනය කරන විට කලින් කියවීම් දැනටමත් ලැබී තිබෙනවා. මධ්‍යම රාත්‍රියේදීම ඊළඟ දවසම සඳහා forecast එකක් කරනවා නම්, ඒ දවසේ පසුව ලැබෙන සැබෑ අගයන් lag features සඳහා භාවිත කළ නොහැකියි. කේතය ඔබේ forecasting horizon එකට ගළපන්න.

## හඬ, වීඩියෝ සහ අසමතුලිත දත්ත

හඬ සඳහා sample rate, channel ගණන, clipping සහ පටිගත කිරීමේ ගුණාත්මකභාවය පරීක්ෂා කරන්න. Resample කිරීම, channels තේරීම හෝ එකතු කිරීම, අදාළ නොවන නිශ්ශබ්ද කොටස් ඉවත් කිරීම, windows වලට බෙදීම, spectrograms හෝ MFCCs ගණනය කිරීම කළ හැකියි. Speech model එකක් මේ පියවරවලින් සමහරක් දැනටමත් කරනවා විය හැකියි. ස්ථානය හෝ දිශාව වැදගත් නම් channel තොරතුරු ඉවත් නොකරන්න. ශබ්ද ප්‍රබලතාව target එකට වැදගත් නම් එය normalize කර නැති නොකරන්න.

වීඩියෝ සඳහා frames තෝරාගැනීමේ රීතියක් යොදා, ගැළපෙන image transforms භාවිත කර, කාල අනුපිළිවෙළ රකින්න. Audio, frames සහ labels සමමුහුර්තව තබන්න. ළඟ ළඟ clips train සහ test දෙකේම සමානව තිබීම වැළැක්වීමට මුල් පටිගත කිරීම හෝ පුද්ගලයා අනුව දත්ත වෙන් කරන්න.

Class imbalance ද බොහෝ දත්ත වර්ගවල ගැටලුවක්. Class weights, සුදුසු metrics සහ අවශ්‍ය නම් training resampling සලකා බලන්න. Oversampling හෝ synthetic examples සාදනවා නම් ඒවා සෑම fold එකකම training කොටස තුළ පමණක් කරන්න. ලකුණු හොඳ කර පෙන්වීමට held-out test දත්තවල කාණ්ඩ අනුපාත වෙනස් නොකරන්න.

## ප්‍රායෝගික පරීක්ෂණ ලැයිස්තුවක්

- දත්ත වර්ග, ඒකක, හිස් අගයන්, අනුපිටපත්, ලේබල සහ අදාළ කණ්ඩායම් නියෝජනය පරීක්ෂා කරන්න.
- සැබෑ භාවිතයේදී නව දත්ත ලැබෙන ආකාරයට ගැළපෙන split එකක් තෝරාගන්න.
- ඉගෙනගන්නා සැකසුම් training දත්ත මත පමණක් fit කර fitted pipeline එක සුරකින්න.
- Shapes, ranges, අලුත් කාණ්ඩ, හිස් පෙළ, විවෘත නොවන ගොනු සහ නොමැති timestamps පරීක්ෂා කරන්න.
- Test කිරීමේදීත් deployment එකේදීත් එකම inference සැකසුම් භාවිත කරන්න.
- සරල baseline එකක් සමඟ සසඳා සෑම සැකසුමක්ම කාර්යයට උපකාරීදැයි බලන්න.
- Package versions සහ සැකසුම් තේරීම් මොඩලය සමඟ සටහන් කර තබන්න.

Preprocessing මොඩලයේ හැසිරීමේ කොටසක්. එය පරීක්ෂා කළ හැකි, තේරුම් ගත හැකි සහ නැවත භාවිත කළ හැකි කේතයක් ලෙස සකස් කරන්න.
